import asyncio
import logging
import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app import config, job_store
from app.instance_lock import acquire_singleton_lock
from app.pipeline import extract_audio, transcribe
from app.validation import validate_upload

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("voice2text")


def process_job(app: FastAPI, job_id: str) -> None:
    """Runs on the single worker thread: one job at a time, CPU-bound."""
    job = job_store.get(job_id)
    if job is None:
        logger.warning("job %s vanished before processing", job_id)
        return

    video_path = Path(job["video_path"])
    audio_path = video_path.with_suffix(".wav")

    # Set/cleared from this worker thread, read from the event loop thread by
    # GET /status -- a plain attribute assignment is GIL-atomic, so /status
    # never sees a torn value, only a possibly-one-tick-stale one (fine for a
    # status indicator). Set inside the try (not before it) so a failure in
    # job_store.update itself still reaches `finally` and clears the flag,
    # instead of leaving processing_job_id stuck for the rest of the process.
    try:
        app.state.processing_job_id = job_id
        job_store.update(job_id, status="processing")
        extract_audio(video_path, audio_path)
        words = transcribe(app.state.model, audio_path)
        job_store.update(job_id, status="completed", words=words)
    except Exception as exc:  # noqa: BLE001 - convert any pipeline failure into a job status
        logger.exception("job %s failed", job_id)
        job_store.update(job_id, status="failed", error=str(exc))
    finally:
        app.state.processing_job_id = None
        video_path.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)


async def worker_loop(app: FastAPI) -> None:
    loop = asyncio.get_running_loop()
    while True:
        job_id = await app.state.queue.get()
        try:
            await loop.run_in_executor(app.state.executor, process_job, app, job_id)
        finally:
            app.state.queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    acquire_singleton_lock(config.INSTANCE_LOCK_PATH)
    config.JOBS_DIR.mkdir(parents=True, exist_ok=True)
    config.TMP_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Loading Whisper model '%s' (device=%s, compute_type=%s, cpu_threads=%s)...",
        config.WHISPER_MODEL,
        config.WHISPER_DEVICE,
        config.WHISPER_COMPUTE_TYPE,
        config.WHISPER_CPU_THREADS,
    )
    from faster_whisper import WhisperModel

    # Loaded once here, reused by every job -- do not instantiate WhisperModel
    # per-request, that would pay model-load cost on every job.
    app.state.model = WhisperModel(
        config.WHISPER_MODEL,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
        cpu_threads=config.WHISPER_CPU_THREADS,
    )
    logger.info("Whisper model loaded.")

    app.state.queue: asyncio.Queue = asyncio.Queue(maxsize=config.MAX_QUEUED_JOBS)
    # max_workers=1: exactly one transcription runs at a time, matching the
    # single-instance guard above -- this process never oversubscribes CPU.
    app.state.executor = ThreadPoolExecutor(max_workers=1)
    app.state.processing_job_id = None
    app.state.worker_task = asyncio.create_task(worker_loop(app))

    try:
        yield
    finally:
        app.state.worker_task.cancel()
        app.state.executor.shutdown(wait=False)


app = FastAPI(title="Voice-to-Text Timestamp API", lifespan=lifespan)


async def _save_upload(file: UploadFile, dest: Path, max_bytes: int) -> int:
    size = 0
    try:
        with dest.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds max upload size of {max_bytes // (1024 * 1024)}MB",
                    )
                out.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    return size


@app.post("/transcribe", status_code=202)
async def create_transcription(file: UploadFile = File(...)):
    validate_upload(file)

    # Fail fast before writing to disk when the backlog is clearly full. The
    # real enforcement is the put_nowait() below (atomic w.r.t. the event
    # loop); this is just an optimization to skip the wasted upload write.
    if app.state.queue.qsize() >= config.MAX_QUEUED_JOBS:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Server busy: {config.MAX_QUEUED_JOBS} job(s) already queued. "
                "Try again shortly."
            ),
        )

    job_id = uuid.uuid4().hex
    video_path = config.TMP_DIR / f"{job_id}.mp4"
    await _save_upload(file, video_path, config.MAX_UPLOAD_BYTES)

    job_store.create(job_id, video_path=str(video_path))
    try:
        app.state.queue.put_nowait(job_id)
    except asyncio.QueueFull:
        video_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=429,
            detail=(
                f"Server busy: {config.MAX_QUEUED_JOBS} job(s) already queued. "
                "Try again shortly."
            ),
        )

    return {"job_id": job_id, "status": "queued"}


@app.get("/transcribe/{job_id}")
async def get_transcription(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job_id not found")

    response = {"job_id": job_id, "status": job["status"]}
    if job["status"] == "completed":
        response["words"] = job.get("words", [])
    elif job["status"] == "failed":
        response["error"] = job.get("error", "unknown error")
    return response


@app.get("/health")
async def health():
    return {"status": "ok"}


def _readiness_checks() -> dict:
    # getattr(..., None): defensive, not because these are ever unset while
    # uvicorn is actually serving requests (lifespan sets them before startup
    # completes) -- it's so this never 500s with an AttributeError if the app
    # is exercised without its lifespan (e.g. a TestClient used without the
    # `with` context manager), keeping the documented 200/503 contract honest.
    worker_task = getattr(app.state, "worker_task", None)
    return {
        "model_loaded": getattr(app.state, "model", None) is not None,
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
        "jobs_dir_writable": os.access(config.JOBS_DIR, os.W_OK),
        "tmp_dir_writable": os.access(config.TMP_DIR, os.W_OK),
        "worker_alive": worker_task is not None and not worker_task.done(),
    }


@app.get("/status", responses={503: {"description": "Not ready"}})
async def status():
    """Readiness (not just liveness): are dependencies actually usable."""
    checks = _readiness_checks()
    ready = all(checks.values())
    queue: asyncio.Queue = getattr(app.state, "queue", None)
    body = {
        "ready": ready,
        "checks": checks,
        "queue": {
            "processing": 1 if getattr(app.state, "processing_job_id", None) else 0,
            "queued": queue.qsize() if queue is not None else 0,
            "max_queued": config.MAX_QUEUED_JOBS,
        },
    }
    return JSONResponse(status_code=200 if ready else 503, content=body)
