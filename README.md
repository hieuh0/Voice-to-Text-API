# Voice-to-Text Timestamp API

MVP FastAPI service that transcribes an uploaded MP4 into word-level timestamps
using a local Whisper model, without blocking the HTTP request on the
transcription itself.

## Architecture

```
POST /transcribe (MP4 upload)
  -> validate extension/content-type/size
  -> save to tmp/, create job (status=queued), enqueue job_id
  -> respond immediately: {"job_id", "status": "queued"}

Background worker (single thread, one job at a time)
  MP4 -> ffmpeg (extract 16kHz mono WAV)
      -> faster-whisper (word-level timestamps)
      -> write result to jobs/{job_id}.json (status=completed|failed)
      -> delete temp MP4 + WAV

GET /transcribe/{job_id}
  -> queued / processing: {"job_id", "status"}
  -> completed: {"job_id", "status", "words": [{"word","start","end"}, ...]}
  -> failed: {"job_id", "status", "error"}

GET /health   -> liveness: {"status": "ok"} whenever the process is up
GET /status   -> readiness: 200 + {"ready": true, ...} once the model is
                 loaded, ffmpeg is on PATH, and jobs/tmp dirs are writable;
                 503 + {"ready": false, ...} otherwise
```

No database: job state is one JSON file per job under `jobs/`. Temp media
lives under `tmp/` and is deleted as soon as a job finishes (success or
failure).

### Why these choices

- **faster-whisper** (CTranslate2 backend), not the original `openai-whisper`
  package: int8 quantization gives noticeably lower RAM use and faster
  CPU-only inference, and it supports word-level timestamps natively
  (`word_timestamps=True`, DTW-based) with no separate alignment model.
- **Model size: `small`** by default (`WHISPER_MODEL` env var). `tiny`/`base`
  are faster but too inaccurate for word-timestamp use; `medium`/`large-v3`
  are noticeably slower on CPU-only hardware for an MVP. `small` + int8 is the
  standard sweet spot for CPU-only faster-whisper deployments. Drop to
  `base`/`tiny` if a machine is slower than expected, or raise to `medium` if
  you have time budget and need better accuracy.
- **One background worker, one job at a time**: transcription is CPU-heavy;
  running several jobs concurrently on a mobile CPU thrashes cache/RAM and
  slows every job down. A single `asyncio.Queue` + `ThreadPoolExecutor(max_workers=1)`
  keeps each job's latency predictable and keeps the event loop (and thus
  `GET /transcribe/{job_id}` polling) responsive while a job runs.
- **The Whisper model is loaded once at startup**, not per job — reloading it
  per request would pay model-load latency on every single job.
- **`WHISPER_CPU_THREADS`** is set explicitly (default: `logical_cpus // 2 - 1`,
  e.g. 5 on a 6-core/12-thread Ryzen 5 Mobile) rather than left at the library
  default, so one running job doesn't consume every core and stall the API.
- **Single-instance guard**: the whole "serialize jobs" design only holds if
  exactly one process runs the app. The server takes an OS-level file lock
  (`.instance.lock`) at startup and refuses to start a second instance, so
  running `uvicorn --workers 2` (or two copies of the server) fails loudly
  instead of silently causing N model copies to fight over the same CPU cores.
- **File-based job store**, not pure in-memory: survives a server restart
  mid-job (or reads back state after a crash) at negligible extra complexity
  over an in-memory dict, while staying dependency-free.
- **Bounded queue (`MAX_QUEUED_JOBS`, default 5)**: since only one job runs at
  a time, an unbounded backlog would let clients pile up uploads faster than
  they can be processed, filling `tmp/` with saved videos waiting their turn.
  Once the backlog is full, new uploads get `429` immediately instead of being
  accepted and left to wait indefinitely.
  Total in-flight capacity is therefore 1 processing + `MAX_QUEUED_JOBS`
  waiting (6 by default). A `429` means "no slot right now" — the upload is
  rejected outright (nothing is queued, no video is kept on disk), not
  silently held. The client must retry `POST /transcribe` later; only once a
  slot has actually freed up does the request succeed (`202` +
  `{"status": "queued"}`).

### Known gaps (MVP scope, not handled)

- Job files under `jobs/` have no TTL/expiry and accumulate indefinitely —
  fine for an MVP; add a cleanup sweep if this runs long-lived.
- The single-instance lock (`fcntl.flock`) is POSIX-only; on Windows it just
  prints a warning instead of enforcing — don't run more than one process
  there manually.
- No auth, no horizontal scaling, no LLM post-processing/punctuation cleanup,
  no streaming/real-time transcription, no speaker diarization. Out of scope
  by design for this MVP.

## Requirements

- Python 3.10+ (3.12 recommended)
- [ffmpeg](https://ffmpeg.org/) on `PATH` (`brew install ffmpeg` / `apt install ffmpeg`)
- ~2GB free disk for the `small` model weights (downloaded once, cached under
  `~/.cache/huggingface`) plus headroom for temp video/audio files

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # optional: tune WHISPER_MODEL, thread count, etc.
```

## Run

```bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

`--workers 1` is required — see "Single-instance guard" above. The first
request that loads the model will download `small` weights from Hugging Face
(one-time, then cached).

## Test

### Automated end-to-end check

With the server running, in another terminal:

```bash
source .venv/bin/activate
python tests/test_api.py
```

This drives the real API against `tests/fixtures/sample.mp4` (a short real
speech clip, "Where are you going today") through the whole
queued -> processing -> completed flow, and checks upload validation (400 for
a non-MP4 file) and unknown-job handling (404) along the way.

### Manual curl walkthrough

```bash
# 1. Submit a video
curl -X POST http://127.0.0.1:8000/transcribe \
  -F "file=@tests/fixtures/sample.mp4;type=video/mp4"
# -> {"job_id":"<id>","status":"queued"}

# 2. Poll for the result
curl http://127.0.0.1:8000/transcribe/<id>
# -> {"job_id":"<id>","status":"processing"}
# ... then ...
# -> {"job_id":"<id>","status":"completed","words":[{"word":"Where","start":0.0,"end":0.16}, ...]}

# 3. Check readiness (model loaded, ffmpeg on PATH, dirs writable, worker alive)
curl http://127.0.0.1:8000/status
# -> 200 {"ready":true,"checks":{"model_loaded":true,"ffmpeg_available":true,
#          "jobs_dir_writable":true,"tmp_dir_writable":true,"worker_alive":true},
#         "queue":{"processing":0,"queued":0,"max_queued":5}}
# -> 503 with the same shape (ready:false) if a check fails, e.g. ffmpeg missing
#    or the background worker thread died
```

## Configuration (env vars, see `.env.example`)

| Variable | Default | Notes |
|---|---|---|
| `WHISPER_MODEL` | `small` | `tiny`\|`base`\|`small`\|`medium`\|`large-v3` |
| `WHISPER_DEVICE` | `cpu` | |
| `WHISPER_COMPUTE_TYPE` | `int8` | int8 quantization for CPU speed/RAM |
| `WHISPER_CPU_THREADS` | `logical_cpus // 2 - 1` | tune per machine |
| `MAX_UPLOAD_SIZE_MB` | `500` | rejects larger uploads with 413 |
| `FFMPEG_TIMEOUT_SEC` | `600` | kills a stuck ffmpeg extraction |
| `MAX_QUEUED_JOBS` | `5` | jobs waiting (excludes the one processing); rejects new uploads with 429 once full |
| `JOBS_DIR` | `./jobs` | job status JSON files |
| `TMP_DIR` | `./tmp` | temp video/audio, deleted per job |

## Extending

- Swap the job store (`app/job_store.py`) for Redis/SQLite without touching
  the API layer — it's a 4-function interface (`create`/`get`/`update`).
- Swap the queue (`app/main.py`) for a real task queue (Celery/RQ) if you
  outgrow single-process serialization; the pipeline functions in
  `app/pipeline.py` are already decoupled from FastAPI/asyncio.
- Add languages/translation by passing `language=` / `task="translate"` to
  `model.transcribe(...)` in `app/pipeline.py`.
