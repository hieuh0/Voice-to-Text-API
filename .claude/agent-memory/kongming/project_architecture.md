---
name: project-architecture
description: Voice-to-Text Timestamp API MVP architecture decisions and hardware constraints
metadata:
  type: project
---

Building "Voice-to-Text Timestamp API" — FastAPI MVP, CPU-only, no DB, no LLM.

**Hardware constraint:** Ryzen 5 Mobile (6 cores/12 threads), 16GB RAM, no GPU/CUDA. All model sizing and concurrency decisions must respect this ceiling.

**Fixed flow (not up for debate, confirmed 2026-09-14):**
- `POST /transcribe`: accepts MP4, validates, returns `{job_id, status:"queued"}` immediately (non-blocking).
- Background pipeline: MP4 -> FFmpeg audio extract -> faster-whisper (CTranslate2) local transcription -> word-level timestamps -> persist -> status "completed".
- `GET /transcribe/{job_id}`: returns queued/processing/completed(+words)/failed(+error).
- Statuses: queued / processing / completed / failed.
- Auto-cleanup of temp video/audio after each job.

**Architecture decisions confirmed as sound (kongming review, 2026-09-14):**
- faster-whisper (CTranslate2), not openai-whisper — int8 CPU quantization + native word_timestamps=True via DTW/cross-attention, no separate alignment model (unlike WhisperX) needed.
- Model default "small" (env var `WHISPER_MODEL`) — right sweet spot for CPU-only; tiny/base sacrifice word-timestamp accuracy, medium/large-v3 too slow on this hardware.
- Serial in-process worker: `asyncio.Queue` + `loop.run_in_executor(ThreadPoolExecutor(max_workers=1))`, not Celery/RQ/Redis (too much infra, DB forbidden) and not FastAPI BackgroundTasks directly (no queue/serialization). One job at a time avoids CPU/RAM thrashing on this hardware. Accepted non-goal: not horizontally scalable.
- File-based job store (`jobs/{job_id}.json`) over pure in-memory dict — survives restart, debuggable, negligible extra complexity, still zero-dependency.
- Upload validation: extension + content-type check, max size cap (e.g. 500MB) to bound worst-case RAM/CPU time.
- Cleanup: delete temp MP4 + extracted WAV in try/finally after job reaches completed/failed.

**Risks flagged during architecture review (must address in implementation):**
1. Load the faster-whisper `WhisperModel` **once** as a module-level singleton at app startup, reused by the single worker thread — do NOT instantiate it per job (adds real per-job latency + RAM churn). Confirmed via research: this is the standard faster-whisper/FastAPI production pattern.
2. Explicitly set CTranslate2 `cpu_threads` — do not leave at library default, and do not set it to all 12 logical threads. Since only one job runs at a time, that one job's intra-op compute will otherwise starve the asyncio event loop thread (blocking GET /transcribe polling responsiveness). Recommend tuning to roughly physical-core count minus 1 (~5) and benchmarking, not guessing.
3. Must run as a **single process** (`uvicorn --workers 1`, no gunicorn multi-worker). The whole "serialize jobs to avoid CPU thrashing" design is process-local (asyncio.Queue + ThreadPoolExecutor(max_workers=1) live in one process). Multi-worker deployment would duplicate the loaded model (wasted RAM) and let multiple processes compete for the same 6 physical cores simultaneously, defeating the entire point of serialization. Document this constraint prominently (README + maybe a startup assertion).
4. Minor/deferred: file-based job store (`jobs/*.json`) has no TTL/pruning — will accumulate forever. Not urgent for MVP but worth a one-line README note or a simple age-based cleanup later.

**Why these matter:** all four are gaps in an otherwise sound architecture that the user (kimberly.mitchell@beyondexpectationsmiami.com) is executing solo on CPU-only mobile hardware — small inefficiencies (redundant model loads, event-loop starvation, accidental multi-worker deployment) would be disproportionately costly on this hardware relative to a cloud/GPU deployment.
