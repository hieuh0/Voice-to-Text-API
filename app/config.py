import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# --- Whisper ---
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# Heuristic default: assume half of logical CPUs are physical cores (SMT/hyperthreading),
# then leave one physical core free for the event loop / OS so job polling stays responsive
# while a transcription is running. Override explicitly per machine via WHISPER_CPU_THREADS.
_logical_cpus = os.cpu_count() or 4
_default_cpu_threads = max(1, _logical_cpus // 2 - 1)
WHISPER_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", str(_default_cpu_threads)))

# --- Upload validation ---
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4"}
ALLOWED_CONTENT_TYPES = {"video/mp4"}

# Jobs waiting in the queue (not counting the one currently processing).
# Bounds worst-case temp disk usage to roughly (MAX_QUEUED_JOBS + 1) *
# MAX_UPLOAD_SIZE_MB, and keeps backlog wait times bounded on a single-worker
# server. New uploads are rejected with 429 once this many jobs are queued.
MAX_QUEUED_JOBS = int(os.getenv("MAX_QUEUED_JOBS", "5"))

# --- Storage ---
JOBS_DIR = Path(os.getenv("JOBS_DIR", str(BASE_DIR / "jobs")))
TMP_DIR = Path(os.getenv("TMP_DIR", str(BASE_DIR / "tmp")))

# --- Pipeline ---
FFMPEG_TIMEOUT_SEC = int(os.getenv("FFMPEG_TIMEOUT_SEC", "600"))

# --- Single-instance guard ---
# This service serializes CPU-heavy jobs in one asyncio.Queue + one worker thread.
# Running multiple processes/workers against the same model defeats that design
# (N processes thrash the same CPU cores). This lock makes a second instance fail loudly.
INSTANCE_LOCK_PATH = BASE_DIR / ".instance.lock"
