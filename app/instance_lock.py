"""Single-instance guard.

The worker design in app.main serializes CPU-heavy transcription jobs through
one asyncio.Queue and a ThreadPoolExecutor(max_workers=1). That guarantee only
holds if exactly one process runs the app. Starting a second instance (or
`uvicorn --workers N` with N > 1) would spin up N independent model copies
competing for the same CPU cores -- the exact thrashing scenario the serial
design exists to avoid. This module fails loudly instead of allowing that.
"""
import sys
from pathlib import Path

# Keep the lock's file descriptor referenced for the process lifetime; closing
# it (e.g. via garbage collection) would release the OS-level lock.
_held_fds = []


def acquire_singleton_lock(path: Path) -> None:
    if sys.platform == "win32":
        print(
            "[voice2text] WARNING: single-instance lock is not enforced on Windows. "
            "Make sure only one process (and `--workers 1`) runs this server.",
            file=sys.stderr,
        )
        return

    import fcntl

    fd = open(path, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        fd.close()
        raise RuntimeError(
            "Another instance of this server appears to be running already "
            "(lock held on " + str(path) + "). This service intentionally "
            "serializes transcription jobs in a single process -- run with a "
            "single worker, e.g. `uvicorn app.main:app --workers 1`, and stop "
            "any other running instance before starting a new one."
        ) from exc

    _held_fds.append(fd)
