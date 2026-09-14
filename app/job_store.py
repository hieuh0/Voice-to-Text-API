"""File-based job store.

No database for this MVP: each job is one JSON file under JOBS_DIR, written
atomically (write to a temp file, then os.replace) so a reader never sees a
half-written file. Job JSON may carry internal fields (e.g. video_path) that
API responses filter out; see app.main for the response shape.

Known gap (documented, not handled): job files have no TTL/expiry and will
accumulate indefinitely. Fine for an MVP; add a cleanup sweep if this runs
long-lived.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app import config

config.JOBS_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(job_id: str) -> Path:
    return config.JOBS_DIR / f"{job_id}.json"


def _write(job_id: str, data: dict) -> None:
    final_path = _path(job_id)
    tmp_path = final_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp_path.replace(final_path)


def create(job_id: str, **fields: Any) -> dict:
    data = {
        "job_id": job_id,
        "status": "queued",
        "created_at": _now(),
        "updated_at": _now(),
        **fields,
    }
    _write(job_id, data)
    return data


def get(job_id: str) -> Optional[dict]:
    path = _path(job_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def update(job_id: str, **fields: Any) -> dict:
    data = get(job_id) or {"job_id": job_id}
    data.update(fields)
    data["updated_at"] = _now()
    _write(job_id, data)
    return data
