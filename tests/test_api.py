"""End-to-end self-test against a *running* server.

Not a mocked unit test: this drives the real POST /transcribe -> poll
GET /transcribe/{job_id} flow with tests/fixtures/sample.mp4 (a short real
speech clip), the same way an actual client would. Requires the server
to be already running (see README) since it needs the background worker
loop and a loaded Whisper model.

Run: python tests/test_api.py [base_url]
"""
import sys
import time
from pathlib import Path

import httpx

FIXTURE = Path(__file__).parent / "fixtures" / "sample.mp4"
BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
POLL_TIMEOUT_SEC = 120
POLL_INTERVAL_SEC = 1.5


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    if not FIXTURE.exists():
        fail(f"missing fixture {FIXTURE}")

    with httpx.Client(base_url=BASE_URL, timeout=30) as client:
        print("== health check ==")
        r = client.get("/health")
        if r.status_code != 200:
            fail(f"/health returned {r.status_code}: {r.text}")
        print("OK:", r.json())

        print("== reject non-mp4 upload ==")
        r = client.post("/transcribe", files={"file": ("note.txt", b"hello", "text/plain")})
        if r.status_code != 400:
            fail(f"expected 400 for .txt upload, got {r.status_code}: {r.text}")
        print("OK: rejected with 400 as expected ->", r.json())

        print("== unknown job_id ==")
        r = client.get("/transcribe/does-not-exist")
        if r.status_code != 404:
            fail(f"expected 404 for unknown job_id, got {r.status_code}")
        print("OK: 404 as expected")

        print("== submit real sample.mp4 ==")
        with FIXTURE.open("rb") as f:
            r = client.post(
                "/transcribe",
                files={"file": ("sample.mp4", f, "video/mp4")},
            )
        if r.status_code != 202:
            fail(f"expected 202 from POST /transcribe, got {r.status_code}: {r.text}")
        body = r.json()
        if body.get("status") != "queued" or "job_id" not in body:
            fail(f"unexpected queued response shape: {body}")
        job_id = body["job_id"]
        print("OK: queued ->", body)

        print(f"== polling GET /transcribe/{job_id} ==")
        deadline = time.time() + POLL_TIMEOUT_SEC
        final = None
        while time.time() < deadline:
            r = client.get(f"/transcribe/{job_id}")
            if r.status_code != 200:
                fail(f"GET /transcribe/{job_id} returned {r.status_code}: {r.text}")
            body = r.json()
            print("  status:", body.get("status"))
            if body["status"] in ("completed", "failed"):
                final = body
                break
            time.sleep(POLL_INTERVAL_SEC)

        if final is None:
            fail(f"job did not finish within {POLL_TIMEOUT_SEC}s")

        if final["status"] == "failed":
            fail(f"job failed: {final.get('error')}")

        words = final.get("words")
        if not isinstance(words, list) or not words:
            fail(f"expected non-empty 'words' list, got: {final}")

        for w in words:
            for key in ("word", "start", "end"):
                if key not in w:
                    fail(f"word entry missing '{key}': {w}")
            if not (isinstance(w["start"], (int, float)) and isinstance(w["end"], (int, float))):
                fail(f"non-numeric start/end in word entry: {w}")
            if w["end"] < w["start"]:
                fail(f"word end before start: {w}")

        print(f"OK: completed with {len(words)} words ->", words)

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
