import asyncio
import pytest

from app import job_store
from app.main import get_transcription
from app.validation import alignment, validate_evidence


def valid_result():
    words = [{"word": "Xin", "start": 0.0, "end": 0.4}, {"word": "chào", "start": 0.4, "end": 1.0}]
    return {
        "transcript": "Xin chào",
        "segments": [{"text": "Xin chào", "start": 0.0, "end": 1.0, "words": words}],
        "words": words,
    }


def test_evidence_shape_accepts_source_bounded_segments_and_words():
    result = valid_result()
    validate_evidence(result, 1.0)
    assert {"transcript", "segments", "words"} <= result.keys()
    assert result["segments"][0]["words"] == result["words"]


def test_invalid_timestamp_fails_instead_of_emitting_evidence():
    result = valid_result()
    result["segments"][0]["words"][1]["end"] = 1.1
    with pytest.raises(ValueError, match="unsafe"):
        validate_evidence(result, 1.0)


def test_expected_text_normalization_and_conservative_mismatch():
    matched = alignment("  XIN  chào! ", "Xin chào")
    mismatch = alignment("Xin chào", "Xin bạn")
    assert matched["match"] is True
    assert matched["normalized_edit_distance"] == 0.0
    assert mismatch["match"] is False
    assert mismatch["status"] == "mismatch"
    assert 0.0 < mismatch["normalized_edit_distance"] <= 1.0


def test_completed_polling_response_contains_provenance_contract(monkeypatch):
    job = {
        "job_id": "job1", "status": "completed", "transcript": "Xin chào",
        "segments": valid_result()["segments"], "words": valid_result()["words"],
        "source_sha256": "a" * 64, "duration_sec": 1.0, "language": "vi",
        "model": "small", "provenance": "raw-asr", "authoritative": False,
        "evidence_status": "validated_match",
        "alignment": {"status": "matched", "match": True},
    }
    monkeypatch.setattr(job_store, "get", lambda _: job)
    response = asyncio.run(get_transcription("job1"))
    assert response["status"] == "completed"
    assert response["source_sha256"] == "a" * 64
    assert response["segments"][0]["text"] == "Xin chào"
    assert response["provenance"] == "raw-asr"
    assert response["alignment"]["match"] is True
    assert response["authoritative"] is False
