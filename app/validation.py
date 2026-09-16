import math
import unicodedata
from pathlib import Path

from fastapi import HTTPException, UploadFile

from app import config


def validate_upload(file: UploadFile) -> None:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension '{ext or '(none)'}'. Only {', '.join(sorted(config.ALLOWED_EXTENSIONS))} is accepted.")
    if file.content_type not in config.ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported content type '{file.content_type}'. Expected {', '.join(sorted(config.ALLOWED_CONTENT_TYPES))}.")


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = "".join(char for char in value if char.isalnum() or char.isspace())
    return " ".join(value.split())


def bounded_edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, char in enumerate(left, 1):
        current = [i]
        for j, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char != other)))
        previous = current
    return previous[-1]


def alignment(expected: str, transcript: str) -> dict:
    expected_normalized = normalize_text(expected)
    transcript_normalized = normalize_text(transcript)
    distance = bounded_edit_distance(expected_normalized, transcript_normalized)
    scale = max(len(expected_normalized), len(transcript_normalized), 1)
    return {
        "expected_text": expected,
        "expected_text_normalized": expected_normalized,
        "transcript_normalized": transcript_normalized,
        "edit_distance": distance,
        "normalized_edit_distance": round(distance / scale, 4),
        "match": expected_normalized == transcript_normalized,
        "status": "matched" if expected_normalized == transcript_normalized else "mismatch",
    }


def validate_evidence(result: dict, duration: float) -> None:
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("source duration is invalid")
    transcript = result.get("transcript", "").strip()
    segments = result.get("segments")
    words = result.get("words")
    if not transcript or not isinstance(segments, list) or not segments:
        raise ValueError("ASR returned no non-empty transcript segments")
    if not isinstance(words, list) or not words:
        raise ValueError("ASR returned no word timestamps")
    previous_segment_end = 0.0
    flattened = []
    for segment in segments:
        segment_text, start, end = segment.get("text", "").strip(), segment.get("start"), segment.get("end")
        if not segment_text or not _valid_bounds(start, end, duration) or start < previous_segment_end:
            raise ValueError("ASR segment timestamps or text are unsafe")
        previous_segment_end = end
        segment_words = segment.get("words")
        if not isinstance(segment_words, list) or not segment_words:
            raise ValueError("ASR segment has no word timestamps")
        previous_word_end = start
        segment_word_texts = []
        for word in segment_words:
            word_text, word_start, word_end = word.get("word", "").strip(), word.get("start"), word.get("end")
            if not word_text or not _valid_bounds(word_start, word_end, duration) or word_start < previous_word_end or word_start < start or word_end > end:
                raise ValueError("ASR word timestamps are unsafe")
            previous_word_end = word_end
            segment_word_texts.append(word_text)
            flattened.append(word)
        if normalize_text(segment_text) != normalize_text(" ".join(segment_word_texts)):
            raise ValueError("ASR segment text does not match its word text")
    if len(flattened) != len(words):
        raise ValueError("ASR word list does not match segment words")


def _valid_bounds(start, end, duration: float) -> bool:
    return (isinstance(start, (int, float)) and isinstance(end, (int, float))
            and math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration)


