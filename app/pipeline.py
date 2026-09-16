"""FFmpeg media handling and faster-whisper transcription.

Runs on the single worker thread; nothing here touches the asyncio event loop.
"""
import json
import math
import subprocess
from pathlib import Path

from app import config


def source_duration(video_path: Path) -> float:
    """Read a finite media duration from ffprobe, rejecting malformed media."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(video_path)],
            capture_output=True, text=True, timeout=config.FFMPEG_TIMEOUT_SEC,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe was not found on PATH; install ffmpeg.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffprobe timed out after {config.FFMPEG_TIMEOUT_SEC}s") from exc
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed (exit {result.returncode}): {result.stderr[-2000:]}")
    try:
        duration = float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("Could not determine a valid source duration.") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("Source media duration is not a positive finite value.")
    return round(duration, 3)


def extract_audio(video_path: Path, audio_path: Path) -> None:
    """Extract 16kHz mono PCM WAV from the input video via ffmpeg."""
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
           "-ar", "16000", "-ac", "1", str(audio_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=config.FFMPEG_TIMEOUT_SEC)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg was not found on PATH. Install it and make sure it's on PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg audio extraction timed out after {config.FFMPEG_TIMEOUT_SEC}s") from exc
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}): {result.stderr[-2000:]}")
    if not audio_path.exists() or audio_path.stat().st_size <= 44:
        raise RuntimeError("No audio track detected in the uploaded video.")


def transcribe(model, audio_path: Path) -> dict:
    """Return raw segment and word evidence from faster-whisper."""
    kwargs = {"language": config.WHISPER_LANGUAGE, "word_timestamps": True,
              "beam_size": config.WHISPER_BEAM_SIZE, "vad_filter": config.WHISPER_VAD_FILTER}
    segments_iter, info = model.transcribe(str(audio_path), **kwargs)
    segments = []
    words = []
    for segment in segments_iter:
        segment_words = []
        for word in segment.words or []:
            item = {"word": (word.word or "").strip(), "start": word.start, "end": word.end}
            segment_words.append(item)
            words.append(item)
        segments.append({"text": (segment.text or "").strip(), "start": segment.start,
                         "end": segment.end, "words": segment_words})
    model_version = getattr(info, "model_version", None) or getattr(model, "model_version", None)
    return {"transcript": " ".join(s["text"] for s in segments).strip(), "segments": segments,
            "words": words, "model_version": model_version}
