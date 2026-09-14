"""FFmpeg audio extraction + faster-whisper transcription.

Runs on a worker thread (see app.main.process_job); nothing here touches the
asyncio event loop.
"""
import subprocess
from pathlib import Path

from app import config


def extract_audio(video_path: Path, audio_path: Path) -> None:
    """Extract 16kHz mono PCM WAV from the input video via ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(audio_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.FFMPEG_TIMEOUT_SEC,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg was not found on PATH. Install it (e.g. `brew install ffmpeg` "
            "or `apt install ffmpeg`) and make sure it's on PATH."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ffmpeg audio extraction timed out after {config.FFMPEG_TIMEOUT_SEC}s"
        ) from exc

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}): {result.stderr[-2000:]}")

    # WAV header alone is 44 bytes; anything at or below that means no audio
    # stream was actually extracted (e.g. a silent/video-only MP4).
    if not audio_path.exists() or audio_path.stat().st_size <= 44:
        raise RuntimeError("No audio track detected in the uploaded video.")


def transcribe(model, audio_path: Path) -> list[dict]:
    """Run faster-whisper with word-level timestamps and flatten to a word list."""
    segments, _info = model.transcribe(str(audio_path), word_timestamps=True)

    words: list[dict] = []
    for segment in segments:
        for w in segment.words or []:
            words.append(
                {
                    "word": w.word.strip(),
                    "start": round(w.start, 2),
                    "end": round(w.end, 2),
                }
            )
    return words
