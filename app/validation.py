from pathlib import Path

from fastapi import HTTPException, UploadFile

from app import config


def validate_upload(file: UploadFile) -> None:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext or '(none)'}'. Only "
            f"{', '.join(sorted(config.ALLOWED_EXTENSIONS))} is accepted.",
        )

    if file.content_type not in config.ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type '{file.content_type}'. Expected "
            f"{', '.join(sorted(config.ALLOWED_CONTENT_TYPES))}.",
        )
