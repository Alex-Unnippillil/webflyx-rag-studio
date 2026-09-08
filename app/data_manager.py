import json
import os
from pathlib import Path

import httpx

from .config import MOVIES_PATH, MOVIES_URL

MAX_DATASET_BYTES = 75 * 1024 * 1024


def validate_dataset(data: object) -> list[dict]:
    if not isinstance(data, dict):
        raise ValueError("Dataset root must be a JSON object.")

    movies = data.get("movies")
    if not isinstance(movies, list) or not movies:
        raise ValueError("Dataset does not contain a valid non-empty movies list.")

    clean = []
    for movie in movies:
        if not isinstance(movie, dict):
            continue

        title = movie.get("title")
        description = movie.get("description")
        if not isinstance(title, str) or not title.strip():
            continue
        if not isinstance(description, str):
            continue

        clean.append(movie)

    if not clean:
        raise ValueError("No valid movie records were found.")
    return clean


def _write_atomically(content: bytes) -> None:
    temporary = MOVIES_PATH.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(MOVIES_PATH)


def _validated_bytes(content: bytes) -> list[dict]:
    if len(content) > MAX_DATASET_BYTES:
        raise ValueError("Dataset exceeds the permitted 75 MB size limit.")
    try:
        data = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Dataset is not valid UTF-8 JSON.") from exc
    return validate_dataset(data)


def ensure_dataset() -> Path:
    if MOVIES_PATH.exists():
        try:
            with MOVIES_PATH.open("r", encoding="utf-8") as handle:
                validate_dataset(json.load(handle))
            return MOVIES_PATH
        except (OSError, ValueError, json.JSONDecodeError):
            MOVIES_PATH.unlink(missing_ok=True)

    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        response = client.get(MOVIES_URL)
        response.raise_for_status()
        content = response.content

    _validated_bytes(content)
    _write_atomically(content)
    return MOVIES_PATH


def import_dataset(source: str | Path) -> int:
    path = Path(source)
    if not path.is_file():
        raise ValueError("Choose an existing JSON dataset file.")

    content = path.read_bytes()
    movies = _validated_bytes(content)
    _write_atomically(content)
    return len(movies)


def reset_dataset() -> None:
    MOVIES_PATH.unlink(missing_ok=True)


def load_movies() -> list[dict]:
    path = ensure_dataset()
    with path.open("r", encoding="utf-8") as handle:
        return validate_dataset(json.load(handle))
