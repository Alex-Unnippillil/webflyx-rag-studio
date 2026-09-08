import json
import os
from pathlib import Path

import httpx

from .config import MOVIES_PATH, MOVIES_URL

MAX_DATASET_BYTES = 75 * 1024 * 1024


def _validate_dataset(data: object) -> list[dict]:
    if not isinstance(data, dict):
        raise ValueError("Dataset root must be a JSON object.")

    movies = data.get("movies")

    if not isinstance(movies, list) or not movies:
        raise ValueError("Dataset does not contain a valid movies list.")

    clean = []

    for movie in movies:
        if not isinstance(movie, dict):
            continue

        title = movie.get("title")
        description = movie.get("description")

        if not isinstance(title, str) or not isinstance(description, str):
            continue

        clean.append(movie)

    if not clean:
        raise ValueError("No valid movie records were found.")

    return clean


def ensure_dataset() -> Path:
    if MOVIES_PATH.exists():
        try:
            with MOVIES_PATH.open("r", encoding="utf-8") as handle:
                _validate_dataset(json.load(handle))
            return MOVIES_PATH
        except (OSError, ValueError, json.JSONDecodeError):
            MOVIES_PATH.unlink(missing_ok=True)

    with httpx.Client(
        timeout=60.0,
        follow_redirects=True,
    ) as client:
        response = client.get(MOVIES_URL)
        response.raise_for_status()

    content = response.content

    if len(content) > MAX_DATASET_BYTES:
        raise ValueError("Downloaded dataset exceeds the permitted size.")

    data = json.loads(content)
    _validate_dataset(data)

    temporary = MOVIES_PATH.with_suffix(".tmp")

    with temporary.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())

    temporary.replace(MOVIES_PATH)

    return MOVIES_PATH


def load_movies() -> list[dict]:
    path = ensure_dataset()

    with path.open("r", encoding="utf-8") as handle:
        return _validate_dataset(json.load(handle))
