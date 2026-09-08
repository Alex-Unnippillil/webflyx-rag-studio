import platform
import shutil
import sys
from pathlib import Path

from .config import CACHE_DIR, DATA_DIR, LOG_DIR, MOVIES_PATH
from .security import has_openrouter_key


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
    )


def human_size(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def diagnostics_report(movie_count: int | None = None) -> str:
    try:
        key_ready = has_openrouter_key()
    except Exception:
        key_ready = False

    disk = shutil.disk_usage(DATA_DIR)
    lines = [
        f"Python: {sys.version.split()[0]}",
        f"Operating system: {platform.platform()}",
        f"Architecture: {platform.machine()}",
        "",
        f"Movie dataset: {'Ready' if MOVIES_PATH.exists() else 'Not downloaded yet'}",
        f"Movie records: {movie_count if movie_count is not None else 'Not loaded'}",
        f"OpenRouter key: {'Configured' if key_ready else 'Not configured'}",
        f"Cache usage: {human_size(directory_size(CACHE_DIR))}",
        f"Free disk space: {human_size(disk.free)}",
        "",
        f"Data directory:\n{DATA_DIR}",
        "",
        f"Cache directory:\n{CACHE_DIR}",
        "",
        f"Log directory:\n{LOG_DIR}",
    ]
    return "\n".join(lines)
