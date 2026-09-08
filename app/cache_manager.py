import shutil

from .config import CACHE_DIR


def clear_application_cache() -> None:
    if CACHE_DIR.exists():
        for item in CACHE_DIR.iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
