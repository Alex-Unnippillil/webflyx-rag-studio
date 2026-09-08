import os
from pathlib import Path

from platformdirs import user_cache_dir, user_data_dir, user_log_dir

APP_NAME = "Webflyx RAG Studio"
APP_AUTHOR = "Alex-Unnippillil"

DATA_DIR = Path(user_data_dir(APP_NAME, APP_AUTHOR))
CACHE_DIR = Path(user_cache_dir(APP_NAME, APP_AUTHOR))
LOG_DIR = Path(user_log_dir(APP_NAME, APP_AUTHOR))

for directory in (DATA_DIR, CACHE_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

HF_CACHE = CACHE_DIR / "huggingface"
HF_CACHE.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(HF_CACHE))

MOVIES_PATH = DATA_DIR / "movies.json"

MOVIES_URL = (
    "https://storage.googleapis.com/"
    "qvault-webapp-dynamic-assets/course_assets/"
    "course-rag-movies.json"
)

SEMANTIC_MODEL = "all-MiniLM-L6-v2"
CLIP_MODEL = "clip-ViT-B-32"
RERANK_MODEL = "cross-encoder/ms-marco-TinyBERT-L2-v2"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "openrouter/free"
