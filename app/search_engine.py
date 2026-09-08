import hashlib
import json
import re
import threading
from pathlib import Path

import numpy as np
from PIL import Image
from rank_bm25 import BM25Okapi

from .config import CACHE_DIR, CLIP_MODEL, RERANK_MODEL, SEMANTIC_MODEL

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class SearchEngine:
    def __init__(self, movies: list[dict]) -> None:
        self.movies = movies

        self.texts = [
            f"{movie.get('title', '')}: {movie.get('description', '')}"
            for movie in movies
        ]

        self.tokens = [tokenize(text) for text in self.texts]
        self.bm25 = BM25Okapi(self.tokens)

        fingerprint_data = [
            (
                movie.get("id"),
                movie.get("title"),
                movie.get("description"),
            )
            for movie in movies
        ]

        encoded = json.dumps(
            fingerprint_data,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        self.dataset_hash = hashlib.sha256(encoded).hexdigest()[:16]

        self._semantic_model = None
        self._semantic_embeddings = None
        self._clip_model = None
        self._clip_embeddings = None
        self._cross_encoder = None

        self._lock = threading.RLock()

    def _embedding_cache(self, prefix: str) -> Path:
        return CACHE_DIR / f"{prefix}-{self.dataset_hash}.npy"

    def _encode_or_load(
        self,
        model,
        cache_name: str,
    ) -> np.ndarray:
        cache = self._embedding_cache(cache_name)

        if cache.exists():
            try:
                embeddings = np.load(cache)

                if embeddings.shape[0] == len(self.movies):
                    return embeddings
            except (ValueError, OSError):
                cache.unlink(missing_ok=True)

        embeddings = model.encode(
            self.texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        temporary = cache.with_suffix(".tmp")

        with temporary.open("wb") as handle:
            np.save(handle, embeddings)

        temporary.replace(cache)

        return embeddings

    def _get_semantic(self):
        with self._lock:
            if self._semantic_model is None:
                from sentence_transformers import SentenceTransformer

                self._semantic_model = SentenceTransformer(
                    SEMANTIC_MODEL,
                    device="cpu",
                )

            if self._semantic_embeddings is None:
                self._semantic_embeddings = self._encode_or_load(
                    self._semantic_model,
                    "semantic",
                )

        return self._semantic_model, self._semantic_embeddings

    def _get_clip(self):
        with self._lock:
            if self._clip_model is None:
                from sentence_transformers import SentenceTransformer

                self._clip_model = SentenceTransformer(
                    CLIP_MODEL,
                    device="cpu",
                )

            if self._clip_embeddings is None:
                self._clip_embeddings = self._encode_or_load(
                    self._clip_model,
                    "clip-text",
                )

        return self._clip_model, self._clip_embeddings

    def _get_cross_encoder(self):
        with self._lock:
            if self._cross_encoder is None:
                from sentence_transformers import CrossEncoder

                self._cross_encoder = CrossEncoder(
                    RERANK_MODEL,
                    device="cpu",
                )

        return self._cross_encoder

    def _result(self, index: int, **extra) -> dict:
        movie = self.movies[index]

        return {
            "id": movie.get("id"),
            "title": movie.get("title", ""),
            "description": movie.get("description", ""),
            **extra,
        }

    def text_search(
        self,
        query: str,
        limit: int = 10,
        rerank: bool = True,
    ) -> list[dict]:
        query = query.strip()

        if not query:
            return []

        candidate_limit = min(
            max(limit * 10, 100),
            len(self.movies),
        )

        bm25_scores = np.asarray(
            self.bm25.get_scores(tokenize(query)),
            dtype=np.float32,
        )

        bm25_order = np.argsort(bm25_scores)[::-1][:candidate_limit]

        model, semantic_embeddings = self._get_semantic()

        query_embedding = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        semantic_scores = semantic_embeddings @ query_embedding
        semantic_order = np.argsort(semantic_scores)[::-1][:candidate_limit]

        scores: dict[int, float] = {}
        ranks: dict[int, dict] = {}

        for rank, index in enumerate(bm25_order, start=1):
            idx = int(index)
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (60 + rank)
            ranks.setdefault(idx, {})["bm25_rank"] = rank

        for rank, index in enumerate(semantic_order, start=1):
            idx = int(index)
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (60 + rank)
            ranks.setdefault(idx, {})["semantic_rank"] = rank

        ordered = sorted(
            scores,
            key=scores.get,
            reverse=True,
        )

        pre_rerank_limit = min(
            max(limit * 5, 25),
            len(ordered),
        )

        candidates = [
            self._result(
                idx,
                rrf_score=float(scores[idx]),
                **ranks[idx],
            )
            for idx in ordered[:pre_rerank_limit]
        ]

        if not rerank or not candidates:
            return candidates[:limit]

        cross_encoder = self._get_cross_encoder()

        pairs = [
            [
                query,
                f"{item['title']} - {item['description']}",
            ]
            for item in candidates
        ]

        cross_scores = cross_encoder.predict(pairs)

        for item, score in zip(candidates, cross_scores, strict=True):
            item["rerank_score"] = float(score)

        candidates.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return candidates[:limit]

    def image_search(
        self,
        image_path: str,
        limit: int = 10,
    ) -> list[dict]:
        model, text_embeddings = self._get_clip()

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            image_embedding = model.encode(
                [image],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0]

        similarities = text_embeddings @ image_embedding

        order = np.argsort(similarities)[::-1][:limit]

        return [
            self._result(
                int(index),
                similarity=float(similarities[index]),
            )
            for index in order
        ]
