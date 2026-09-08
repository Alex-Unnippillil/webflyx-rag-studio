import numpy as np

from app.search_engine import SearchEngine, _top_indices, tokenize


def test_tokenize():
    assert tokenize("Bear in London!") == [
        "bear",
        "in",
        "london",
    ]


def test_top_indices_returns_highest_scores_in_order():
    scores = np.asarray([0.2, 0.9, 0.1, 0.7], dtype=np.float32)
    assert _top_indices(scores, 2).tolist() == [1, 3]


def test_keyword_search_works_without_ml_models():
    movies = [
        {"id": 1, "title": "Bear Story", "description": "A bear in a forest"},
        {"id": 2, "title": "Fast Cars", "description": "A racing movie"},
        {"id": 3, "title": "Another Bear", "description": "A friendly bear adventure"},
    ]
    engine = SearchEngine(movies)
    results = engine.keyword_search("friendly bear", limit=2)
    assert len(results) == 2
    assert results[0]["title"] == "Another Bear"
    assert "bm25_score" in results[0]
