from app.search_engine import tokenize


def test_tokenize():
    assert tokenize("Bear in London!") == [
        "bear",
        "in",
        "london",
    ]
