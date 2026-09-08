import pytest

from app.security import validate_api_key


def test_rejects_short_api_key():
    with pytest.raises(ValueError):
        validate_api_key("abc")


def test_rejects_whitespace():
    with pytest.raises(ValueError):
        validate_api_key("x" * 20 + " bad")


def test_accepts_reasonable_key():
    key = "x" * 40
    assert validate_api_key(key) == key
