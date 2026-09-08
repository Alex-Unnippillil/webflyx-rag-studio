import json

import pytest

from app.data_manager import validate_dataset


def test_validate_dataset_accepts_movie_records():
    data = {
        "movies": [
            {"id": 1, "title": "Example", "description": "A movie"},
        ]
    }
    movies = validate_dataset(data)
    assert len(movies) == 1
    assert movies[0]["title"] == "Example"


def test_validate_dataset_rejects_wrong_root():
    with pytest.raises(ValueError):
        validate_dataset([])


def test_validate_dataset_filters_invalid_rows():
    data = {
        "movies": [
            {"id": 1, "title": "", "description": "invalid title"},
            {"id": 2, "title": "Good", "description": "valid"},
        ]
    }
    movies = validate_dataset(data)
    assert [movie["id"] for movie in movies] == [2]


def test_dataset_shape_is_json_serializable():
    data = {"movies": [{"id": 1, "title": "Good", "description": "valid"}]}
    assert json.loads(json.dumps(data)) == data
