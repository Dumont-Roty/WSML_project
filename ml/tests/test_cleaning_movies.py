import json
from pathlib import Path

import polars as pl

from ml.src import cleaning_movies as cm


def test_read_movies_json_and_dedup(tmp_path):
    data = [
        {"url": "u1", "title": "A", "year": 2020},
        {"url": "u1", "title": "A dupe", "year": 2021},
        {"url": "u2", "title": "B", "year": 2022},
    ]
    src = tmp_path / "data.json"
    src.write_text(json.dumps(data), encoding="utf-8")

    df = cm.read_movies_json(src)
    assert isinstance(df, pl.DataFrame)
    assert df.shape[0] == 3

    dedup = cm.elimine_doublons(df)
    assert dedup.shape[0] == 2  # duplicate url removed


def test_construit_features_numeriques_counts_and_fill():
    df = pl.DataFrame(
        {
            "url": ["u1"],
            "title": ["A"],
            "year": [2020],
            "duration": [None],
            "directors": [["Alice", "Bob"]],
            "genres": [[]],
            "rating": [4.5],
        }
    )

    out = cm.construit_features_numeriques(df)
    assert set(out.columns) == {"url", "title", "year", "duration", "rating", "directors_count", "genres_count"}
    # duration null filled to 0, counts computed even for empty list
    assert out.select("duration")[0, 0] == 0
    assert out.select("directors_count")[0, 0] == 2
    assert out.select("genres_count")[0, 0] == 0


def test_main_writes_parquet(tmp_path, monkeypatch):
    data = [
        {"url": "u1", "title": "A", "year": 2020, "rating": 4.0},
        {"url": "u1", "title": "A dupe", "year": 2020, "rating": 4.0},
    ]
    src = tmp_path / "data.json"
    src.write_text(json.dumps(data), encoding="utf-8")
    out = tmp_path / "out.parquet"

    rc = cm.main(["--input", str(src), "--output", str(out)])
    assert rc == 0
    assert out.exists()

    df = pl.read_parquet(out)
    # Dedup kept one row
    assert df.shape[0] == 1
    assert "rating" in df.columns
