import sys
from pathlib import Path

import pandas as pd

from ml.src import prepare_dataset


def test_prepare_dataset_main_splits_and_filters(tmp_path, monkeypatch):
    data = pd.DataFrame(
        {
            "rating": [4.0, None, 3.5, 2.0],
            "year": [2020, 2021, 2022, 2023],
            "duration": [100, 90, 80, 70],
            "title": ["a", "b", "c", "d"],
        }
    )
    data_path = tmp_path / "cleaned.parquet"
    data.to_parquet(data_path, index=False)

    out_train = tmp_path / "train.parquet"
    out_test = tmp_path / "test.parquet"

    argv = [
        "prepare_dataset",
        "--data",
        str(data_path),
        "--out-train",
        str(out_train),
        "--out-test",
        str(out_test),
        "--test-size",
        "0.5",
        "--seed",
        "0",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    rc = prepare_dataset.main()
    assert rc == 0
    assert out_train.exists()
    assert out_test.exists()

    train_df = pd.read_parquet(out_train)
    test_df = pd.read_parquet(out_test)
    # One row with NaN target should be dropped; remaining 3 split roughly in half.
    assert len(train_df) + len(test_df) == 3
    assert set(train_df.columns) == {"rating", "year", "duration", "title"}
