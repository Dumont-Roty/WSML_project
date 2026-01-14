import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor

from src.ml import optimize_model as opt


def test_main_writes_model_and_report(tmp_path, monkeypatch):
    train_df = pd.DataFrame(
        {
            "rating": [3.0, 4.0, 2.5, 3.5],
            "year": [2020, 2021, 2022, 2023],
            "duration": [100, 90, 110, 95],
        }
    )
    test_df = pd.DataFrame(
        {
            "rating": [3.0, 4.5],
            "year": [2024, 2025],
            "duration": [105, 85],
        }
    )

    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    def fake_candidates(seed: int, use_identities: bool, target: str):
        return {"dummy_mean": DummyRegressor(strategy="mean")}

    monkeypatch.setattr(opt, "_build_candidates", fake_candidates)

    model_out = tmp_path / "model.joblib"
    report_out = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "optimize_model",
            "--train",
            str(train_path),
            "--test",
            str(test_path),
            "--target",
            "rating",
            "--cv",
            "2",
            "--search",
            "grid",
            "--save-model",
            str(model_out),
            "--save-report",
            str(report_out),
            "--drop-cols",
            "",
        ],
    )

    rc = opt.main()
    assert rc == 0
    assert model_out.exists()
    assert report_out.exists()

    report = json.loads(report_out.read_text(encoding="utf-8"))
    assert report["selected_model"] == "dummy_mean"
    assert set(report["test_metrics"].keys()) == {"r2", "mae", "rmse", "acc_within_0_25", "acc_within_0_50"}
    assert report["n_test"] == 2
