import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.src import optimize_model as opt


def test_load_parquet_missing_raises(tmp_path):
    missing = tmp_path / "missing.parquet"
    with pytest.raises(FileNotFoundError):
        opt._load_parquet(missing)


def test_load_table_unsupported_suffix(tmp_path):
    bad = tmp_path / "data.csv"
    bad.write_text("a,b\n1,2", encoding="utf-8")
    with pytest.raises(ValueError):
        opt._load_table(bad)


def test_main_clips_rating_predictions(tmp_path, monkeypatch):
    train_df = pd.DataFrame(
        {
            "rating": [1.0, 4.0, 3.0],
            "year": [2020, 2021, 2022],
        }
    )
    test_df = pd.DataFrame(
        {
            "rating": [1.0, 4.0],
            "year": [2023, 2024],
        }
    )

    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    class StubModel:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.array([-2.0, 10.0])[: len(X)]

    def fake_candidates(seed: int, use_identities: bool, target: str):
        return {"stub": StubModel()}

    def fake_eval(*args, **kwargs):
        return [opt.CandidateResult(name="stub", cv_mean=0.0, cv_std=0.0)]

    def fake_fit_search(**kwargs):
        return StubModel()

    monkeypatch.setattr(opt, "_build_candidates", fake_candidates)
    monkeypatch.setattr(opt, "_evaluate_candidates", fake_eval)
    monkeypatch.setattr(opt, "_fit_search", fake_fit_search)
    monkeypatch.setattr(opt.joblib, "dump", lambda obj, path: path)

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
            "--clip-predictions",
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
    report = json.loads(report_out.read_text(encoding="utf-8"))
    metrics = report["test_metrics"]
    # After clipping [-2, 10] -> [0, 5] against y_true [1, 4]
    assert pytest.approx(metrics["mae"], rel=1e-6) == 1.0
    assert pytest.approx(metrics["rmse"], rel=1e-6) == 1.0
    assert metrics["acc_within_0_25"] == 0.0
    assert metrics["acc_within_0_50"] == 0.0
    assert report["clip_predictions"] is True


def test_main_budget_log_transform_no_clip(tmp_path, monkeypatch):
    train_df = pd.DataFrame({"budget": [10.0, 20.0, 30.0], "year": [2020, 2021, 2022]})
    test_df = pd.DataFrame({"budget": [5.0, 15.0], "year": [2023, 2024]})

    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    class StubModel:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.full(len(X), np.log1p(100.0))

    def fake_candidates(seed: int, use_identities: bool, target: str):
        return {"stub": StubModel()}

    def fake_eval(*args, **kwargs):
        return [opt.CandidateResult(name="stub", cv_mean=0.0, cv_std=0.0)]

    def fake_fit_search(**kwargs):
        return StubModel()

    monkeypatch.setattr(opt, "_build_candidates", fake_candidates)
    monkeypatch.setattr(opt, "_evaluate_candidates", fake_eval)
    monkeypatch.setattr(opt, "_fit_search", fake_fit_search)
    monkeypatch.setattr(opt.joblib, "dump", lambda obj, path: path)

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
            "budget",
            "--cv",
            "2",
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
    report = json.loads(report_out.read_text(encoding="utf-8"))
    metrics = report["test_metrics"]
    # Predictions inverse-log1p -> 100.0 constant; no clipping expected for budget
    assert report["clip_predictions"] is False
    assert metrics["mae"] > 0
    assert report["target_transform"]["name"] == "log1p"
