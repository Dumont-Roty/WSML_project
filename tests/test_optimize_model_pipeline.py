import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from src.ml import optimize_model as opt


def test_default_artifacts_for_target_variants():
    model_out, report_out = opt._default_artifacts_for_target("rating")
    assert model_out.name == "best_model.joblib"
    assert report_out.name == "metrics.json"

    m2, r2 = opt._default_artifacts_for_target("budget")
    assert "budget" in m2.stem
    assert "budget" in r2.stem


def test_filter_target_rows_budget_and_missing_col():
    df = pd.DataFrame({"budget": [100.0, -5.0, 0.0, 50.0], "year": [2020, 2021, 2022, 2023]})
    out = opt._filter_target_rows(df, "budget")
    assert list(out["budget"]) == [100.0, 50.0]
    # Missing column: should return input unchanged
    same = opt._filter_target_rows(df, "rating")
    assert same.shape == df.shape


def test_maybe_transform_and_inverse_budget():
    y = pd.Series([0.0, 9.0])
    y_t, meta = opt._maybe_transform_target(y, "budget")
    assert meta and meta["name"] == "log1p"
    restored = opt._inverse_transform_pred(y_t, meta)
    assert np.allclose(restored, y.to_numpy())


def test_metrics_masks_nan_and_empty():
    y_true = np.array([1.0, np.nan, 2.0, np.inf])
    y_pred = np.array([1.1, 0.0, 1.8, 2.0])
    m = opt._metrics(y_true, y_pred)
    assert set(m.keys()) == {"r2", "mae", "rmse", "acc_within_0_25", "acc_within_0_50"}
    assert m["mae"] > 0
    # Empty after masking
    m_empty = opt._metrics(np.array([]), np.array([]))
    assert np.isnan(m_empty["r2"])


def test_evaluate_candidates_ranks_better_model_first():
    X = pd.DataFrame({"year": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    candidates = {
        "dummy": DummyRegressor(strategy="mean"),
        "ridge": Ridge(random_state=0),
    }
    cv = KFold(n_splits=2, shuffle=True, random_state=0)
    res = opt._evaluate_candidates(
        X,
        y,
        candidates=candidates,
        cv=cv,
        scoring="r2",
        target="rating",
        use_identities=False,
    )
    assert res[0].name == "ridge"
    assert res[0].cv_mean >= res[1].cv_mean


def test_fit_search_with_identities_pipeline_predicts():
    X = pd.DataFrame(
        {
            "year": [2020, 2021, 2022, 2023],
            "directors": [["a"], ["b"], ["a"], [None]],
        }
    )
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    cv = KFold(n_splits=2, shuffle=True, random_state=0)
    est = opt._fit_search(
        name="dummy_mean",
        base_model=DummyRegressor(strategy="mean"),
        X=X,
        y=y,
        cv=cv,
        scoring="r2",
        search="grid",
        n_iter=5,
        seed=0,
        target="rating",
        use_identities=True,
        numeric_cols=["year"],
        identity_cols=["directors"],
        hash_dim=8,
    )
    preds = est.predict(X)
    assert preds.shape == (4,)


def test_param_grid_for_budget_and_default():
    grid_budget_rf = opt._param_grid_for("rf", target="budget")
    assert "model__n_estimators" in grid_budget_rf

    grid_svr = opt._param_grid_for("svr", target="rating")
    assert "model__C" in grid_svr
