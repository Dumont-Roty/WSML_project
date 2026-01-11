import pandas as pd
import numpy as np
import pytest

from ml.src import optimize_model as opt


def test_ensure_list_col_populates_missing_and_strings():
    df = pd.DataFrame({"directors": [None, "Alice", ["Bob", ""]]})
    opt._ensure_list_col(df, "directors")
    assert df["directors"].iloc[0] == ["__MISSING__"]
    assert df["directors"].iloc[1] == ["Alice"]
    assert df["directors"].iloc[2] == ["Bob"]


def test_split_xy_filters_target_and_numeric_only():
    df = pd.DataFrame(
        {
            "rating": [4.0, np.nan, 3.0],
            "year": [2020, 2021, 2022],
            "title": ["a", "b", "c"],
        }
    )
    X, y = opt._split_xy(df, target="rating", drop_cols=["title"])
    assert len(y) == 2  # one NaN removed
    assert list(X.columns) == ["year"]


def test_split_xy_identities_handles_lists_and_missing():
    df = pd.DataFrame(
        {
            "rating": [4.0, 3.5],
            "year": [2020, 2021],
            "duration": [100, 90],
            "directors": [["Alice"], None],
        }
    )
    X, y, num_cols, id_cols = opt._split_xy_identities(
        df,
        target="rating",
        drop_cols=[],
        identity_cols=["directors"],
        numeric_cols=["year", "duration"],
    )
    assert len(y) == 2
    assert num_cols == ["year", "duration"]
    assert id_cols == ["directors"]
    assert X["directors"].iloc[1] == ["__MISSING__"]


def test_split_xy_identities_raises_when_no_identity_cols():
    df = pd.DataFrame(
        {
            "rating": [4.0, 3.5],
            "year": [2020, 2021],
        }
    )
    with pytest.raises(ValueError):
        opt._split_xy_identities(
            df,
            target="rating",
            drop_cols=[],
            identity_cols=["directors"],
            numeric_cols=["year"],
        )


def test_split_xy_identities_raises_when_target_missing():
    df = pd.DataFrame({"year": [2020, 2021], "duration": [100, 90]})
    with pytest.raises(ValueError):
        opt._split_xy_identities(
            df,
            target="rating",
            drop_cols=[],
            identity_cols=["directors"],
            numeric_cols=["year"],
        )


def test_split_xy_identities_raises_when_no_numeric_cols():
    df = pd.DataFrame(
        {
            "rating": [4.0, 3.5],
            "directors": [["a"], ["b"]],
        }
    )
    with pytest.raises(ValueError):
        opt._split_xy_identities(
            df,
            target="rating",
            drop_cols=[],
            identity_cols=["directors"],
            numeric_cols=["year", "duration"],
        )


def test_split_xy_raises_when_no_numeric_features_left():
    df = pd.DataFrame(
        {
            "rating": [4.0, 3.5],
            "title": ["a", "b"],
            "url": ["u1", "u2"],
        }
    )
    with pytest.raises(ValueError):
        opt._split_xy(df, target="rating", drop_cols=["title", "url"])
