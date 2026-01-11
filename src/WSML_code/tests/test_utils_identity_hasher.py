import numpy as np
import pandas as pd

from WSML_code.utils.identity_hasher import IdentityHasher


def test_wsml_identity_hasher_basic_and_missing():
    cols = ("directors",)
    hasher = IdentityHasher(columns=cols, n_features=8)

    df = pd.DataFrame([
        {"directors": ["Alice", "Bob"]},
        {"directors": None},
    ])

    X = hasher.transform(df)
    assert X.shape == (2, 8)
    assert np.any(X[1] != 0)


def test_wsml_identity_hasher_normalizes_spacing():
    hasher = IdentityHasher(columns=("genres",), n_features=4)
    df = pd.DataFrame([
        {"genres": ["Sci Fi", "sci  fi"]},
    ])

    X = hasher.transform(df)
    assert X.shape == (1, 4)
    assert np.isclose(float(X.sum()), 1.0)
