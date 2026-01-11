import numpy as np
import pandas as pd

from ml.src.identity_hasher import IdentityHasher


def test_identity_hasher_basic_and_missing():
    cols = ("directors", "genres")
    hasher = IdentityHasher(columns=cols, n_features=8)

    df = pd.DataFrame(
        [
            {"directors": ["Alice"], "genres": ["Drama"]},
            {"directors": None, "genres": []},
        ]
    )

    X = hasher.transform(df)
    assert X.shape == (2, 8)
    # Row with missing values should still hash a __MISSING__ token => not all zeros.
    assert np.any(X[1] != 0)


def test_identity_hasher_normalizes_tokens():
    cols = ("directors",)
    hasher = IdentityHasher(columns=cols, n_features=4)

    df = pd.DataFrame([
        {"directors": ["Jean Luc", "JEAN  luc"]},
    ])

    X = hasher.transform(df)
    # Duplicate normalized token should collapse into a single count of 1.0 per token.
    assert X.shape == (1, 4)
    assert np.isclose(X.sum(), 1.0)
