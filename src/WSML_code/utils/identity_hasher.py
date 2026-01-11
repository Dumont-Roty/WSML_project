from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import FeatureHasher


class IdentityHasher(BaseEstimator, TransformerMixin):
    def __init__(self, columns: Tuple[str, ...], n_features: int = 1024):
        self.columns = columns
        self.n_features = int(n_features)
        self._hasher = FeatureHasher(
            n_features=self.n_features,
            input_type="dict",
            alternate_sign=False,
        )

    @staticmethod
    def _norm_token(s: str) -> str:
        s = s.strip().lower()
        # Collapse whitespace runs to a single underscore for stable hashing.
        return re.sub(r"\s+", "_", s)

    def fit(self, X: Any, y: Any = None):
        return self

    def transform(self, X: Any):
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            df = pd.DataFrame(X, columns=self.columns)

        rows: List[Dict[str, float]] = []
        for _, r in df.iterrows():
            feats: Dict[str, float] = {}
            for col in self.columns:
                v = r.get(col)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    items = ["__MISSING__"]
                elif isinstance(v, list):
                    items = v if v else ["__MISSING__"]
                else:
                    items = [v]

                for item in items:
                    tok = f"{col}={self._norm_token(str(item))}"
                    feats[tok] = 1.0
            rows.append(feats)

        X_sparse = self._hasher.transform(iter(rows))
        return np.asarray(X_sparse.todense()).astype(np.float32)
