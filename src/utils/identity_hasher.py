from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import FeatureHasher


class IdentityHasher(BaseEstimator, TransformerMixin):
    """Encodeur par hashing pour colonnes catégorielles de type liste (réalisateurs, genres, ...).
    
    OBJECTIF : Transformer des listes de noms (ex: ["Spielberg", "Nolan"]) en features numériques
               pour l'entraînement de modèles ML.
    
    POURQUOI HASHING ?
    - One-Hot Encoding classique explose en dimension avec des milliers de noms uniques
    - Le hashing projette les noms dans un espace fixe (ex: 1024 dimensions)
    - Collision possible mais impact faible sur la précision
    
    FONCTIONNEMENT :
    1. Normalisation : "Steven Spielberg" -> "steven_spielberg"
    2. Préfixage : "directors=steven_spielberg" pour distinguer réalisateurs/acteurs
    3. Hashing : chaque token est projeté dans un vecteur de taille n_features
    4. Agrégation : somme des vecteurs pour tous les tokens d'une colonne
    
    EXEMPLE :
    Input:  {"directors": ["Nolan", "Spielberg"], "genres": ["Sci-Fi"]}
    Output: array de 1024 floats (vecteur dense)
    
    - Normalise les tokens en minuscules avec underscores à la place des espaces
    - Remplace les entrées manquantes par un token __MISSING__
    - Produit une matrice dense (float32) via sklearn FeatureHasher
    """
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
        # Collapse any whitespace run to a single underscore for stable hashing.
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
