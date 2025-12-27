"""Version réécrite et commentée du script pédagogique du professeur.

But: fournir un script autonome, lisible et réutilisable pour apprentissage ML.
- Charge `cleaned_data.parquet` depuis le répertoire courant ou chemin fourni
- Prépare X/y sur colonnes d'exemple
- Entraîne plusieurs modèles de baseline et compare via CV
- Entraîne un modèle final (GradientBoosting) et le sérialise avec joblib

Usage (depuis la racine du repo) :
    python ml/src/ml_prof_rewrite.py --data ml/data/cleaned_data.parquet --output ml/models/final_model.joblib
"""
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

RND = 42


def load_data(path: Path) -> pd.DataFrame:
    """Charge le dataset (Parquet ou CSV) en DataFrame pandas."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    if path.suffix.lower() in (".parquet",):
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path)


def prepare_features(df: pd.DataFrame, target: str = "rating") -> tuple[pd.DataFrame, np.ndarray]:
    """Prépare un jeu de features simple à partir de `merged_results.json`.

    - `target` : nom de la colonne cible (ex. `rating`, `budget`, `revenue`).
    - Construit des features numériques simples : year, duration, counts (directors, cast, genres),
      popularity metrics (nbr_watched, nbr_likes, fans_favoris), et encodage basique de `kind`.

    Retourne `(X_dataframe, y_numpy_array)`.
    """
    if target not in df.columns:
        raise ValueError(f"Cible '{target}' non trouvée dans le DataFrame")

    # cible numérique
    y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)

    # colonnes numériques candidates (présentes dans merged_results.json)
    numeric_candidates = [
        "year",
        "duration",
        "nbr_watched",
        "nbr_appearence",
        "nbr_likes",
        "fans_favoris",
        "budget",
        "revenue",
    ]

    X = pd.DataFrame(index=df.index)

    for c in numeric_candidates:
        if c in df.columns and c != target:
            X[c] = pd.to_numeric(df[c], errors="coerce")

    # helper to get length of list-like fields
    def _len_or_zero(v):
        try:
            if v is None:
                return 0
            # strings are not lists in our schema; treat them as single
            if isinstance(v, str):
                return 1
            return len(v)
        except Exception:
            return 0

    # list-like fields -> counts
    for list_col in ("directors", "casting", "producers", "writers", "genres", "languages", "themes"):
        if list_col in df.columns:
            X[f"{list_col}_count"] = df[list_col].apply(_len_or_zero)

    # simple categorical encoding for small-cardinality fields
    if "kind" in df.columns:
        X["kind_code"] = df["kind"].astype("category").cat.codes.replace({-1: 0})
    if "source_page" in df.columns:
        X["source_page_code"] = df["source_page"].astype("category").cat.codes.replace({-1: 0})

    # drop rows where target missing
    mask = ~np.isnan(y)
    X = X.loc[mask].fillna(0)
    y = y[mask]

    return X, y


def infer_feature_types(df: pd.DataFrame):
    """Retourne listes de colonnes numériques et catégorielles détectées."""
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # retirer la cible si elle s'est glissée
    for c in ["prix"]:
        if c in num_cols:
            num_cols.remove(c)
        if c in cat_cols:
            cat_cols.remove(c)
    return num_cols, cat_cols


def build_preprocessing_pipeline(num_cols, cat_cols):
    """Construit un `ColumnTransformer` pour num/obj avec scaling et encodage."""
    num_pipeline = Pipeline([("scaler", StandardScaler())])
    cat_pipeline = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))])

    preproc = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
    ], remainder="drop")
    return preproc


def eval_cv(model, X, y, cv: int = 5) -> dict:
    """Évalue `model` via cross_val_score et renvoie métriques simples."""
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    # cross_val_score renvoie valeurs négatives pour certains scoring; on abs() si besoin
    rmse_scores = np.abs(scores)
    return {"rmse_mean": float(rmse_scores.mean()), "rmse_std": float(rmse_scores.std())}


def baseline_models(Xtr, Xte, ytr, yte):
    """Entraîne et compare plusieurs modèles de baseline."""
    models = {
        "Dummy": DummyRegressor("mean"),
        "Linear": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=RND, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=RND),
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(Xtr, ytr)
        ypred = model.predict(Xte)
        rmse = mean_squared_error(yte, ypred, squared=False)
        r2 = r2_score(yte, ypred)
        mae = mean_absolute_error(yte, ypred)
        print(f"{name} — RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        results[name] = {"rmse": rmse, "mae": mae, "r2": r2}
    return results


def grid_search_mlp(Xtr, ytr):
    """Simple GridSearch pour MLP (coûteux) — utilisé comme démonstration."""
    from sklearn.neural_network import MLPRegressor

    param_grid = {
        "hidden_layer_sizes": [(10,), (30,), (50,)],
        "max_iter": [1000],
    }
    gs = GridSearchCV(MLPRegressor(random_state=RND), param_grid=param_grid, cv=3, scoring="neg_root_mean_squared_error")
    gs.fit(Xtr, ytr)
    print("Best MLP params:", gs.best_params_)
    return gs.best_estimator_


def train_final_model(Xtr, Xte, ytr, yte):
    """Entraîne le modèle final (ici GradientBoosting) et retourne l'objet entraîné."""
    final = GradientBoostingRegressor(random_state=RND)
    final.fit(Xtr, ytr)
    ytr_pred = final.predict(Xtr)
    yte_pred = final.predict(Xte)
    print(f"Final model train RMSE: {mean_squared_error(ytr, ytr_pred, squared=False):.4f}")
    print(f"Final model test  RMSE: {mean_squared_error(yte, yte_pred, squared=False):.4f}")
    return final


def save_metrics(metrics: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Script ML réécrit pour apprentissage")
    parser.add_argument("--data", default="ml/data/cleaned_data.parquet", help="Chemin vers le dataset (parquet ou csv)")
    parser.add_argument("--output", default="ml/models/final_model.joblib", help="Chemin de sortie pour le modèle sérialisé")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    df = load_data(Path(args.data))
    print(f"Chargé {len(df)} lignes depuis {args.data}")

    X, y = prepare_features(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, random_state=RND)

    print("\n--- Baseline models ---")
    baseline_models(Xtr, Xte, ytr, yte)

    # Optionnel : recherche d'hyperparamètres pour MLP (lent)
    # best_mlp = grid_search_mlp(Xtr, ytr)

    print("\n--- Entrainement du modèle final ---")
    final = train_final_model(Xtr, Xte, ytr, yte)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, out_path)
    print(f"Modèle sérialisé dans {out_path}")


if __name__ == "__main__":
    main()
