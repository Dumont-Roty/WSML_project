from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

# Permet d'importer `ml.*` même si le script est lancé via `python ml/src/optimize_model.py`
# (dans ce cas sys.path[0] pointe sur ml/src, pas sur la racine du repo).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.src.identity_hasher import IdentityHasher


DEFAULT_TRAIN = Path("ml/data/train.parquet")
DEFAULT_TEST = Path("ml/data/test.parquet")
DEFAULT_MODEL_OUT = Path("ml/models/best_model.joblib")
DEFAULT_REPORT_OUT = Path("ml/models/metrics.json")


IDENTITY_COLS_DEFAULT = [
    "directors",
    "casting",
    "producers",
    "writers",
    "composer",
    "studio",
    "languages",
    "genres",
    "themes",
]

NUMERIC_COLS_DEFAULT = [
    "year",
    "duration",
    "nbr_watched",
    "nbr_appearence",
    "nbr_likes",
    "fans_favoris",
    "budget",
    "revenue",
]


BUDGET_NUMERIC_COLS_DEFAULT = [
    # Hypothèse "avant production": infos disponibles a priori.
    "year",
    "duration",
]


@dataclass(frozen=True)
class CandidateResult:
    name: str
    cv_mean: float
    cv_std: float


def _load_parquet(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    return pd.read_parquet(path)


def _load_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path)
    raise ValueError(f"Format non supporté: {path} (attendu .parquet ou .json)")


def _ensure_list_col(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        df[col] = None

    def _to_list(v: Any) -> List[str]:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ["__MISSING__"]
        if isinstance(v, list):
            items = [str(x).strip() for x in v if x is not None and str(x).strip()]
            return items if items else ["__MISSING__"]
        if isinstance(v, str):
            s = v.strip()
            return [s] if s else ["__MISSING__"]
        s = str(v).strip()
        return [s] if s else ["__MISSING__"]

    df[col] = df[col].apply(_to_list)


def _split_xy(
    df: pd.DataFrame,
    target: str,
    drop_cols: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise ValueError(f"Colonne cible '{target}' absente. Colonnes: {list(df.columns)}")

    df = df.copy()
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df[df[target].notna()]

    to_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=[target, *to_drop], errors="ignore")
    y = df[target]

    # Par défaut on reste proche de ton nettoyage: features numériques.
    # On supprime les colonnes non numériques restantes (souvent url/title).
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        X = X.drop(columns=non_numeric)

    if X.shape[1] == 0:
        raise ValueError(
            "Aucune feature utilisable après filtrage (features numériques). "
            "Vérifie les colonnes du parquet ou ajuste --drop-cols."
        )

    return X, y


def _split_xy_identities(
    df: pd.DataFrame,
    target: str,
    drop_cols: List[str],
    *,
    identity_cols: List[str],
    numeric_cols: List[str],
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    if target not in df.columns:
        raise ValueError(f"Colonne cible '{target}' absente. Colonnes: {list(df.columns)}")

    df = df.copy()
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df[df[target].notna()]

    to_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=[target, *to_drop], errors="ignore")
    y = df[target]

    present_numeric = [c for c in numeric_cols if c in X.columns]
    present_identity = [c for c in identity_cols if c in X.columns]
    if not present_numeric:
        raise ValueError(
            "Aucune feature numérique trouvée pour le mode identités. "
            f"Attendu une partie de: {numeric_cols}. Colonnes disponibles: {list(X.columns)}"
        )
    if not present_identity:
        raise ValueError(
            "Aucune colonne d'identités trouvée (directors/casting/...). "
            "Utilise un JSON enrichi (ex: ml/data/final_results_28.json) ou désactive --use-identities."
        )

    X = X[present_numeric + present_identity].copy()
    for col in present_numeric:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in present_identity:
        _ensure_list_col(X, col)
    return X, y, present_numeric, present_identity


def _build_candidates(seed: int, *, use_identities: bool, target: str) -> Dict[str, Any]:
    target_name = str(target).strip().lower()
    if use_identities:
        # Le budget est beaucoup plus lent à entraîner (CV + hashing + modèles). On limite.
        if target_name == "budget":
            return {
                "dummy_mean": DummyRegressor(strategy="mean"),
                "ridge": Ridge(random_state=seed),
                "gbr": GradientBoostingRegressor(random_state=seed),
                "rf": RandomForestRegressor(
                    n_estimators=200,
                    random_state=seed,
                    n_jobs=-1,
                ),
            }
        return {
            "dummy_mean": DummyRegressor(strategy="mean"),
            "ridge": Ridge(random_state=seed),
            "rf": RandomForestRegressor(
                n_estimators=400,
                random_state=seed,
                n_jobs=-1,
            ),
            "gbr": GradientBoostingRegressor(random_state=seed),
        }
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(random_state=seed),
        "rf": RandomForestRegressor(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
        ),
        "gbr": GradientBoostingRegressor(random_state=seed),
        "knn": KNeighborsRegressor(),
        "svr": SVR(),
        "mlp": MLPRegressor(
            random_state=seed,
            max_iter=800,
            early_stopping=True,
        ),
    }


def _build_pipeline_identities(
    model: Any,
    *,
    numeric_cols: List[str],
    identity_cols: List[str],
    hash_dim: int,
) -> Pipeline:
    num = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", num, list(numeric_cols)),
            ("id", IdentityHasher(tuple(identity_cols), n_features=int(hash_dim)), list(identity_cols)),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return Pipeline(steps=[("pre", pre), ("model", model)])


def _build_pipeline(model: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def _param_grid_for(name: str, *, target: str) -> Dict[str, Any]:
    target_name = str(target).strip().lower()
    # Noms avec préfixe pipeline: model__...
    if name == "ridge":
        if target_name == "budget":
            return {"model__alpha": np.logspace(-3, 3, 9)}
        return {"model__alpha": np.logspace(-4, 4, 13)}

    if name == "rf":
        if target_name == "budget":
            # Grille plus petite pour éviter des runs interminables.
            return {
                "model__n_estimators": [200],
                "model__max_depth": [None, 20],
                "model__min_samples_leaf": [1, 5],
                "model__max_features": ["sqrt"],
            }
        return {
            "model__n_estimators": [300, 600],
            "model__max_depth": [None, 10, 30],
            "model__min_samples_leaf": [1, 2, 5],
            "model__max_features": ["sqrt", 1.0],
        }

    if name == "gbr":
        if target_name == "budget":
            return {
                "model__n_estimators": [200, 400],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
                "model__subsample": [1.0],
            }
        return {
            "model__n_estimators": [100, 300, 600],
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [1.0, 0.8],
        }

    if name == "knn":
        return {
            "model__n_neighbors": list(range(2, 25)),
            "model__weights": ["uniform", "distance"],
        }

    if name == "svr":
        return {
            "model__C": [0.1, 1.0, 10.0],
            "model__gamma": ["scale", "auto"],
            "model__epsilon": [0.05, 0.1, 0.2],
        }

    if name == "mlp":
        return {
            "model__hidden_layer_sizes": [(50,), (100,), (100, 50)],
            "model__alpha": [1e-5, 1e-4, 1e-3],
            "model__learning_rate_init": [1e-3, 5e-4],
            "model__max_iter": [800, 1200],
        }

    return {}


def _evaluate_candidates(
    X: pd.DataFrame,
    y: pd.Series,
    candidates: Dict[str, Any],
    cv: KFold,
    scoring: str,
    *,
    target: str,
    use_identities: bool,
    numeric_cols: List[str] | None = None,
    identity_cols: List[str] | None = None,
    hash_dim: int = 1024,
) -> List[CandidateResult]:
    results: List[CandidateResult] = []

    for name, model in candidates.items():
        if use_identities:
            pipe = _build_pipeline_identities(
                model,
                numeric_cols=list(numeric_cols or []),
                identity_cols=list(identity_cols or []),
                hash_dim=int(hash_dim),
            )
        else:
            pipe = _build_pipeline(model)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        results.append(CandidateResult(name=name, cv_mean=float(np.mean(scores)), cv_std=float(np.std(scores))))

    results.sort(key=lambda r: r.cv_mean, reverse=True)
    return results


def _fit_search(
    *,
    name: str,
    base_model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: KFold,
    scoring: str,
    search: str,
    n_iter: int,
    seed: int,
    target: str,
    use_identities: bool,
    numeric_cols: List[str] | None = None,
    identity_cols: List[str] | None = None,
    hash_dim: int = 1024,
) -> Any:
    if use_identities:
        pipe = _build_pipeline_identities(
            base_model,
            numeric_cols=list(numeric_cols or []),
            identity_cols=list(identity_cols or []),
            hash_dim=int(hash_dim),
        )
    else:
        pipe = _build_pipeline(base_model)
    grid = _param_grid_for(name, target=str(target))

    if not grid:
        pipe.fit(X, y)
        return pipe

    if search == "random":
        # Ici on reste sans scipy: RandomizedSearchCV peut tirer dans des listes.
        searcher = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=grid,
            n_iter=min(n_iter, max(1, int(np.prod([len(v) for v in grid.values()])))),
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            random_state=seed,
            refit=True,
        )
    else:
        searcher = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            refit=True,
        )

    searcher.fit(X, y)
    return searcher.best_estimator_


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"Tailles incohérentes: y_true={y_true.shape[0]} vs y_pred={y_pred.shape[0]}")

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not bool(np.all(mask)):
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    if y_true.size == 0:
        nan = float("nan")
        return {"r2": nan, "mae": nan, "rmse": nan}

    mse = mean_squared_error(y_true, y_pred)

    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = float("nan")

    return {
        "r2": r2,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
    }


def _default_artifacts_for_target(target: str) -> tuple[Path, Path]:
    t = str(target).strip().lower()
    if t == "rating":
        return DEFAULT_MODEL_OUT, DEFAULT_REPORT_OUT
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in t)
    return Path(f"ml/models/{safe}_model.joblib"), Path(f"ml/models/{safe}_metrics.json")


def _filter_target_rows(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in df.columns:
        return df
    out = df.copy()
    out[target] = pd.to_numeric(out[target], errors="coerce")
    out = out[out[target].notna()]
    if str(target).strip().lower() == "budget":
        out = out[out[target] > 0]
    return out


def _maybe_transform_target(y: pd.Series, target: str) -> tuple[np.ndarray, dict[str, Any] | None]:
    t = str(target).strip().lower()
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if t == "budget":
        # Stabilise l'apprentissage (budgets très étalés)
        return np.log1p(y_arr), {"name": "log1p", "inverse": "expm1"}
    return y_arr, None


def _inverse_transform_pred(y_pred: np.ndarray, transform: dict[str, Any] | None) -> np.ndarray:
    if not transform:
        return np.asarray(y_pred, dtype=float)
    if transform.get("name") == "log1p":
        return np.expm1(np.asarray(y_pred, dtype=float))
    return np.asarray(y_pred, dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Entraîne et optimise un modèle de régression (rating) à partir de train/test.parquet.\n"
            "Approche type cours: baseline, comparaison via CV, puis GridSearchCV/RandomizedSearchCV."
        )
    )
    parser.add_argument("--train", default=str(DEFAULT_TRAIN), help="Train (.parquet ou .json)")
    parser.add_argument("--test", default=str(DEFAULT_TEST), help="Test (.parquet ou .json)")
    parser.add_argument("--target", default="rating", help="Colonne cible")
    parser.add_argument(
        "--drop-cols",
        default="url,title",
        help="Colonnes à exclure (séparées par des virgules). Ex: url,title",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument(
        "--test-size",
        type=float,
        default=None,
        help="Si renseigné, ignore --test et crée un split train/test depuis --train (ex: 0.2).",
    )
    parser.add_argument("--cv", type=int, default=5, help="Nombre de folds CV")
    parser.add_argument(
        "--scoring",
        default="r2",
        help="Score sklearn (ex: r2, neg_mean_absolute_error)",
    )
    parser.add_argument(
        "--search",
        choices=["grid", "random"],
        default="grid",
        help="Méthode d'optimisation hyperparamètres",
    )
    parser.add_argument(
        "--clip-predictions",
        action="store_true",
        help="Borne les prédictions dans [0, 5] (utile car rating ∈ [0, 5]).",
    )
    parser.add_argument(
        "--use-identities",
        action="store_true",
        help=(
            "Utilise les identités (directors/casting/...) si disponibles (JSON enrichi). "
            "Active un encodage par hashing pour que les noms aient un impact."
        ),
    )
    parser.add_argument(
        "--hash-dim",
        type=int,
        default=1024,
        help="Dimension du hashing pour les identités.",
    )
    parser.add_argument("--n-iter", type=int, default=30, help="n_iter si --search=random")
    parser.add_argument(
        "--save-model",
        default=None,
        help=(
            "Chemin sortie .joblib. Si omis: rating -> ml/models/best_model.joblib, "
            "sinon ml/models/<target>_model.joblib"
        ),
    )
    parser.add_argument(
        "--save-report",
        default=None,
        help=(
            "Chemin sortie métriques JSON. Si omis: rating -> ml/models/metrics.json, "
            "sinon ml/models/<target>_metrics.json"
        ),
    )
    args = parser.parse_args()

    target_name = str(args.target).strip().lower()
    if args.clip_predictions and target_name != "rating":
        print("[info] --clip-predictions ignoré car la cible n'est pas 'rating'.")

    # Pour budget, on réduit le coût par défaut (sinon CV+grid peut être très long).
    if target_name == "budget" and int(args.cv) == 5:
        print("[info] target=budget: cv passe de 5 à 3 pour éviter des runs trop longs.")
        args.cv = 3

    train_df = _filter_target_rows(_load_table(Path(args.train)), target=str(args.target))
    if args.test_size is not None:
        train_df, test_df = train_test_split(
            train_df,
            test_size=float(args.test_size),
            random_state=int(args.seed),
            shuffle=True,
        )
    else:
        test_df = _filter_target_rows(_load_table(Path(args.test)), target=str(args.target))

    drop_cols = [c.strip() for c in str(args.drop_cols).split(",") if c.strip()]
    numeric_cols: List[str] | None = None
    identity_cols: List[str] | None = None

    # Choix des features numériques selon la cible
    numeric_defaults = NUMERIC_COLS_DEFAULT if target_name == "rating" else BUDGET_NUMERIC_COLS_DEFAULT

    if args.use_identities:
        Xtr, ytr, numeric_cols, identity_cols = _split_xy_identities(
            train_df,
            target=args.target,
            drop_cols=drop_cols,
            identity_cols=IDENTITY_COLS_DEFAULT,
            numeric_cols=numeric_defaults,
        )
        Xte, yte, _, _ = _split_xy_identities(
            test_df,
            target=args.target,
            drop_cols=drop_cols,
            identity_cols=IDENTITY_COLS_DEFAULT,
            numeric_cols=numeric_defaults,
        )
    else:
        Xtr, ytr = _split_xy(train_df, target=args.target, drop_cols=drop_cols)
        Xte, yte = _split_xy(test_df, target=args.target, drop_cols=drop_cols)

    # Transformation de la cible (ex: log1p pour budget)
    ytr_model, target_transform = _maybe_transform_target(ytr, target=str(args.target))
    yte_model, _ = _maybe_transform_target(yte, target=str(args.target))
    ytr_model_s = pd.Series(ytr_model, index=ytr.index)
    yte_model_s = pd.Series(yte_model, index=yte.index)

    cv = KFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    candidates = _build_candidates(seed=args.seed, use_identities=bool(args.use_identities), target=str(args.target))

    results = _evaluate_candidates(
        Xtr,
        ytr_model_s,
        candidates=candidates,
        cv=cv,
        scoring=args.scoring,
        target=str(args.target),
        use_identities=bool(args.use_identities),
        numeric_cols=numeric_cols,
        identity_cols=identity_cols,
        hash_dim=int(args.hash_dim),
    )

    print("\n--- Comparaison (CV sur train) ---")
    for r in results:
        print(f"{r.name:10s}  mean={r.cv_mean:+.4f}  std={r.cv_std:.4f}")

    # On évite de sélectionner la baseline si un autre modèle fait mieux
    best_name = results[0].name
    if best_name.startswith("dummy") and len(results) > 1:
        best_name = results[1].name

    print(f"\nChoisi pour tuning: {best_name} (search={args.search})")

    best_estimator = _fit_search(
        name=best_name,
        base_model=candidates[best_name],
        X=Xtr,
        y=ytr_model_s,
        cv=cv,
        scoring=args.scoring,
        search=args.search,
        n_iter=args.n_iter,
        seed=args.seed,
        target=str(args.target),
        use_identities=bool(args.use_identities),
        numeric_cols=numeric_cols,
        identity_cols=identity_cols,
        hash_dim=int(args.hash_dim),
    )

    best_estimator.fit(Xtr, ytr_model_s)
    y_pred_model = np.asarray(best_estimator.predict(Xte), dtype=float).reshape(-1)
    y_pred = _inverse_transform_pred(y_pred_model, target_transform)
    if args.clip_predictions and target_name == "rating":
        y_pred = np.clip(y_pred, 0.0, 5.0)

    # Intervalle de prédiction (heuristique): quantiles des résidus sur la cible transformée
    pred_interval: dict[str, Any] | None = None
    if target_transform and target_transform.get("name") == "log1p":
        resid = np.asarray(yte_model_s, dtype=float).reshape(-1) - y_pred_model
        resid = resid[np.isfinite(resid)]
        if resid.size:
            q_low = float(np.quantile(resid, 0.10))
            q_high = float(np.quantile(resid, 0.90))
            pred_interval = {
                "method": "residual_quantiles_log1p",
                "coverage": 0.80,
                "resid_q_low": q_low,
                "resid_q_high": q_high,
            }

    report: Dict[str, Any] = {
        "target": args.target,
        "target_transform": target_transform,
        "features": list(Xtr.columns),
        "n_features": int(Xtr.shape[1]),
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "selected_model": best_name,
        "scoring": args.scoring,
        "clip_predictions": bool(args.clip_predictions and target_name == "rating"),
        "clip_range": [0.0, 5.0] if (args.clip_predictions and target_name == "rating") else None,
        "feature_mode": "identities" if args.use_identities else "numeric_only",
        "input_schema": {
            "numeric_cols": numeric_cols if args.use_identities else list(Xtr.columns),
            "identity_cols": identity_cols if args.use_identities else [],
            "hash_dim": int(args.hash_dim) if args.use_identities else None,
        },
        "test_metrics": _metrics(np.asarray(yte), np.asarray(y_pred)),
        "prediction_interval": pred_interval,
        "cv_results": [asdict(r) for r in results],
    }

    default_model_out, default_report_out = _default_artifacts_for_target(str(args.target))
    model_out = Path(args.save_model) if args.save_model else default_model_out
    report_out = Path(args.save_report) if args.save_report else default_report_out
    model_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_estimator, model_out)
    report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n--- Test metrics ---")
    for k, v in report["test_metrics"].items():
        print(f"{k:>6s}: {v:.4f}")

    print(f"\nModèle sauvegardé: {model_out}")
    print(f"Rapport sauvegardé: {report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
