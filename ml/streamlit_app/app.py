from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# Assure que `ml.*` est importable pour joblib (IdentityHasher est picklé comme ml.src.identity_hasher.IdentityHasher)
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ml.src.identity_hasher  # noqa: F401


APP_ROOT = Path(__file__).resolve().parents[1]  # .../ml
MODEL_PATH = APP_ROOT / "models" / "best_model.joblib"
METRICS_PATH = APP_ROOT / "models" / "metrics.json"
BUDGET_MODEL_PATH = APP_ROOT / "models" / "budget_model.joblib"
BUDGET_METRICS_PATH = APP_ROOT / "models" / "budget_metrics.json"
MERGED_RESULTS_PATH = APP_ROOT.parents[0] / "merged_results.json"  # repo root
REF_DATA_PATH = APP_ROOT / "data" / "final_results_28.json"
TRAIN_PATH = APP_ROOT / "data" / "train.parquet"


@st.cache_resource
def _load_model(model_path: Path):
    return joblib.load(model_path)


def _load_schema(metrics_path: Path) -> tuple[list[str], list[str], list[str]]:
    if not metrics_path.exists():
        return [], [], []
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    schema = payload.get("input_schema")
    if isinstance(schema, dict):
        numeric_cols = schema.get("numeric_cols")
        identity_cols = schema.get("identity_cols")
        if isinstance(numeric_cols, list) and isinstance(identity_cols, list):
            n = [str(c) for c in numeric_cols]
            i = [str(c) for c in identity_cols]
            return n, i, [*n, *i]

    features = payload.get("features")
    if not isinstance(features, list):
        return [], [], []
    f = [str(c) for c in features]
    return f, [], f


def _load_budget_interval(metrics_path: Path) -> dict[str, Any] | None:
    if not metrics_path.exists():
        return None
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    interval = payload.get("prediction_interval")
    if isinstance(interval, dict) and interval.get("method") == "residual_quantiles_log1p":
        return interval
    return None


def _format_money(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    v = float(value)
    if v >= 1_000_000_000:
        return f"{v/1_000_000_000:.2f} Md"
    if v >= 1_000_000:
        return f"{v/1_000_000:.2f} M"
    if v >= 1_000:
        return f"{v/1_000:.1f} k"
    return f"{v:.0f}"


@st.cache_data
def _load_reference_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path)


@st.cache_data
def _load_train_features(path: Path, features: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    cols = [c for c in features if c in df.columns]
    return df[cols].copy()


def _safe_median(series: pd.Series, default: float = 0.0) -> float:
    s = pd.to_numeric(series, errors="coerce")
    if not s.notna().any():
        return float(default)
    val = float(s.median(skipna=True))
    return float(val) if np.isfinite(val) else float(default)


def _q_range(series: pd.Series, fallback_min: float, fallback_max: float) -> tuple[float, float]:
    s = pd.to_numeric(series, errors="coerce")
    s = s[np.isfinite(s)]
    if s.empty:
        return float(fallback_min), float(fallback_max)
    lo = float(s.quantile(0.02))
    hi = float(s.quantile(0.98))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return float(fallback_min), float(fallback_max)
    pad = 0.1 * (hi - lo)
    return float(max(fallback_min, lo - pad)), float(hi + pad)


def _flatten_unique_lists(df: pd.DataFrame, col: str, top_k: int) -> list[str]:
    if df.empty or col not in df.columns:
        return []
    values = df[col].dropna().tolist()
    flat: list[str] = []
    for v in values:
        if isinstance(v, list):
            flat.extend([str(x) for x in v if x is not None and str(x).strip()])
        elif isinstance(v, str) and v.strip():
            flat.append(v.strip())
    if not flat:
        return []
    counts: dict[str, int] = {}
    for name in flat:
        counts[name] = counts.get(name, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [name for name, _ in ranked[:top_k]]


st.set_page_config(page_title="Prédiction de note", layout="centered")
st.title("Prédiction de note (rating 0–5)")

numeric_cols, identity_cols, features = _load_schema(METRICS_PATH)
if not features:
    st.error(
        "Impossible de trouver la liste des features. "
        "Lance d'abord l'entraînement pour générer ml/models/metrics.json."
    )
    st.stop()

if not MODEL_PATH.exists():
    st.error(
        "Modèle introuvable. Lance d'abord l'entraînement pour générer ml/models/best_model.joblib."
    )
    st.stop()

model = _load_model(MODEL_PATH)

budget_model = None
budget_features: list[str] = []
budget_interval = None
if BUDGET_MODEL_PATH.exists() and BUDGET_METRICS_PATH.exists():
    _, _, budget_features = _load_schema(BUDGET_METRICS_PATH)
    budget_interval = _load_budget_interval(BUDGET_METRICS_PATH)
    try:
        budget_model = _load_model(BUDGET_MODEL_PATH)
    except Exception as e:
        budget_model = None
        st.warning(f"Modèle budget détecté mais non chargeable: {e}")

ref_df = _load_reference_df(REF_DATA_PATH)
if ref_df.empty:
    ref_df = _load_reference_df(MERGED_RESULTS_PATH)

train_df = _load_train_features(TRAIN_PATH, features)
stats_df = train_df if not train_df.empty else ref_df

medians: dict[str, float] = {}
for col in features:
    if col in stats_df.columns:
        medians[col] = _safe_median(stats_df[col], default=0.0)
    else:
        medians[col] = 0.0

st.write(
    "Renseigne les caractéristiques de ton film fictif, puis clique sur **Prédire**. "
    "La prédiction est bornée dans l'intervalle [0, 5]."
)

st.subheader("Film de référence (pour suggérer des valeurs réalistes)")
ref_row: dict[str, Any] | None = None
if ref_df.empty:
    st.info("Aucun film de référence disponible (final_results_28.json / merged_results.json introuvable ou vide).")
else:
    df_labels = ref_df[["title", "year"]].copy()
    df_labels["year"] = pd.to_numeric(df_labels["year"], errors="coerce").fillna(0).astype(int)
    df_labels["_label"] = df_labels.apply(lambda r: f"{r['title']} ({r['year']})", axis=1)
    labels = ["Aucun"] + df_labels["_label"].tolist()
    choice = st.selectbox("Choisir un film", options=labels, index=0)
    if choice != "Aucun":
        idx = int(df_labels.index[df_labels["_label"] == choice][0])
        ref_row = {str(k): v for k, v in ref_df.loc[idx].to_dict().items()}


name_options = {
    "directors": _flatten_unique_lists(ref_df, "directors", top_k=200),
    "casting": _flatten_unique_lists(ref_df, "casting", top_k=400),
    "producers": _flatten_unique_lists(ref_df, "producers", top_k=200),
    "writers": _flatten_unique_lists(ref_df, "writers", top_k=200),
    "composer": _flatten_unique_lists(ref_df, "composer", top_k=200),
    "studio": _flatten_unique_lists(ref_df, "studio", top_k=200),
    "languages": _flatten_unique_lists(ref_df, "languages", top_k=150),
    "genres": _flatten_unique_lists(ref_df, "genres", top_k=150),
    "themes": _flatten_unique_lists(ref_df, "themes", top_k=250),
}


def _suggestion_caption(label: str, suggested: object | None) -> None:
    if suggested is None:
        return
    if isinstance(suggested, list):
        if suggested:
            short = ", ".join(map(str, suggested[:12]))
            st.caption(f"Suggestion (référence) — {label}: {short}{'…' if len(suggested) > 12 else ''}")
        return
    st.caption(f"Suggestion (référence) — {label}: {suggested}")


def _optional_slider(
    *,
    feature: str,
    label: str,
    is_int: bool,
    fallback_min: float,
    fallback_max: float,
    suggested: float | None,
) -> tuple[bool, float]:
    use = st.checkbox(f"Utiliser {label}", value=False, key=f"use_{feature}")
    _suggestion_caption(label, suggested)
    if not use:
        return False, float(medians.get(feature, 0.0))

    series = stats_df[feature] if feature in stats_df.columns else pd.Series(dtype=float)
    lo, hi = _q_range(series, fallback_min=fallback_min, fallback_max=fallback_max)
    default = float(medians.get(feature, 0.0))

    if is_int:
        val = st.slider(label, int(lo), int(hi), int(round(default)), key=f"val_{feature}")
        return True, float(val)
    val = st.slider(label, float(lo), float(hi), float(default), key=f"val_{feature}")
    return True, float(val)


def _optional_multiselect_count(
    *,
    feature: str,
    label: str,
    ref_col: str,
) -> tuple[bool, float]:
    use = st.checkbox(f"Utiliser {label}", value=False, key=f"use_{feature}")
    suggested_list: list[str] | None = None
    if ref_row is not None:
        ref_val = ref_row.get(ref_col)
        if isinstance(ref_val, list):
            suggested_list = [str(x) for x in ref_val]
    _suggestion_caption(label, suggested_list)

    if not use:
        return False, float(medians.get(feature, 0.0))

    options = name_options.get(ref_col, [])
    selected = st.multiselect(label, options=options, default=[], key=f"val_{feature}")
    return True, float(len(selected))


def _optional_multiselect_list(
    *,
    feature: str,
    label: str,
    ref_col: str,
) -> tuple[bool, list[str]]:
    use = st.checkbox(f"Utiliser {label}", value=False, key=f"use_{feature}")
    suggested_list: list[str] | None = None
    if ref_row is not None:
        ref_val = ref_row.get(ref_col)
        if isinstance(ref_val, list):
            suggested_list = [str(x) for x in ref_val]
    _suggestion_caption(label, suggested_list)

    if not use:
        return False, ["__MISSING__"]

    options = name_options.get(ref_col, [])
    selected = st.multiselect(label, options=options, default=[], key=f"val_{feature}")
    return True, [str(x) for x in selected]


values: dict[str, object] = {}
used: dict[str, bool] = {}


st.subheader("Chiffres (curseurs, optionnels)")
numeric_specs = [
    ("year", "Année", True, 1880.0, 2100.0),
    ("duration", "Durée (minutes)", True, 0.0, 400.0),
    ("budget", "Budget", False, 0.0, 500_000_000.0),
    ("revenue", "Revenu", False, 0.0, 3_000_000_000.0),
    ("nbr_watched", "Nombre de visionnages", True, 0.0, 1_000_000.0),
    ("nbr_appearence", "Nombre d'apparitions", True, 0.0, 500_000.0),
    ("nbr_likes", "Nombre de likes", True, 0.0, 1_000_000.0),
    ("fans_favoris", "Fans / favoris", True, 0.0, 500_000.0),
]

for feature, label, is_int, fmin, fmax in numeric_specs:
    if feature not in features:
        continue
    suggested = None
    if ref_row is not None:
        ref_val = ref_row.get(feature)
        if ref_val is not None:
            try:
                suggested = float(ref_val)
            except Exception:
                suggested = None
    u, v = _optional_slider(
        feature=feature,
        label=label,
        is_int=is_int,
        fallback_min=fmin,
        fallback_max=fmax,
        suggested=suggested,
    )
    used[feature] = u
    values[feature] = v


st.subheader("Personnes / catégories (menus déroulants, optionnels)")
if identity_cols:
    role_map = [
        ("directors", "Réalisateurs", "directors"),
        ("casting", "Acteurs / casting", "casting"),
        ("producers", "Producteurs", "producers"),
        ("writers", "Scénaristes", "writers"),
        ("composer", "Compositeurs", "composer"),
        ("studio", "Studios", "studio"),
        ("languages", "Langues", "languages"),
        ("genres", "Genres", "genres"),
        ("themes", "Thèmes", "themes"),
    ]
    for feature, label, ref_col in role_map:
        if feature not in features:
            continue
        u, v = _optional_multiselect_list(feature=feature, label=label, ref_col=ref_col)
        used[feature] = u
        values[feature] = v
else:
    role_map = [
        ("directors_count", "Réalisateurs", "directors"),
        ("casting_count", "Acteurs / casting", "casting"),
        ("producers_count", "Producteurs", "producers"),
        ("writers_count", "Scénaristes", "writers"),
        ("composer_count", "Compositeurs", "composer"),
        ("studio_count", "Studios", "studio"),
        ("languages_count", "Langues", "languages"),
        ("genres_count", "Genres", "genres"),
        ("themes_count", "Thèmes", "themes"),
    ]
    for feature, label, ref_col in role_map:
        if feature not in features:
            continue
        u, v = _optional_multiselect_count(feature=feature, label=label, ref_col=ref_col)
        used[feature] = u
        values[feature] = v


for feature in features:
    if feature not in values:
        if identity_cols and feature in identity_cols:
            values[feature] = ["__MISSING__"]
        else:
            values[feature] = float(medians.get(feature, 0.0))
        used[feature] = False


st.divider()
if st.button("Prédire"):
    X = pd.DataFrame([values], columns=features)
    try:
        y_pred = float(np.asarray(model.predict(X)).reshape(-1)[0])
        y_pred = float(np.clip(y_pred, 0.0, 5.0))
        st.success("Prédiction terminée")
        st.metric("Note prédite", f"{y_pred:.2f} / 5")

        if budget_model is not None and budget_features:
            Xb = pd.DataFrame([{k: values.get(k) for k in budget_features}], columns=budget_features)
            try:
                # Le modèle budget a été entraîné sur log1p(budget)
                pred_log = float(np.asarray(budget_model.predict(Xb)).reshape(-1)[0])
                pred_budget = float(np.expm1(pred_log))
                low = high = None
                if isinstance(budget_interval, dict):
                    ql = float(budget_interval.get("resid_q_low", 0.0))
                    qh = float(budget_interval.get("resid_q_high", 0.0))
                    low = float(np.expm1(pred_log + ql))
                    high = float(np.expm1(pred_log + qh))
                st.subheader("Suggestion de budget")
                if low is not None and high is not None:
                    st.write(
                        f"Budget estimé: **{_format_money(pred_budget)}** (fourchette ~80%: "
                        f"**{_format_money(low)} – {_format_money(high)}**)."
                    )
                else:
                    st.write(f"Budget estimé: **{_format_money(pred_budget)}** (intervalle indisponible).")
            except Exception as e:
                st.warning(f"Impossible de calculer la suggestion de budget: {e}")

        used_cols = [c for c in features if used.get(c)]
        if used_cols:
            st.caption("Champs utilisés: " + ", ".join(used_cols))
        else:
            st.caption("Aucun champ activé: valeurs neutres (médianes) utilisées.")
    except Exception as e:
        st.error(f"Erreur pendant la prédiction: {e}")
