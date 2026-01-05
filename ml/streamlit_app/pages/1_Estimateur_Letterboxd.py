from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# Bootstrap paths so `ml.*` is importable (IdentityHasher is pickled as ml.src.identity_hasher.IdentityHasher)
HERE = Path(__file__).resolve()
ML_DIR = HERE.parents[2]  # .../ml
REPO_ROOT = ML_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ml.src.identity_hasher  # noqa: F401


APP_ROOT = ML_DIR
MODEL_PATH = APP_ROOT / "models" / "best_model.joblib"
METRICS_PATH = APP_ROOT / "models" / "metrics.json"
BUDGET_MODEL_PATH = APP_ROOT / "models" / "budget_model.joblib"
BUDGET_METRICS_PATH = APP_ROOT / "models" / "budget_metrics.json"
MERGED_RESULTS_PATH = REPO_ROOT / "merged_results.json"
REF_DATA_PATH = APP_ROOT / "data" / "final_results_28.json"
TRAIN_PATH = APP_ROOT / "data" / "train.parquet"

# Import shared helpers from the central module to keep this page presentation-only.
from ml.streamlit_app.helpers import Helpers as H
 


# ------------ Streamlit page ------------

st.set_page_config(page_title="Estimateur — Prédiction de note Letterboxd", layout="wide")

LETTERBOXD_COLORS = {
    "bg": "#0b0b0b",
    "text": "#E6E6E6",
    "accent": "#2AB44B",
    "accent_dark": "#1B5E20",
}

st.markdown(
    f"""
    <style>
    :root {{
        --lb-bg: {LETTERBOXD_COLORS['bg']};
        --lb-text: {LETTERBOXD_COLORS['text']};
        --lb-accent: {LETTERBOXD_COLORS['accent']};
        --lb-accent-dark: {LETTERBOXD_COLORS['accent_dark']};
    }}
    .stApp, .block-container {{
        background-color: var(--lb-bg) !important;
        color: var(--lb-text) !important;
    }}
    .stButton>button {{
        background-color: var(--lb-accent) !important;
        color: #fff !important;
        border: none !important;
    }}
    .stMetric {{
        color: var(--lb-text) !important;
    }}
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
        color: var(--lb-accent) !important;
    }}
    .stSlider > div > div {{
        color: var(--lb-text) !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Prédire la note Letterboxd (0–5)")
st.markdown(
    "Renseigne les caractéristiques de ton film fictif, puis clique sur **Prédire**. "
)

# Presets / Templates
PRESETS: dict[str, dict] = {
    "Petit budget": {
        "budget": 500_000.0,
        "revenue": 2_000_000.0,
        "duration": 85.0,
        "year": 2018.0,
        "genres": ["Drama"],
    },
    "Indépendant": {
        "budget": 1_000_000.0,
        "revenue": 5_000_000.0,
        "duration": 95.0,
        "year": 2016.0,
        "genres": ["Drama", "Art House"],
    },
    "Blockbuster": {
        "budget": 150_000_000.0,
        "revenue": 800_000_000.0,
        "duration": 130.0,
        "year": 2022.0,
        "genres": ["Action", "Adventure"],
    },
}

# Film de référence
st.sidebar.subheader("Film de référence")

numeric_cols, identity_cols, features = H.load_schema(METRICS_PATH)
if not features:
    st.error(
        "Impossible de trouver la liste des features. "
        "Lance d'abord l'entraînement pour générer ml/models/metrics.json."
    )
    st.stop()

if not MODEL_PATH.exists():
    st.error("Modèle introuvable. Lance d'abord l'entraînement pour générer ml/models/best_model.joblib.")
    st.stop()

model = H.load_model(MODEL_PATH)

budget_model = None
budget_features: list[str] = []
budget_interval = None
if BUDGET_MODEL_PATH.exists() and BUDGET_METRICS_PATH.exists():
    _, _, budget_features = H.load_schema(BUDGET_METRICS_PATH)
    budget_interval = H.load_budget_interval(BUDGET_METRICS_PATH)
    try:
        budget_model = H.load_model(BUDGET_MODEL_PATH)
    except Exception as e:
        budget_model = None
        st.warning(f"Modèle budget détecté mais non chargeable: {e}")

ref_df = H.load_reference_df(REF_DATA_PATH)
if ref_df.empty:
    # Fallback only (some environments may not ship final_results_28.json at runtime).
    ref_df = H.load_reference_df(MERGED_RESULTS_PATH)

NAME_OPTION_COLS = [
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

train_cols = sorted(set([*features, *NAME_OPTION_COLS]))
train_df = H.load_train_features(TRAIN_PATH, train_cols)
stats_df = train_df if not train_df.empty else ref_df

medians: dict[str, float] = {}
for col in features:
    if col in stats_df.columns:
        medians[col] = H.safe_median(stats_df[col], default=0.0)
    else:
        medians[col] = 0.0


st.sidebar.caption(
    "Note: l'estimation du budget est fournie à titre indicatif et sert d'aide pour affiner l'estimation de la note finale (elle n'est pas une vérité)."
)

ref_row: dict[str, Any] | None = None
ref_lb_url: str | None = None
ref_tmdb_id: int | None = None
prefill_from_ref = st.sidebar.checkbox("Préremplir depuis le film de référence", value=False, key="prefill_from_ref")
if ref_df.empty:
    st.sidebar.info("Aucun film de référence disponible (final_results_28.json / merged_results.json introuvable ou vide).")
else:
    df_labels = ref_df[["title", "year"]].copy()
    df_labels["year"] = pd.to_numeric(df_labels["year"], errors="coerce").fillna(0).astype(int)
    df_labels["_label"] = df_labels.apply(lambda r: f"{r['title']} ({r['year']})", axis=1)
    labels = ["Aucun"] + df_labels["_label"].tolist()
    choice = st.sidebar.selectbox("Choisir un film", options=labels, index=0)
    if choice != "Aucun":
        idx = int(df_labels.index[df_labels["_label"] == choice][0])
        ref_row = {str(k): v for k, v in ref_df.loc[idx].to_dict().items()}

    # Poster + metadata
    if ref_row is not None:
        lb_url = None
        for k in ("url", "letterboxd_url", "letterboxd", "link"):
            v = ref_row.get(k)
            if isinstance(v, str) and v.strip():
                lb_url = v.strip()
                break

        ref_lb_url = lb_url

        if lb_url and lb_url.startswith("http"):
            poster_url = H.get_letterboxd_poster_url(lb_url)
            if poster_url:
                try:
                    st.sidebar.image(poster_url, width=160)
                except Exception:
                    pass
            else:
                st.sidebar.caption("Poster: indisponible (non trouvé sur Letterboxd).")
                
        title = ref_row.get("title") or ""
        year = ref_row.get("year") or ""
        st.sidebar.markdown(f"**{title}**  ")
        st.sidebar.caption(f"Année: {year}")        
        if lb_url:
            st.sidebar.caption(f"Lien Letterboxd: {lb_url}")
                
            # Option A — TMDB: actor headshots (requires TMDB_API_KEY)
            api_key = H.tmdb_api_key()
            if api_key:
                try:
                    lb_html = H.fetch_html(lb_url)
                    tmdb_id = H.extract_tmdb_movie_id(lb_html)
                except Exception:
                    tmdb_id = None

                ref_tmdb_id = tmdb_id

                if tmdb_id:
                    cast = H.tmdb_movie_cast(tmdb_id, top_n=3)
                    if cast:
                        st.sidebar.subheader("Casting (TMDB)")
                        cols_cast = st.sidebar.columns(3)
                        for i, member in enumerate(cast):
                            with cols_cast[i % 3]:
                                try:
                                    st.image(member["profile_url"], use_container_width=True)
                                except Exception:
                                    pass
                                name = str(member.get("name") or "")
                                role = str(member.get("character") or "")
                                st.caption(name)
                                # if role:
                                #     st.caption(role)
                    else:
                        st.sidebar.caption("Casting TMDB: indisponible.")
                else:
                    st.sidebar.caption("TMDB: ID du film introuvable sur la page Letterboxd.")
            else:
                st.sidebar.caption(
                    "Pour voir les photos des acteurs, configure TMDB_API_KEY (Streamlit secrets ou variable d'environnement)."
                )       

        def _join_if_list(x):
            if isinstance(x, list):
                return ", ".join([str(i) for i in x[:8]]) + ("…" if len(x) > 8 else "")
            return str(x) if x is not None else ""

        directors = _join_if_list(ref_row.get("directors") or ref_row.get("director"))
        genres = _join_if_list(ref_row.get("genres"))
        duration = ref_row.get("duration")
        if directors:
            st.sidebar.text(f"Réalisateur(s): {directors}")
        if genres:
            st.sidebar.text(f"Genres: {genres}")
        if duration:
            st.sidebar.text(f"Durée: {duration} minutes")

# Preset selector (sidebar)

st.sidebar.subheader("Presets / Templates")
preset_choice = st.sidebar.selectbox("Choisir un preset", options=["Aucun"] + list(PRESETS.keys()), index=0)
preset_values: dict[str, object] = PRESETS.get(preset_choice, {}) if preset_choice != "Aucun" else {}

# Widget state suffix: when prefill is enabled, keys depend on the selected reference movie.
# This makes widgets initialize with reference defaults without overwriting user edits until the reference changes.
if prefill_from_ref and isinstance(ref_row, dict):
    _ref_token = f"{ref_row.get('title')}_{ref_row.get('year')}_{'id' if identity_cols else 'count'}"
else:
    _ref_token = "manual"
WIDGET_STATE_SUFFIX = H.sanitize_widget_suffix(_ref_token)


# Note: train.parquet contains engineered "*_count" features, not the original name lists.
# So we build the option lists from the reference JSON.
names_df = ref_df

name_options = {
    # top_k=0 => "all" (users expect to be able to type-search any known entry).
    "directors": H.flatten_unique_lists(names_df, "directors", top_k=0),
    "casting": H.flatten_unique_lists(names_df, "casting", top_k=0),
    "producers": H.flatten_unique_lists(names_df, "producers", top_k=0),
    "writers": H.flatten_unique_lists(names_df, "writers", top_k=0),
    "composer": H.flatten_unique_lists(names_df, "composer", top_k=0),
    "studio": H.flatten_unique_lists(names_df, "studio", top_k=0),
    "languages": H.flatten_unique_lists(names_df, "languages", top_k=0),
    "genres": H.flatten_unique_lists(names_df, "genres", top_k=0),
    "themes": H.flatten_unique_lists(names_df, "themes", top_k=0),
}


 


values: dict[str, object] = {}
used: dict[str, bool] = {}

st.subheader("Chiffres (curseurs, optionnels)")
nums_container = st.container()

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

cols = nums_container.columns(2)
col_idx = 0
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
    with cols[col_idx % 2]:
        u, v = H.optional_slider(
            feature=feature,
            label=label,
            is_int=is_int,
            fallback_min=fmin,
            fallback_max=fmax,
            suggested=suggested,
            preset_values=preset_values,
            medians=medians,
            stats_df=stats_df,
            prefill_from_ref=prefill_from_ref,
            ref_row=ref_row,
            state_suffix=WIDGET_STATE_SUFFIX,
        )
    used[feature] = u
    values[feature] = v
    col_idx += 1

st.subheader("Personnes / catégories (menus déroulants, optionnels)")
people_container = st.container()
cols_p = people_container.columns(2)
colp_idx = 0

role_map_identity = [
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
role_map_count = [
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
role_map = role_map_identity if identity_cols else role_map_count
for feature, label, ref_col in role_map:
    if feature not in features:
        continue
    with cols_p[colp_idx % 2]:
        if identity_cols:
            u, v = H.optional_multiselect_list(
                feature=feature,
                label=label,
                ref_col=ref_col,
                preset_values=preset_values,
                medians=medians,
                ref_row=ref_row,
                prefill_from_ref=prefill_from_ref,
                state_suffix=WIDGET_STATE_SUFFIX,
                name_options=name_options,
                ref_tmdb_id=ref_tmdb_id,
            )
        else:
            u, v = H.optional_multiselect_count(
                feature=feature,
                label=label,
                ref_col=ref_col,
                preset_values=preset_values,
                medians=medians,
                ref_row=ref_row,
                prefill_from_ref=prefill_from_ref,
                state_suffix=WIDGET_STATE_SUFFIX,
                name_options=name_options,
                ref_tmdb_id=ref_tmdb_id,
            )
    used[feature] = u
    values[feature] = v
    colp_idx += 1

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
                pred_log = float(np.asarray(budget_model.predict(Xb)).reshape(-1)[0])
                pred_budget = float(np.expm1(pred_log))
                low = high = None
                if isinstance(budget_interval, dict):
                    ql = float(budget_interval.get("resid_q_low", 0.0))
                    qh = float(budget_interval.get("resid_q_high", 0.0))
                    low = float(np.expm1(pred_log + ql))
                    high = float(np.expm1(pred_log + qh))

                st.subheader("Suggestion de budget (aide pour la note)")
                if low is not None and high is not None:
                    st.write(
                        f"Budget estimé: **{H.format_money(pred_budget)}** (fourchette ~80%: "
                        f"**{H.format_money(low)} – {H.format_money(high)}**)."
                    )
                else:
                    st.write(f"Budget estimé: **{H.format_money(pred_budget)}** (intervalle indisponible).")
            except Exception as e:
                st.warning(f"Impossible de calculer la suggestion de budget: {e}")

        used_cols = [c for c in features if used.get(c)]
        if used_cols:
            st.caption("Champs utilisés: " + ", ".join(used_cols))
        else:
            st.caption("Aucun champ activé: valeurs neutres (médianes) utilisées.")
    except Exception as e:
        st.error(f"Erreur pendant la prédiction: {e}")
