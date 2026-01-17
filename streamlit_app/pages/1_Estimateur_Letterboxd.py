from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import streamlit as st


# Bootstrap paths so `ml.*` is importable (IdentityHasher is pickled as ml.src.identity_hasher.IdentityHasher)
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]  # .../root of project
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.utils.identity_hasher  # noqa: F401


APP_ROOT = REPO_ROOT / "ml"
MODEL_PATH = APP_ROOT / "models" / "best_model.joblib"
METRICS_PATH = APP_ROOT / "models" / "metrics.json"
BUDGET_MODEL_PATH = APP_ROOT / "models" / "budget_model.joblib"
BUDGET_METRICS_PATH = APP_ROOT / "models" / "budget_metrics.json"
MERGED_RESULTS_PATH = REPO_ROOT / "merged_results.json"
REF_DATA_PATH = APP_ROOT / "data" / "final_results_28.json"
TRAIN_PATH = APP_ROOT / "data" / "train.parquet"

# Import shared helpers from the central module to keep this page presentation-only.
from streamlit_app.helpers import Helpers as H
 


# ------------ Streamlit page ------------

st.set_page_config(page_title="Estimateur — Prédiction de note Letterboxd", layout="wide")
H.apply_letterboxd_theme()

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

# RMSE (test) for a rough prediction interval (metrics.json is produced at training time)
rating_rmse = H.load_test_rmse(METRICS_PATH)

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


def _predict_budget_with_artifact(artifact: object, values: dict[str, object]) -> tuple[float, float]:
    """Fallback when the loaded budget model is a ToolModelArtifact without predict().

    Returns (pred_log_space, pred_budget).
    """

    model = getattr(artifact, "model", None)
    hasher = getattr(artifact, "identity_hasher", None)
    numeric_cols = list(getattr(artifact, "numeric_cols", []) or [])
    identity_cols = list(getattr(artifact, "identity_cols", []) or [])
    medians = getattr(artifact, "numeric_medians", {}) or {}
    transform_info = getattr(artifact, "target_transform", None)
    clip_range = getattr(artifact, "clip_range", None)

    if model is None or hasher is None:
        raise ValueError("artifact missing model or hasher")

    def _safe_float(val: Any, default: float) -> float:
        try:
            return float(val)
        except Exception:
            return default

    numeric_row: list[float] = []
    for col in numeric_cols:
        if col.endswith("_log1p"):
            base = col[: -len("_log1p")]
            raw = _safe_float(values.get(base), medians.get(col, 0.0))
            raw = max(0.0, raw)
            numeric_row.append(float(np.log1p(raw)))
        else:
            numeric_row.append(_safe_float(values.get(col), medians.get(col, 0.0)))

    numeric_matrix = np.array([numeric_row], dtype=float)
    id_payload: dict[str, list[str]] = {}
    for col in identity_cols:
        val = values.get(col)
        if isinstance(val, list):
            cleaned = [str(x).strip() for x in val if x is not None and str(x).strip()]
            id_payload[col] = cleaned or ["__MISSING__"]
        elif val is None:
            id_payload[col] = ["__MISSING__"]
        else:
            s = str(val).strip()
            id_payload[col] = [s] if s else ["__MISSING__"]

    id_df = pd.DataFrame([id_payload]) if identity_cols else pd.DataFrame()
    hashed = hasher.transform(id_df) if identity_cols else np.zeros((1, 0), dtype=float)
    Xb = np.hstack([numeric_matrix, hashed])

    pred_raw = float(np.asarray(model.predict(Xb)).reshape(-1)[0])
    pred_budget = pred_raw
    if isinstance(transform_info, dict) and transform_info.get("name") == "log1p":
        pred_budget = float(np.expm1(pred_raw))
    if isinstance(clip_range, (tuple, list)) and len(clip_range) == 2:
        try:
            lo, hi = float(clip_range[0]), float(clip_range[1])
            pred_budget = float(np.clip(pred_budget, lo, hi))
        except Exception:
            pass
    return pred_raw, pred_budget

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
ref_idx: int | None = None
ref_lb_url: str | None = None
ref_tmdb_id: int | None = None
ref_poster_url: str | None = None
ref_backdrop_url: str | None = None
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
        ref_idx = idx
        ref_row = {str(k): v for k, v in ref_df.loc[idx].to_dict().items()}

    # Collect reference URLs / IDs (rendering will be done in the main area, not the sidebar)
    if ref_row is not None:
        lb_url = None
        for k in ("url", "letterboxd_url", "letterboxd", "link"):
            v = ref_row.get(k)
            if isinstance(v, str) and v.strip():
                lb_url = v.strip()
                break

        ref_lb_url = lb_url

        if lb_url and lb_url.startswith("http"):
            # Try to extract TMDB id from the Letterboxd page (no TMDB key required for this)
            try:
                lb_html = H.fetch_html(lb_url)
                ref_tmdb_id = H.extract_tmdb_movie_id(lb_html)
                print(f"[DEBUG] TMDB ID extrait de {lb_url}: {ref_tmdb_id}")
            except Exception as e:
                ref_tmdb_id = None
                print(f"[DEBUG] Erreur extraction TMDB ID: {e}")

            # Privilégier TMDB pour les posters (meilleure qualité)
            api_key = H.tmdb_api_key()
            print(f"[DEBUG] TMDB API key present: {bool(api_key)}")
            if api_key and ref_tmdb_id:
                imgs = H.tmdb_movie_images(int(ref_tmdb_id))
                print(f"[DEBUG] Images TMDB récupérées: {imgs}")
                if isinstance(imgs, dict):
                    # Utiliser TMDB en priorité pour poster et backdrop
                    ref_poster_url = imgs.get("poster_url_large") or imgs.get("poster_url") or ref_poster_url
                    ref_backdrop_url = imgs.get("backdrop_url_large") or imgs.get("backdrop_url") or ref_backdrop_url
                    print(f"[DEBUG] Poster TMDB: {ref_poster_url}")
            
            # Fallback sur Letterboxd uniquement si TMDB n'a pas de poster
            if not ref_poster_url:
                poster_url = H.get_letterboxd_poster_url(lb_url)
                if poster_url:
                    ref_poster_url = poster_url
                    print(f"[DEBUG] Poster Letterboxd (fallback): {poster_url}")

# Header (main area): Letterboxd-like film page header when a reference movie is selected
if ref_row is not None and ref_poster_url:
    def _join_if_list(x):
        if isinstance(x, list):
            return ", ".join([str(i) for i in x[:4]]) + ("…" if len(x) > 4 else "")
        return str(x) if x is not None else ""

    facts: list[str] = []
    yr = ref_row.get("year")
    if yr is not None:
        facts.append(f"{int(float(yr))}" if str(yr).strip() else "")
    dur = ref_row.get("duration")
    if dur is not None and str(dur).strip():
        facts.append(f"{int(float(dur))} min")
    dirs = _join_if_list(ref_row.get("directors") or ref_row.get("director"))
    if dirs:
        facts.append(f"Réal.: {dirs}")
    gens = _join_if_list(ref_row.get("genres"))
    if gens:
        facts.append(gens)
    facts = [f for f in facts if f]

    cast_for_hero = None
    if ref_tmdb_id and H.tmdb_api_key():
        cast_for_hero = H.tmdb_movie_cast(int(ref_tmdb_id), top_n=5)

    H.render_reference_film_hero(
        title=str(ref_row.get("title") or ""),
        year=ref_row.get("year"),
        poster_url=str(ref_poster_url),
        background_url=(str(ref_backdrop_url) if ref_backdrop_url else None),
        facts=facts,
        cast=cast_for_hero,
        letterboxd_url=ref_lb_url,
    )

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

# Build cache of actor photos from the database (uses cast_profiles field if present)
actor_photos_cache = H.build_actor_photos_cache(ref_df)
print(f"[DEBUG] Cache photos acteurs : {len(actor_photos_cache)} entrées")

# Feature input widgets
values: dict[str, object] = {}
used: dict[str, bool] = {}

st.subheader("Chiffres (curseurs, optionnels)")
nums_container = st.container()

numeric_specs = [
    ("year", "Année", True, 1885.0, 2030.0),
    ("duration", "Durée (minutes)", True, 0.0, 400.0),
    ("budget", "Budget", False, 0.0, 500_000_000.0),
    ("revenue", "Revenu", False, 0.0, 3_000_000_000.0),
    ("nbr_watched", "Nombre de visionnages", True, 0.0, 10_000_000.0),
    ("nbr_appearence", "Nombre d'apparitions", True, 0.0, 1_000_000.0),
    ("nbr_likes", "Nombre de likes", True, 0.0, 5_000_000.0),
    ("fans_favoris", "Fans / favoris", True, 0.0, 600_000.0),
]

numeric_labels = {feat: lbl for feat, lbl, *_ in numeric_specs}
numeric_int_features = {feat for feat, _, is_int, *_ in numeric_specs if bool(is_int)}

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
                actor_photos_cache=actor_photos_cache,
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
                actor_photos_cache=actor_photos_cache,
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

# Suggestion de budget (liée aux infos du film/personnes), seulement si le budget n'est pas renseigné.
if budget_model is not None and budget_features and (not used.get("budget", False)):
    # S'assurer que toutes les features du modèle budget existent dans `values`.
    for k in budget_features:
        if k not in values:
            if identity_cols and k in identity_cols:
                values[k] = ["__MISSING__"]
            else:
                values[k] = float(medians.get(k, 0.0))

    st.subheader("Suggestion de budget (si manquant)")
    try:
        if hasattr(budget_model, "predict"):
            Xb = pd.DataFrame([{k: values.get(k) for k in budget_features}], columns=budget_features)
            pred_log = float(np.asarray(budget_model.predict(Xb)).reshape(-1)[0])
            pred_budget = float(np.expm1(pred_log))
        else:
            pred_log, pred_budget = _predict_budget_with_artifact(budget_model, values)

        low = high = None
        if isinstance(budget_interval, dict):
            ql = float(budget_interval.get("resid_q_low", 0.0))
            qh = float(budget_interval.get("resid_q_high", 0.0))
            low = float(np.expm1(pred_log + ql))
            high = float(np.expm1(pred_log + qh))

        if low is not None and high is not None:
            st.write(
                f"Budget estimé: **{H.format_money(pred_budget)}** (fourchette ~80%: "
                f"**{H.format_money(low)} – {H.format_money(high)}**)."
            )
        else:
            st.write(f"Budget estimé: **{H.format_money(pred_budget)}** (intervalle indisponible).")
    except Exception as e:
        st.warning(f"Impossible de calculer la suggestion de budget: {e}")

st.divider()

# Persist prediction across Streamlit reruns (changing any widget reruns the script)
if "_last_prediction" not in st.session_state:
    st.session_state["_last_prediction"] = None
if "_last_prediction_payload" not in st.session_state:
    st.session_state["_last_prediction_payload"] = None
if "_last_prediction_error" not in st.session_state:
    st.session_state["_last_prediction_error"] = None

predict_clicked = st.button("Prédire")
if predict_clicked:
    X = pd.DataFrame([values], columns=features)
    try:
        y_pred = float(np.asarray(model.predict(X)).reshape(-1)[0])
        y_pred = float(np.clip(y_pred, 0.0, 5.0))

        st.session_state["_last_prediction"] = y_pred
        st.session_state["_last_prediction_error"] = None
        st.session_state["_last_prediction_payload"] = {
            "values": dict(values),
            "used": dict(used),
        }
    except Exception as e:
        st.session_state["_last_prediction"] = None
        st.session_state["_last_prediction_payload"] = None
        st.session_state["_last_prediction_error"] = str(e)

if st.session_state.get("_last_prediction_error"):
    st.error(f"Erreur pendant la prédiction: {st.session_state['_last_prediction_error']}")

if st.session_state.get("_last_prediction") is not None and st.session_state.get("_last_prediction_payload"):
    y_pred = float(st.session_state["_last_prediction"])
    payload = st.session_state["_last_prediction_payload"]
    last_values = dict(payload.get("values") or {})
    last_used = dict(payload.get("used") or {})

    st.success("Prédiction terminée")
    st.metric("Note prédite", f"{y_pred:.2f} / 5")

    st.caption(
        "Ce score est une **prédiction du modèle** (pas une vérité). Les indicateurs ci-dessous aident à savoir "
        "si ta saisie ressemble aux films du dataset et si l'estimation est plutôt stable."
    )

    # Fiabilité: (1) complétude des champs + (2) intervalle (approx) basé sur RMSE
    q: dict[str, Any] = {}
    completeness = 0.0
    comp_level = "unknown"
    comp_msg = ""
    try:
        q = H.prediction_quality_info(last_used, numeric_cols, identity_cols)
        comp_raw: Any = q.get("completeness")
        try:
            completeness = float(comp_raw) if comp_raw is not None else 0.0
        except Exception:
            completeness = 0.0
        comp_level = str(q.get("level") or "unknown")
        comp_msg = str(q.get("message") or "")
    except Exception:
        pass

    interval = None
    if rating_rmse is not None:
        interval = H.approx_prediction_interval_from_rmse(y_pred, float(rating_rmse), coverage=0.8, clip=(0.0, 5.0))

    ood = None
    try:
        ood = H.ood_numeric_info(last_values, last_used, stats_df, numeric_cols, q_low=0.01, q_high=0.99)
    except Exception:
        ood = None

    # Combine indicators conservatively (take the worst)
    def _level_score(lvl: str) -> int:
        s = (lvl or "").lower().strip()
        if s == "high":
            return 2
        if s == "medium":
            return 1
        if s == "low":
            return 0
        return 1

    int_level = "unknown"
    int_msg = ""
    if isinstance(interval, dict) and interval.get("width") is not None:
        width_val = interval.get("width")
        try:
            w = float(width_val) if width_val is not None else float("nan")
        except Exception:
            w = float("nan")
        if np.isfinite(w):
            # On a 0–5 scale: ~0.6 width is fairly tight; >1.0 is wide.
            if w <= 0.6:
                int_level = "high"
                int_msg = "Intervalle serré."
            elif w <= 1.0:
                int_level = "medium"
                int_msg = "Intervalle modéré."
            else:
                int_level = "low"
                int_msg = "Intervalle large."

    ood_level = "unknown"
    ood_msg = ""
    if isinstance(ood, dict):
        ood_level = str(ood.get("level") or "unknown")
        ood_msg = str(ood.get("message") or "")

    overall_score = min(_level_score(comp_level), _level_score(int_level), _level_score(ood_level))
    pct = int(round(100.0 * float(completeness)))

    interval_text = ""
    if isinstance(interval, dict) and interval.get("low") is not None and interval.get("high") is not None:
        try:
            low_val = interval.get("low")
            high_val = interval.get("high")
            low = float(low_val) if low_val is not None else float("nan")
            high = float(high_val) if high_val is not None else float("nan")
            interval_text = f"Intervalle (~80%, approx.): **{low:.2f} – {high:.2f}**"
        except Exception:
            interval_text = ""
    else:
        interval_text = "Intervalle: indisponible (RMSE manquante)."

    ood_text = ""
    if isinstance(ood, dict):
        n_checked_raw: Any = ood.get("n_checked")
        n_out_raw: Any = ood.get("n_outside")
        try:
            n_checked = int(float(n_checked_raw)) if n_checked_raw is not None else 0
        except Exception:
            n_checked = 0
        try:
            n_out = int(float(n_out_raw)) if n_out_raw is not None else 0
        except Exception:
            n_out = 0

        if n_checked > 0:
            if n_out == 0:
                ood_text = "Valeurs: dans les plages usuelles."
            else:
                ood_text = f"Valeurs: **{n_out}** atypique(s) vs dataset (quantiles 1%–99%)."

    base = f"Fiabilité (indicateurs): **{pct}%** de champs fournis. {interval_text}"
    if ood_text:
        base = base + " " + ood_text
    detail = ""
    if comp_msg or int_msg or ood_msg:
        detail = f"\n\n_{comp_msg} {int_msg} {ood_msg}_"

    if overall_score >= 2:
        st.success(base + detail)
    elif overall_score == 1:
        st.info(base + detail)
    else:
        st.warning(base + detail)

    if interval_text and (rating_rmse is not None):
        st.caption(
            "L'intervalle est **une approximation statistique** (calculée à partir des erreurs observées pendant l'entraînement). "
            "Il ne garantit pas qu'un film précis tombera dans cette plage."
        )

    # Similarités top 3 avec explications détaillées + vue visuelle
    with st.expander("Similarités (top 3)", expanded=False):
        # Inclure le rating prédit dans les valeurs pour la similarité
        values_with_rating = dict(last_values)
        values_with_rating["rating"] = y_pred
        sims = H.similar_movies_with_explanations(ref_df, values_with_rating, numeric_cols, identity_cols, top_n=4)

        # Exclure le film de référence des résultats
        if ref_idx is not None:
            sims = [s for s in sims if s.get("idx") != ref_idx]
        elif ref_row is not None:
            ref_title = str(ref_row.get("title") or "").strip().lower()
            ref_year = str(ref_row.get("year") or "").strip()
            sims = [s for s in sims if not (
                str(s.get("title") or "").strip().lower() == ref_title
                and str(s.get("year") or "").strip() == ref_year
            )]
        if not sims:
            st.info("Aucune similarité calculable (données insuffisantes).")
        else:
            # Onglets: Vue visuelle | Détails complets
            tab_visual, tab_details = st.tabs(["🎬 Vue visuelle", "📋 Détails complets"])
            
            with tab_visual:
                st.markdown("##### Aperçu visuel des films similaires")
                # Vue visuelle simplifiée avec posters
                cols = st.columns(min(3, len(sims)))
                for i, sim in enumerate(sims[:3]):
                    with cols[i % len(cols)]:
                        title = sim.get("title") or "(Titre manquant)"
                        year_raw = sim.get("year")
                        year = int(year_raw) if isinstance(year_raw, (int, float)) else None
                        sim_pct = sim.get("similarity_pct")
                        url_raw = sim.get("url")
                        url = str(url_raw) if isinstance(url_raw, str) else None
                        poster_raw = sim.get("poster_url")
                        poster = str(poster_raw) if isinstance(poster_raw, str) else None
                        label = f"{title} ({year})" if year is not None else title
                        st.markdown(f"**#{i+1} — {label}**")
                        st.metric("Similarité", f"{sim_pct}%")
                        if poster:
                            try:
                                st.image(str(poster), width=160, caption="Affiche")
                            except Exception:
                                pass
                        if url:
                            st.link_button("Fiche Letterboxd", url, use_container_width=True)
            
            with tab_details:
                st.markdown("##### Films similaires à votre saisie")
                
                # Construire une vue détaillée de chaque film
                for i, sim in enumerate(sims):
                    with st.container():
                        title = sim.get("title") or "(Titre manquant)"
                        year_raw = sim.get("year")
                        year = int(year_raw) if isinstance(year_raw, (int, float)) else None
                        sim_pct = sim.get("similarity_pct")
                        url_raw = sim.get("url")
                        url = str(url_raw) if isinstance(url_raw, str) else None
                        feats_raw = sim.get("features")
                        feats = list(feats_raw) if isinstance(feats_raw, list) else []
                        label = f"{title} ({year})" if year is not None else title
                        
                        # En-tête du film
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.markdown(f"### #{i+1} — {label}")
                        with col2:
                            st.metric("Similarité", f"{sim_pct}%")
                        with col3:
                            if url:
                                st.link_button("Letterboxd", url)
                        
                        # Tableau détaillé avec 5 colonnes
                        if feats:
                            st.markdown("##### Comparaison détaillée")
                            table_data = []
                            for f in feats:
                                name = str(f.get("name") or "")
                                typ = str(f.get("type") or "")
                                sim_val = f.get("similarity")
                                det = f.get("details") or {}
                                status = f.get("status", "unknown")
                                
                                # Valeur utilisateur
                                if status == "missing":
                                    user_value = "Manquant"
                                elif typ == "numeric":
                                    user_value = det.get("user")
                                else:  # identity
                                    user_count = det.get("user_count", 0)
                                    user_value = f"{user_count} item(s)"
                                
                                # Valeur film
                                if status == "missing":
                                    film_value = "Manquant"
                                elif typ == "numeric":
                                    film_value = det.get("ref")
                                else:  # identity
                                    ref_count = det.get("ref_count", 0)
                                    film_value = f"{ref_count} item(s)"
                                
                                # Calcul de diff (%) pour numériques
                                diff_display = "—"
                                if status == "ok" and typ == "numeric":
                                    user_v = det.get("user")
                                    ref_v = det.get("ref")
                                    if user_v is not None and ref_v is not None and ref_v != 0:
                                        diff_pct = abs(user_v - ref_v) / abs(ref_v) * 100
                                        diff_display = f"{diff_pct:.1f}%"
                                elif status == "ok" and typ == "identity":
                                    # Pour les listes, diff = 100 - similarité
                                    diff_display = f"{100 - sim_val:.1f}%"
                                
                                # Couleur/emoji du status
                                if status == "ok":
                                    status_display = f"🟢 {sim_val}%"
                                elif status == "missing":
                                    status_display = "🔴 0%"
                                elif status == "empty":
                                    status_display = "⚠️ 0%"
                                elif status == "no_overlap":
                                    status_display = f"🟠 {sim_val}%"
                                else:
                                    status_display = "❓ N/A"
                                
                                table_data.append({
                                    "Critère": name,
                                    "Votre valeur": str(user_value) if user_value is not None else "—",
                                    "Valeur film": str(film_value) if film_value is not None else "—",
                                    "Similarité": status_display,
                                    "Différence (%)": diff_display,
                                })
                            
                            df_comparison = pd.DataFrame(table_data)
                            st.dataframe(
                                df_comparison,
                                use_container_width=True,
                                column_config={
                                    "Critère": st.column_config.TextColumn("Critère", width="medium"),
                                    "Votre valeur": st.column_config.TextColumn("Votre valeur", width="medium"),
                                    "Valeur film": st.column_config.TextColumn("Valeur film", width="medium"),
                                    "Similarité": st.column_config.TextColumn("Similarité", width="small"),
                                    "Différence (%)": st.column_config.TextColumn("Différence (%)", width="small"),
                                },
                                hide_index=True,
                            )
                        st.divider()
            
    # Explicabilité (locale): neutralisation d'un champ à la fois
    with st.expander("Explicabilité (effets locaux)", expanded=False):
        st.caption(
            "Idée simple : on **modifie un seul champ à la fois** (en le remplaçant par une valeur ‘neutre’ : médiane, ou `__MISSING__`) "
            "et on regarde comment la prédiction change.\n\n"
            "À lire comme : \"**Dans le modèle**, ce champ pousse la note vers le haut/bas **pour ce film-là**\".\n\n"
            "Ce n'est **pas** une preuve de causalité dans la vraie vie."
        )

        # Labels for nicer display
        people_labels = {feat: lbl for feat, lbl, _ in role_map}
        feature_labels = dict(numeric_labels)
        feature_labels.update(people_labels)

        # 1) Numeric sanity-check: inputs vs medians
        try:
            deltas = H.explain_numeric_deltas(
                values=last_values,
                used_flags=last_used,
                medians=medians,
                labels=numeric_labels,
                top_n=6,
                int_features=numeric_int_features,
            )
        except Exception:
            deltas = []

        if deltas:
            st.subheader("Écarts des chiffres vs valeurs typiques")
            st.caption(
                "L'écart (Δ) compare ta valeur à une valeur ‘typique’ du dataset. "
                "Δ > 0 signifie ‘au-dessus du typique’, Δ < 0 ‘en dessous’."
            )
            show = H.numeric_deltas_table_with_mean(deltas, stats_df=stats_df, int_features=numeric_int_features)
            st.dataframe(show, width='stretch', hide_index=True)

        # 2) Local effects: one-feature neutralization
        st.subheader("Effets locaux estimés sur la note")
        st.caption(
            "Interprétation : Δ > 0 ⇒ en gardant tout le reste identique, ce champ **augmente** la note prédite (par rapport à sa valeur neutre). "
            "Δ < 0 ⇒ il la **diminue**.\n\n"
            "Important : ce sont des **effets locaux** (pour ta saisie) et ils peuvent changer si tu modifies d'autres champs."
        )
        effects_df = H.local_feature_effects(
            model,
            features=features,
            values=last_values,
            used_flags=last_used,
            medians=medians,
            identity_cols=identity_cols,
            clip=(0.0, 5.0),
            max_rows=16,
        )

        if effects_df is None or effects_df.empty:
            st.info("Aucun effet local à afficher (aucun champ activé ou calcul impossible).")
        else:
            pretty = effects_df.copy()
            pretty["Variable"] = pretty["feature"].map(lambda x: feature_labels.get(str(x), str(x)))
            pretty = pretty[["Variable", "delta", "neutral_pred", "base_pred"]].rename(
                columns={
                    "delta": "Δ (note)",
                    "neutral_pred": "Sans cette info",
                    "base_pred": "Prédiction",
                }
            )
            st.dataframe(pretty, width='stretch', hide_index=True)

            chart = H.altair_local_effects_bar(effects_df, feature_labels=feature_labels, width=560, height=360)
            if chart is not None:
                st.altair_chart(chart, use_container_width=True)

    used_cols = [c for c in features if last_used.get(c)]
    if used_cols:
        st.caption("Champs utilisés: " + ", ".join(used_cols))
    else:
        st.caption("Aucun champ activé: valeurs neutres (médianes) utilisées.")

    # Explications statistiques (dataset): corrélations et graphiques
    with st.expander("Analyses statistiques (corrélations + graphiques)", expanded=False):
        st.caption(
            "Ces analyses viennent du dataset de référence (Letterboxd).\n\n"
            "Elles montrent des **tendances** : quand une variable monte/descend, la note a tendance à monter/descendre. "
            "Ça ne prouve **pas** qu'une variable ‘cause’ la note (corrélation ≠ causalité)."
        )

        if ref_df.empty or ("rating" not in ref_df.columns):
            st.info("Impossible d'afficher les analyses: la colonne `rating` est manquante dans les données de référence.")
        else:

            ref_for_stats = ref_df

            numeric_for_stats = [f for f in numeric_labels.keys() if f in ref_for_stats.columns]

            robust_clip = st.checkbox(
                "Limiter les valeurs extrêmes (1%–99%)",
                value=True,
                help="Évite que quelques films très atypiques écrasent l'échelle des graphiques. Recommandé.",
            )
            clip_q = (0.01, 0.99) if robust_clip else None

            corr_df = H.numeric_correlations(ref_for_stats, features=numeric_for_stats, target="rating", min_n=30, clip_quantiles=clip_q)

            if corr_df.empty:
                st.warning("Pas assez de données numériques valides pour calculer des corrélations (N < 30).")
            else:
                st.caption(
                    "Lecture rapide :\n"
                    "- valeur proche de 0 ⇒ lien faible\n"
                    "- valeur positive ⇒ quand la variable augmente, la note a tendance à augmenter\n"
                    "- valeur négative ⇒ quand la variable augmente, la note a tendance à baisser\n\n"
                    "Pearson capte surtout les liens ‘linéaires’. Spearman capte des liens monotones (plus robuste aux formes non linéaires)."
                )
                display = corr_df.copy()
                display["Variable"] = display["feature"].map(lambda x: numeric_labels.get(str(x), str(x)))
                display = display[["Variable", "pearson", "spearman", "n"]]
                display = display.rename(columns={"pearson": "Pearson", "spearman": "Spearman", "n": "N"})
                st.subheader("Corrélations avec la note (rating)")
                st.dataframe(display, width='stretch', hide_index=True)

                # Choose a feature to visualize
                corr_features = [str(x) for x in corr_df["feature"].tolist()]
                used_numeric = [f for f in numeric_labels.keys() if last_used.get(f)]
                default_feat = None
                if used_numeric:
                    tmp = corr_df[corr_df["feature"].isin(used_numeric)]
                    if not tmp.empty:
                        default_feat = str(tmp.iloc[0]["feature"])
                if default_feat is None:
                    default_feat = str(corr_df.iloc[0]["feature"])

                feat_choice = st.selectbox(
                    "Voir le lien entre une variable et la note",
                    options=corr_features,
                    index=(corr_features.index(default_feat) if default_feat in corr_features else 0),
                    format_func=lambda x: numeric_labels.get(str(x), str(x)),
                )

                transform_x = None
                if str(feat_choice) in {"budget", "revenue"}:
                    # Affichage uniquement : l'échelle log rend ces graphiques lisibles (budget/revenu très asymétriques)
                    transform_x = "log1p"
                    st.caption(
                        "Affichage : axe en échelle log (log1p) pour budget/revenu (lisibilité uniquement, ne change pas la prédiction)."
                    )

                user_x = None
                if last_used.get(feat_choice):
                    try:
                        raw_x = last_values.get(feat_choice)
                        user_x = float(raw_x) if raw_x is not None else None
                    except Exception:
                        user_x = None

                st.subheader("Nuage de points + tendance")
                chart = H.altair_scatter_with_regression(
                    ref_for_stats,
                    feature=str(feat_choice),
                    target="rating",
                    user_x=user_x,
                    user_y=y_pred,
                    clip_quantiles=clip_q,
                    transform_x=transform_x,
                    title_col="title",
                )
                if chart is None:
                    st.info("Graphique indisponible (Altair non disponible ou données insuffisantes).")
                else:
                    st.altair_chart(chart)
                    if user_x is not None:
                        st.caption("La règle verticale marque ta valeur. Le point plein correspond à ta note prédite.")
                    else:
                        st.caption("Active la variable dans les champs pour afficher ta valeur sur le graphique.")

                st.subheader("Distribution de la variable")
                hist = H.altair_histogram_with_rule(
                    ref_for_stats,
                    feature=str(feat_choice),
                    user_x=user_x,
                    clip_quantiles=clip_q,
                    transform_x=transform_x,
                )
                if hist is None:
                    st.info("Histogramme indisponible (Altair non disponible ou données insuffisantes).")
                else:
                    st.altair_chart(hist)

            # Actor/director diagnostic (bootstrap + raw points)
            if identity_cols and ("rating" in ref_for_stats.columns):
                def _selected_names(key: str) -> list[str]:
                    raw = last_values.get(key)
                    if not isinstance(raw, list):
                        return []
                    out: list[str] = []
                    seen: set[str] = set()
                    for x in raw:
                        s = str(x).strip()
                        if not s or s == "__MISSING__":
                            continue
                        if s in seen:
                            continue
                        seen.add(s)
                        out.append(s)
                    return out

                director_candidates = _selected_names("directors")
                actor_candidates = _selected_names("casting")

                chosen_director: str | None = None
                chosen_actor: str | None = None

                if director_candidates and ("directors" in ref_for_stats.columns):
                    if len(director_candidates) == 1:
                        chosen_director = director_candidates[0]
                    else:
                        chosen_director = st.selectbox(
                            "Choisir un réalisateur à analyser",
                            options=director_candidates,
                            index=0,
                            key=f"diag_director_{WIDGET_STATE_SUFFIX}",
                        )

                if actor_candidates and ("casting" in ref_for_stats.columns):
                    if len(actor_candidates) == 1:
                        chosen_actor = actor_candidates[0]
                    else:
                        chosen_actor = st.selectbox(
                            "Choisir un acteur à analyser",
                            options=actor_candidates,
                            index=0,
                            key=f"diag_actor_{WIDGET_STATE_SUFFIX}",
                        )

                if chosen_director or chosen_actor:
                    st.subheader("Diagnostic acteur / réalisateur (dataset)")
                    st.caption(
                        "On compare la note moyenne des films **avec** vs **sans** la personne dans le dataset.\n\n"
                        "À lire comme un **signal descriptif** (pas causal) : un réalisateur/acteur peut être associé à certains genres, budgets, périodes, etc.\n\n"
                        "Plus il y a de films ‘avec’ (N), plus l'indication est crédible. Si N est faible, Δ et l'intervalle peuvent être instables."
                    )

                if chosen_director and ("directors" in ref_for_stats.columns):
                    st.write(f"Réalisateur: **{chosen_director}**")
                    chart_d, stats_d = H.altair_token_effect_chart(
                        ref_for_stats,
                        token_col="directors",
                        token_name=str(chosen_director),
                        target="rating",
                        title_col="title",
                        width=700,
                        height=320,
                        n_boot=1000,
                    )
                    if chart_d is not None:
                        st.altair_chart(chart_d)
                    if isinstance(stats_d, dict) and stats_d:
                        n_yes = int(stats_d.get("n_yes", 0) or 0)
                        n_no = int(stats_d.get("n_no", 0) or 0)
                        mean_yes = stats_d.get("mean_yes")
                        mean_no = stats_d.get("mean_no")
                        diff = stats_d.get("diff")
                        d = stats_d.get("cohen_d")
                        ci = stats_d.get("ci_yes")
                        st.write(f"Films avec: **{n_yes}**, sans: **{n_no}**")
                        if n_yes > 0 and mean_yes is not None:
                            if isinstance(ci, (list, tuple)) and len(ci) >= 3 and np.isfinite(ci[1]) and np.isfinite(ci[2]):
                                st.write(f"Moyenne (avec): **{float(mean_yes):.2f}** — IC bootstrap 95%: **{float(ci[1]):.2f} – {float(ci[2]):.2f}**")
                            else:
                                st.write(f"Moyenne (avec): **{float(mean_yes):.2f}**")
                        if n_no > 0 and mean_no is not None:
                            st.write(f"Moyenne (sans): **{float(mean_no):.2f}**")
                        if diff is not None and np.isfinite(float(diff)):
                            extra = f"Δ (avec − sans): **{float(diff):+.2f}**"
                            if d is not None and np.isfinite(float(d)):
                                extra += f" | d (Cohen): **{float(d):+.2f}**"
                            st.caption(extra)
                            st.caption("Repère (très approximatif) pour d (Cohen) : ~0,2 = petit, ~0,5 = moyen, ~0,8 = grand.")

                if chosen_actor and ("casting" in ref_for_stats.columns):
                    st.write(f"Acteur: **{chosen_actor}**")
                    chart_a, stats_a = H.altair_token_effect_chart(
                        ref_for_stats,
                        token_col="casting",
                        token_name=str(chosen_actor),
                        target="rating",
                        title_col="title",
                        width=700,
                        height=320,
                        n_boot=1000,
                    )
                    if chart_a is not None:
                        st.altair_chart(chart_a)
                    if isinstance(stats_a, dict) and stats_a:
                        n_yes = int(stats_a.get("n_yes", 0) or 0)
                        n_no = int(stats_a.get("n_no", 0) or 0)
                        mean_yes = stats_a.get("mean_yes")
                        mean_no = stats_a.get("mean_no")
                        diff = stats_a.get("diff")
                        d = stats_a.get("cohen_d")
                        ci = stats_a.get("ci_yes")
                        st.write(f"Films avec: **{n_yes}**, sans: **{n_no}**")
                        if n_yes > 0 and mean_yes is not None:
                            if isinstance(ci, (list, tuple)) and len(ci) >= 3 and np.isfinite(ci[1]) and np.isfinite(ci[2]):
                                st.write(f"Moyenne (avec): **{float(mean_yes):.2f}** — IC bootstrap 95%: **{float(ci[1]):.2f} – {float(ci[2]):.2f}**")
                            else:
                                st.write(f"Moyenne (avec): **{float(mean_yes):.2f}**")
                        if n_no > 0 and mean_no is not None:
                            st.write(f"Moyenne (sans): **{float(mean_no):.2f}**")
                        if diff is not None and np.isfinite(float(diff)):
                            extra = f"Δ (avec − sans): **{float(diff):+.2f}**"
                            if d is not None and np.isfinite(float(d)):
                                extra += f" | d (Cohen): **{float(d):+.2f}**"
                            st.caption(extra)
                            st.caption("Repère (très approximatif) pour d (Cohen) : ~0,2 = petit, ~0,5 = moyen, ~0,8 = grand.")

            # Categorical quick stats (only if identity columns are used)
            if identity_cols and ("genres" in ref_df.columns) and ("themes" in ref_df.columns) and ("rating" in ref_df.columns):
                st.subheader("Comparaisons par genre / thème (moyennes)")
                global_mean = float(pd.to_numeric(ref_df["rating"], errors="coerce").mean())

                def _mean_by_token(col: str, selected: list[str]) -> pd.DataFrame:
                    tmp = ref_for_stats[[col, "rating"]].dropna()
                    tmp = tmp[tmp[col].apply(lambda x: isinstance(x, list))]
                    if tmp.empty:
                        return pd.DataFrame()
                    exploded = tmp.explode(col)
                    exploded[col] = exploded[col].astype(str)
                    exploded["rating"] = pd.to_numeric(exploded["rating"], errors="coerce")
                    exploded = exploded.dropna(subset=["rating"])
                    if selected:
                        exploded = exploded[exploded[col].isin([str(x) for x in selected])]
                    # compute per-token stats
                    grp = (
                        exploded.groupby(col, as_index=False)
                        .agg(n=("rating", "count"), mean=("rating", "mean"), sd=("rating", "std"))
                    )
                    overall_sd = float(pd.to_numeric(ref_for_stats["rating"], errors="coerce").std())
                    global_m = float(pd.to_numeric(ref_for_stats["rating"], errors="coerce").mean())
                    # Cohen's d-like heuristic against global mean (use overall sd)
                    def _relation(row):
                        n = int(row.get("n") or 0)
                        m = float(row.get("mean") or 0.0)
                        sd = float(row.get("sd") or 0.0)
                        diff = m - global_m
                        d = 0.0
                        if overall_sd and overall_sd > 0:
                            d = diff / overall_sd
                        related = "Non"
                        if n >= 20 and abs(d) >= 0.2:
                            related = "Oui"
                        elif n < 20:
                            related = "Insuffisant (n<20)"
                        return pd.Series({"N": n, "Mean": m, "SD": sd, "Diff": diff, "Cohen_d": d, "Related": related})

                    stats = grp.apply(_relation, axis=1)
                    stats.index = grp[col].tolist()
                    stats = stats.reset_index().rename(columns={"index": col})
                    stats = stats.sort_values("Mean", ascending=False)
                    return stats

                selected_genres = last_values.get("genres") if last_used.get("genres") else []
                selected_themes = last_values.get("themes") if last_used.get("themes") else []
                sel_g = [x for x in (selected_genres if isinstance(selected_genres, list) else []) if x != "__MISSING__"]
                sel_t = [x for x in (selected_themes if isinstance(selected_themes, list) else []) if x != "__MISSING__"]

                if sel_g:
                    gdf = _mean_by_token("genres", sel_g)
                    if not gdf.empty:
                        st.caption(f"Note moyenne globale: {global_mean:.2f} / 5")
                        st.write("Genres sélectionnés (moyenne dans le dataset):")
                        display_g = gdf.rename(columns={"genres": "Genre", "Mean": "Note moyenne", "Cohen_d": "Cohen d", "Related": "Relation"})
                        st.dataframe(display_g, width='stretch', hide_index=True)
                if sel_t:
                    tdf = _mean_by_token("themes", sel_t)
                    if not tdf.empty:
                        st.caption(f"Note moyenne globale: {global_mean:.2f} / 5")
                        st.write("Thèmes sélectionnés (moyenne dans le dataset):")
                        display_t = tdf.rename(columns={"themes": "Thème", "Mean": "Note moyenne", "Cohen_d": "Cohen d", "Related": "Relation"})
                        st.dataframe(display_t, width='stretch', hide_index=True)

            # (Diagnostic acteur/réalisateur disponible ici; plus de détails dans Exploration)
