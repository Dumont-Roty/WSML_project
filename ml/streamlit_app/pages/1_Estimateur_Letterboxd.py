from __future__ import annotations

import json
import re
import sys
import urllib.error
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

    def _with_comma(val: float, fmt: str) -> str:
        s = fmt.format(val)
        return s.replace(".", ",")

    if v >= 1_000_000_000:
        return f"{_with_comma(v/1_000_000_000, '{:.2f}')} Md"
    if v >= 1_000_000:
        return f"{_with_comma(v/1_000_000, '{:.2f}')} M"
    if v >= 1_000:
        return f"{_with_comma(v/1_000, '{:.1f}')} k"
    return f"{int(round(v))}"


@st.cache_data
def _load_reference_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path)


@st.cache_data(ttl=24 * 60 * 60)
def _fetch_html(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read()
    return raw.decode("utf-8", errors="ignore")


def _extract_letterboxd_poster_url(html: str) -> str | None:
    # Letterboxd meta tags can sometimes point to the page header/backdrop.
    # We want the actual poster (usually a.ltrbxd.com/resized/film-poster/...).

    def _img_tag_to_url(tag: str) -> str | None:
        # Prefer the largest in srcset (usually last)
        m_srcset = re.search(r"\bsrcset=\"([^\"]+)\"", tag, flags=re.IGNORECASE)
        if m_srcset:
            parts = [p.strip() for p in m_srcset.group(1).split(",") if p.strip()]
            if parts:
                last = parts[-1]
                last_url = last.split(" ")[0].strip()
                if last_url.startswith("http"):
                    return last_url

        m_src = re.search(r"\bsrc=\"([^\"]+)\"", tag, flags=re.IGNORECASE)
        if m_src:
            u = m_src.group(1).strip()
            if u.startswith("http"):
                return u
        return None

    def _pick_best(urls: list[str]) -> str | None:
        urls = [u.strip() for u in urls if isinstance(u, str) and u.strip()]
        urls = [u for u in urls if u.startswith("http")]
        if not urls:
            return None

        # Prefer real poster crops
        posters = [u for u in urls if "film-poster" in u]
        if posters:
            # Prefer the standard crop if present
            for u in posters:
                if "-230-" in u and "-345-" in u:
                    return u
            return posters[0]

        # Never return backdrops
        urls = [u for u in urls if "/upload/" not in u and "backdrop" not in u]
        return urls[0] if urls else None

    # 0) Best: extract poster from the actual poster column (never the backdrop)
    m_poster_col = re.search(r"\bid=\"js-poster-col\"", html, flags=re.IGNORECASE)
    if m_poster_col:
        start = m_poster_col.start()
        window = html[start : start + 25000]
        # Prefer <img alt="Poster for ..."> if present in that window
        for img in re.finditer(r"<img\b[^>]*>", window, flags=re.IGNORECASE | re.DOTALL):
            tag = img.group(0)
            if re.search(r"\balt=\"Poster\s+for\b", tag, flags=re.IGNORECASE):
                u = _img_tag_to_url(tag)
                if u:
                    return u
        # Fallback: any image inside poster col
        for img in re.finditer(r"<img\b[^>]*>", window, flags=re.IGNORECASE | re.DOTALL):
            u = _img_tag_to_url(img.group(0))
            if u:
                return u

    # 1) Strongest remaining: direct film-poster URLs anywhere in the HTML
    film_posters = re.findall(
        r"https?://a\.ltrbxd\.com/resized/film-poster/[^\"'\s<>]+",
        html,
        flags=re.IGNORECASE,
    )
    best = _pick_best(film_posters)
    if best:
        return best

    candidates: list[str] = []

    # 2) Next: scan <img> tags (poster is often <img class="image" ...>)
    for img in re.finditer(
        r"<img\b[^>]*\bclass=\"[^\"]*\bimage\b[^\"]*\"[^>]*>",
        html,
        flags=re.IGNORECASE,
    ):
        tag = img.group(0)
        m_srcset = re.search(r"\bsrcset=\"([^\"]+)\"", tag, flags=re.IGNORECASE)
        if m_srcset:
            # Prefer the largest in srcset (usually last)
            parts = [p.strip() for p in m_srcset.group(1).split(",") if p.strip()]
            if parts:
                last = parts[-1]
                last_url = last.split(" ")[0].strip()
                if last_url.startswith("http"):
                    candidates.append(last_url)

        m_src = re.search(r"\bsrc=\"([^\"]+)\"", tag, flags=re.IGNORECASE)
        if m_src:
            u = m_src.group(1).strip()
            if u.startswith("http"):
                candidates.append(u)

    best = _pick_best(candidates)
    if best:
        return best

    # 3) Meta tags as a last resort (only if they point to film-poster)
    meta_candidates: list[str] = []
    for pat in (
        r"property=\"og:image\"\s+content=\"([^\"]+)\"",
        r"property=\"og:image:secure_url\"\s+content=\"([^\"]+)\"",
        r"name=\"twitter:image\"\s+content=\"([^\"]+)\"",
    ):
        m = re.search(pat, html, flags=re.IGNORECASE)
        if m:
            u = m.group(1).strip()
            if u.startswith("http"):
                meta_candidates.append(u)

    best = _pick_best(meta_candidates)
    if best and "film-poster" in best:
        return best

    return None


@st.cache_data(ttl=24 * 60 * 60)
def _get_letterboxd_poster_url(letterboxd_url: str, _cache_bust: str = "v3") -> str | None:
    try:
        html = _fetch_html(letterboxd_url)
        return _extract_letterboxd_poster_url(html)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError):
        return None


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

numeric_cols, identity_cols, features = _load_schema(METRICS_PATH)
if not features:
    st.error(
        "Impossible de trouver la liste des features. "
        "Lance d'abord l'entraînement pour générer ml/models/metrics.json."
    )
    st.stop()

if not MODEL_PATH.exists():
    st.error("Modèle introuvable. Lance d'abord l'entraînement pour générer ml/models/best_model.joblib.")
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


st.sidebar.caption(
    "Note: l'estimation du budget est fournie à titre indicatif et sert d'aide pour affiner l'estimation de la note finale (elle n'est pas une vérité)."
)

ref_row: dict[str, Any] | None = None
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

        if lb_url and lb_url.startswith("http"):
            poster_url = _get_letterboxd_poster_url(lb_url)
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
    preset_values: dict | None = None,
) -> tuple[bool, float]:
    pre_has = bool(preset_values and feature in preset_values)
    use = st.checkbox(f"Utiliser {label}", value=pre_has, key=f"use_{feature}")
    _suggestion_caption(label, suggested)
    if not use:
        return False, float(medians.get(feature, 0.0))

    series = stats_df[feature] if feature in stats_df.columns else pd.Series(dtype=float)
    lo, hi = _q_range(series, fallback_min=fallback_min, fallback_max=fallback_max)
    default = float(medians.get(feature, 0.0))
    if preset_values and feature in preset_values:
        try:
            default = float(preset_values.get(feature))
        except Exception:
            default = default

    if is_int:
        val = st.slider(label, int(lo), int(hi), int(round(default)), key=f"val_{feature}")
        return True, float(val)

    val = st.slider(label, float(lo), float(hi), float(default), key=f"val_{feature}", format="%.2f")
    if feature in ("budget", "revenue"):
        try:
            st.caption(f"Valeur sélectionnée : {_format_money(float(val))} (affiché en millions)")
        except Exception:
            pass
    return True, float(val)


def _optional_multiselect_count(
    *,
    feature: str,
    label: str,
    ref_col: str,
    preset_values: dict | None = None,
) -> tuple[bool, float]:
    pre_has = bool(preset_values and feature in preset_values)
    use = st.checkbox(f"Utiliser {label}", value=pre_has, key=f"use_{feature}")
    suggested_list: list[str] | None = None
    if ref_row is not None:
        ref_val = ref_row.get(ref_col)
        if isinstance(ref_val, list):
            suggested_list = [str(x) for x in ref_val]
    _suggestion_caption(label, suggested_list)

    if not use:
        return False, float(medians.get(feature, 0.0))

    options = name_options.get(ref_col, [])
    default_sel: list[str] = []
    if preset_values and feature in preset_values and isinstance(preset_values.get(feature), list):
        default_sel = [str(x) for x in preset_values.get(feature)]
    selected = st.multiselect(label, options=options, default=default_sel, key=f"val_{feature}")
    return True, float(len(selected))


def _optional_multiselect_list(
    *,
    feature: str,
    label: str,
    ref_col: str,
    preset_values: dict | None = None,
) -> tuple[bool, list[str]]:
    pre_has = bool(preset_values and feature in preset_values)
    use = st.checkbox(f"Utiliser {label}", value=pre_has, key=f"use_{feature}")
    suggested_list: list[str] | None = None
    if ref_row is not None:
        ref_val = ref_row.get(ref_col)
        if isinstance(ref_val, list):
            suggested_list = [str(x) for x in ref_val]
    _suggestion_caption(label, suggested_list)

    if not use:
        return False, ["__MISSING__"]

    options = name_options.get(ref_col, [])
    default_sel: list[str] = []
    if preset_values and feature in preset_values and isinstance(preset_values.get(feature), list):
        default_sel = [str(x) for x in preset_values.get(feature)]
    selected = st.multiselect(label, options=options, default=default_sel, key=f"val_{feature}")
    return True, [str(x) for x in selected]


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
        u, v = _optional_slider(
            feature=feature,
            label=label,
            is_int=is_int,
            fallback_min=fmin,
            fallback_max=fmax,
            suggested=suggested,
            preset_values=preset_values,
        )
    used[feature] = u
    values[feature] = v
    col_idx += 1

st.subheader("Personnes / catégories (menus déroulants, optionnels)")
people_container = st.container()
cols_p = people_container.columns(2)
colp_idx = 0

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
        with cols_p[colp_idx % 2]:
            u, v = _optional_multiselect_list(
                feature=feature,
                label=label,
                ref_col=ref_col,
                preset_values=preset_values,
            )
        used[feature] = u
        values[feature] = v
        colp_idx += 1
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
        with cols_p[colp_idx % 2]:
            u, v = _optional_multiselect_count(
                feature=feature,
                label=label,
                ref_col=ref_col,
                preset_values=preset_values,
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
