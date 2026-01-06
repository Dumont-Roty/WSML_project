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
ML_DIR = HERE.parents[1]  # .../ml
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


def _sanitize_widget_suffix(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "manual"
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", s)
    return s[:80] if len(s) > 80 else s


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


def _tmdb_api_key() -> str | None:
    try:
        key = st.secrets.get("TMDB_API_KEY")  # type: ignore[attr-defined]
        if isinstance(key, str) and key.strip():
            return key.strip()
    except Exception:
        pass
    key = os.environ.get("TMDB_API_KEY")
    return key.strip() if isinstance(key, str) and key.strip() else None


def _extract_tmdb_movie_id(letterboxd_html: str) -> int | None:
    for pat in (
        r"data-tmdb-id\s*=\s*['\"](\d+)['\"]",
        r"themoviedb\.org/movie/(\d+)",
        r"tmdb\.org/movie/(\d+)",
    ):
        m = re.search(pat, letterboxd_html, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None


@st.cache_data(ttl=24 * 60 * 60)
def _tmdb_get_json(path: str, *, language: str = "fr-FR", params: dict[str, object] | None = None) -> dict[str, Any] | None:
    api_key = _tmdb_api_key()
    if not api_key:
        return None

    base = "https://api.themoviedb.org/3"
    url = urllib.parse.urljoin(base + "/", path.lstrip("/"))
    query: dict[str, str] = {"api_key": api_key, "language": language}
    if isinstance(params, dict):
        for k, v in params.items():
            if v is None:
                continue
            query[str(k)] = str(v)
    qs = urllib.parse.urlencode(query)
    url = f"{url}?{qs}"

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json,*/*;q=0.8",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
        return json.loads(raw.decode("utf-8", errors="ignore"))
    except Exception:
        return None


@st.cache_data(ttl=7 * 24 * 60 * 60)
def _tmdb_image_base_url() -> str:
    payload = _tmdb_get_json("/configuration", language="en-US")
    if isinstance(payload, dict):
        images = payload.get("images")
        if isinstance(images, dict):
            base = images.get("secure_base_url") or images.get("base_url")
            if isinstance(base, str) and base.startswith("http"):
                return base
    return "https://image.tmdb.org/t/p/"


def _tmdb_profile_url(profile_path: str, *, size: str = "w185") -> str:
    base = _tmdb_image_base_url()
    return urllib.parse.urljoin(base, f"{size}{profile_path}")


@st.cache_data(ttl=7 * 24 * 60 * 60)
def _tmdb_person_profile_url_by_name(name: str) -> str | None:
    q = (name or "").strip()
    if not q:
        return None
    payload = _tmdb_get_json(
        "/search/person",
        language="en-US",
        params={"query": q, "page": 1, "include_adult": "false"},
    )
    if not isinstance(payload, dict):
        return None
    results = payload.get("results")
    if not isinstance(results, list):
        return None
    for row in results:
        if not isinstance(row, dict):
            continue
        profile_path = row.get("profile_path")
        if isinstance(profile_path, str) and profile_path.strip():
            return _tmdb_profile_url(profile_path.strip(), size="w185")
    return None


def _render_selected_people_thumbnails(
    names: list[str],
    *,
    title: str = "Sélection",
    known_photos: dict[str, str] | None = None,
    max_items: int = 32,
    thumb_size_px: int = 56,
) -> None:
    if not names:
        return
    st.caption(f"{title} (aperçu)")
    cols = st.columns(min(8, len(names)))
    for i, name in enumerate(names[: int(max_items)]):
        with cols[i % len(cols)]:
            url = None
            if isinstance(known_photos, dict):
                url = known_photos.get(name)
            if not url:
                url = _tmdb_person_profile_url_by_name(name)
            if url:
                try:
                    st.image(url, width=int(thumb_size_px))
                except Exception:
                    pass


@st.cache_data(ttl=24 * 60 * 60)
def _tmdb_movie_cast(tmdb_movie_id: int, *, top_n: int = 9) -> list[dict[str, Any]]:
    payload = _tmdb_get_json(f"/movie/{tmdb_movie_id}/credits")
    if not isinstance(payload, dict):
        return []
    cast = payload.get("cast")
    if not isinstance(cast, list):
        return []

    out: list[dict[str, Any]] = []
    for mem in cast[: int(top_n)]:
        if not isinstance(mem, dict):
            continue
        profile = mem.get("profile_path")
        if isinstance(profile, str) and profile.strip():
            profile_url = _tmdb_profile_url(profile.strip(), size="w185")
        else:
            profile_url = None
        out.append({"name": mem.get("name"), "character": mem.get("character"), "profile_url": profile_url})
    return out


def _extract_letterboxd_poster_url(html: str) -> str | None:
    # Implement a pragmatic extraction: look for film-poster occurrences and select the first valid
    if not isinstance(html, str):
        return None

    # quick search for film-poster fragment
    m = re.search(r'(https?://[^"\']+?/film-poster/[^"\']+)', html)
    if m:
        return m.group(1)

    # fallback: og:image (handle single or double quotes)
    m2 = re.search(r'property=["\']og:image["\']\s+content=["\']([^"\']+)["\']', html)
    if m2:
        return m2.group(1)
    return None


@st.cache_data(ttl=24 * 60 * 60)
def _get_letterboxd_poster_url(letterboxd_url: str, _cache_bust: str = "v6") -> str | None:
    try:
        html = _fetch_html(letterboxd_url)
    except Exception:
        return None
    url = _extract_letterboxd_poster_url(html)
    return url


@st.cache_data
def _load_train_features(path: Path, features: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    requested = [str(c) for c in features if c is not None and str(c).strip()]
    requested = list(dict.fromkeys(requested))
    try:
        df = pd.read_parquet(path, columns=requested)
        return df.copy()
    except Exception:
        df = pd.read_parquet(path)
        cols = [c for c in requested if c in df.columns]
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

    def _norm_name(s: str) -> str:
        s = (s or "").replace("\u00a0", " ").strip()
        s = re.sub(r"\s+", " ", s)
        return s

    counts: dict[str, int] = {}
    for name in flat:
        n = _norm_name(str(name))
        if not n:
            continue
        counts[n] = counts.get(n, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_k <= 0:
        return [name for name, _ in ranked]
    return [name for name, _ in ranked[:top_k]]


# Small visual helpers used by pages

def _bootstrap_repo_path() -> Path:
    here = Path(__file__).resolve()
    repo_root = ML_DIR.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def _apply_letterboxd_theme() -> None:
    letterboxd_colors = {
        "bg": "#0b0b0b",
        "text": "#E6E6E6",
        "accent": "#2AB44B",
        "accent_dark": "#1B5E20",
    }

    st.markdown(
        f"""
        <style>
        :root {{
          --lb-bg: {letterboxd_colors['bg']};
          --lb-text: {letterboxd_colors['text']};
          --lb-accent: {letterboxd_colors['accent']};
          --lb-accent-dark: {letterboxd_colors['accent_dark']};
        }}
        .stApp, .block-container {{
          background-color: var(--lb-bg) !important;
          color: var(--lb-text) !important;
        }}
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
          color: var(--lb-accent) !important;
        }}
                /* Hide slider tick marks and labels to reduce visual clutter */
                .stSlider .rc-slider-mark, .stSlider .rc-slider-mark-text, .stSlider label {{
                    display: none !important;
                }}
                /* Compact the slider container so captions align neatly */
                .stSlider {{
                    margin-bottom: 0.3rem !important;
                }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# Small UI helpers moved from pages to keep presentation code minimal.
def _suggestion_caption(label: str, suggested: object | None) -> None:
    if suggested is None:
        return
    if isinstance(suggested, list):
        if suggested:
            short = ", ".join(map(str, suggested[:12]))
            st.caption(f"Suggestion (référence) — {label}: {short}{'…' if len(suggested) > 12 else ''}")
        return
    # If numeric, format large numbers as money for readability
    try:
        if isinstance(suggested, (int, float, np.number)):
            s = _format_money(float(suggested))
            st.caption(f"Suggestion (référence) — {label}: {s}")
            return
    except Exception:
        pass
    st.caption(f"Suggestion (référence) — {label}: {suggested}")


def _optional_slider(
    feature: str,
    label: str,
    is_int: bool,
    fallback_min: float,
    fallback_max: float,
    suggested: float | None,
    preset_values: dict | None,
    medians: dict,
    stats_df: pd.DataFrame,
    prefill_from_ref: bool,
    ref_row: dict | None,
    state_suffix: str,
) -> tuple[bool, float]:
    use_key = f"use_{feature}_{state_suffix}"
    val_key = f"val_{feature}_{state_suffix}"

    pre_has = bool(preset_values and feature in preset_values)

    prefill_value = None
    prefill_used = False
    if prefill_from_ref and isinstance(ref_row, dict):
        rr = ref_row
        if isinstance(rr, dict) and feature in rr and rr.get(feature) is not None:
            prefill_value = rr.get(feature)
            prefill_used = True

    use = st.checkbox(f"Utiliser {label}", value=(prefill_used or pre_has), key=use_key)
    # Show suggestion only when it's not simply the same as the prefill-from-ref
    try:
        def _suggestion_matches_prefill(sugg, prefill):
            if prefill is None:
                return False
            if isinstance(sugg, list) and isinstance(prefill, list):
                return set(sugg) <= set(prefill)
            try:
                # numeric comparison
                if (isinstance(sugg, (int, float, np.number)) or isinstance(prefill, (int, float, np.number))):
                    return float(sugg) == float(prefill)
            except Exception:
                pass
            # fallback to string equality
            return str(sugg) == str(prefill)

        if not (prefill_from_ref and prefill_used and _suggestion_matches_prefill(suggested, prefill_value)):
            _suggestion_caption(label, suggested)
    except Exception:
        _suggestion_caption(label, suggested)
    if not use:
        return False, float(medians.get(feature, 0.0))

    series = stats_df[feature] if feature in stats_df.columns else pd.Series(dtype=float)
    lo, hi = _q_range(series, fallback_min=fallback_min, fallback_max=fallback_max)
    default = float(medians.get(feature, 0.0))
    if prefill_value is not None:
        try:
            default = float(prefill_value)
        except Exception:
            default = default
    if preset_values and feature in preset_values:
        try:
            pv = preset_values.get(feature)
            if pv is not None:
                default = float(pv)
        except Exception:
            default = default

    if np.isfinite(default):
        default = float(min(max(default, float(lo)), float(hi)))

    if is_int:
        if feature in ("budget", "revenue"):
            val = st.number_input(label, min_value=int(lo), max_value=int(hi), value=int(round(default)), key=val_key, step=1)
            try:
                st.caption(f"Valeur sélectionnée : {_format_money(float(val))} (affiché en millions)")
            except Exception:
                pass
            return True, float(val)

        val = st.slider(label, int(lo), int(hi), int(round(default)), key=val_key)
        try:
            st.caption(f"Valeur sélectionnée : {int(round(val))}")
        except Exception:
            pass
        return True, float(val)

    if feature in ("budget", "revenue"):
        val = st.number_input(label, min_value=int(float(lo)), max_value=int(float(hi)), value=int(round(float(default))), key=val_key, step=1)
        try:
            st.caption(f"Valeur sélectionnée : {_format_money(float(val))} (affiché en millions)")
        except Exception:
            pass
        return True, float(val)

    val = st.slider(label, float(lo), float(hi), float(default), key=val_key, format="%.2f")
    try:
        st.caption(f"Valeur sélectionnée : {float(val):.2f}")
    except Exception:
        pass
    return True, float(val)


def _optional_multiselect_count(
    feature: str,
    label: str,
    ref_col: str,
    preset_values: dict | None,
    medians: dict,
    ref_row: dict | None,
    prefill_from_ref: bool,
    state_suffix: str,
    name_options: dict,
    ref_tmdb_id: int | None,
) -> tuple[bool, float]:
    use_key = f"use_{feature}_{state_suffix}"
    val_key = f"val_{feature}_{state_suffix}"

    prefill_list: list[str] = []
    prefill_used = False
    if prefill_from_ref and isinstance(ref_row, dict):
        rr = ref_row
        if isinstance(rr, dict):
            val = rr.get(ref_col)
            if ref_col == "directors" and val is None:
                val = rr.get("director")
            if isinstance(val, list):
                prefill_list = [str(x) for x in val if str(x).strip()]
            elif isinstance(val, str) and val.strip():
                prefill_list = [val.strip()]
            if ref_col in {"directors", "casting", "producers", "writers", "composer"}:
                prefill_list = prefill_list[:5]
            prefill_used = bool(prefill_list)

    pre_has = bool(preset_values and feature in preset_values)
    use = st.checkbox(f"Utiliser {label}", value=(prefill_used or pre_has), key=use_key)
    suggested_list: list[str] | None = None
    if ref_row is not None:
        ref_val = ref_row.get(ref_col)
        if isinstance(ref_val, list):
            suggested_list = [str(x) for x in ref_val]
    # If prefill_from_ref is active and the suggested list corresponds to the prefill list, don't show caption
    try:
        if not (prefill_from_ref and prefill_used and suggested_list is not None and set(suggested_list) <= set(prefill_list)):
            _suggestion_caption(label, suggested_list)
    except Exception:
        _suggestion_caption(label, suggested_list)

    if not use:
        return False, float(medians.get(feature, 0.0))

    options = name_options.get(ref_col, [])
    default_sel: list[str] = []
    if prefill_list:
        default_sel = prefill_list
    if preset_values and feature in preset_values:
        pv = preset_values.get(feature)
        if isinstance(pv, list):
            default_sel = [str(x) for x in pv]
    default_sel = [x for x in default_sel if x in options]
    selected = st.multiselect(label, options=options, default=default_sel, key=val_key)

    if selected and _tmdb_api_key() and ref_col in {"directors", "casting", "producers", "writers", "composer"}:
        known: dict[str, str] = {}
        if ref_col == "casting" and ref_tmdb_id:
            cast = _tmdb_movie_cast(int(ref_tmdb_id), top_n=60) or []
            for member in cast:
                name = str(member.get("name") or "").strip()
                url = str(member.get("profile_url") or "").strip()
                if name and url:
                    known[name] = url
        _render_selected_people_thumbnails(
            [str(x) for x in selected],
            title=("Acteurs sélectionnés" if ref_col == "casting" else "Personnes sélectionnées"),
            known_photos=known,
        )

    return True, float(len(selected))


def _optional_multiselect_list(
    feature: str,
    label: str,
    ref_col: str,
    preset_values: dict | None,
    medians: dict,
    ref_row: dict | None,
    prefill_from_ref: bool,
    state_suffix: str,
    name_options: dict,
    ref_tmdb_id: int | None,
) -> tuple[bool, list[str]]:
    use_key = f"use_{feature}_{state_suffix}"
    val_key = f"val_{feature}_{state_suffix}"

    prefill_list: list[str] = []
    prefill_used = False
    if prefill_from_ref and isinstance(ref_row, dict):
        rr = ref_row
        if isinstance(rr, dict):
            val = rr.get(ref_col)
            if ref_col == "directors" and val is None:
                val = rr.get("director")
            if isinstance(val, list):
                prefill_list = [str(x) for x in val if str(x).strip()]
            elif isinstance(val, str) and val.strip():
                prefill_list = [val.strip()]
            if ref_col in {"directors", "casting", "producers", "writers", "composer"}:
                prefill_list = prefill_list[:5]
            prefill_used = bool(prefill_list)

    pre_has = bool(preset_values and feature in preset_values)
    use = st.checkbox(f"Utiliser {label}", value=(prefill_used or pre_has), key=use_key)
    suggested_list: list[str] | None = None
    if ref_row is not None:
        ref_val = ref_row.get(ref_col)
        if isinstance(ref_val, list):
            suggested_list = [str(x) for x in ref_val]
    # If prefill_from_ref is active and the suggested list corresponds to the prefill list, don't show caption
    try:
        if not (prefill_from_ref and prefill_used and suggested_list is not None and set(suggested_list) <= set(prefill_list)):
            _suggestion_caption(label, suggested_list)
    except Exception:
        _suggestion_caption(label, suggested_list)

    if not use:
        return False, ["__MISSING__"]

    options = name_options.get(ref_col, [])
    default_sel: list[str] = []
    if prefill_list:
        default_sel = prefill_list
    if preset_values and feature in preset_values:
        pv = preset_values.get(feature)
        if isinstance(pv, list):
            default_sel = [str(x) for x in pv]
    default_sel = [x for x in default_sel if x in options]
    selected = st.multiselect(label, options=options, default=default_sel, key=val_key)

    if selected and _tmdb_api_key() and ref_col in {"directors", "casting", "producers", "writers", "composer"}:
        known: dict[str, str] = {}
        if ref_col == "casting" and ref_tmdb_id:
            cast = _tmdb_movie_cast(int(ref_tmdb_id), top_n=60) or []
            for member in cast:
                name = str(member.get("name") or "").strip()
                url = str(member.get("profile_url") or "").strip()
                if name and url:
                    known[name] = url
        _render_selected_people_thumbnails(
            [str(x) for x in selected],
            title=("Acteurs sélectionnés" if ref_col == "casting" else "Personnes sélectionnées"),
            known_photos=known,
        )
    return True, [str(x) for x in selected]


def _prediction_quality_info(used_flags: dict[str, bool], numeric_cols: list[str], identity_cols: list[str]) -> dict[str, object]:
    """Estimate a simple prediction quality indicator based on which fields were provided.

    - `used_flags` maps feature name -> bool (True if user provided/used the feature)
    - `numeric_cols` and `identity_cols` are lists of expected features from the input schema

    Returns a dict with:
    - `completeness`: float (0..1)
    - `level`: one of 'high', 'medium', 'low'
    - `message`: short human-friendly explanation
    """
    num_total = len(numeric_cols) + len(identity_cols)
    if num_total == 0:
        return {"completeness": 1.0, "level": "high", "message": "Aucun champ requis listé dans le schéma."}

    provided = 0
    for c in numeric_cols:
        if used_flags.get(c):
            provided += 1
    for c in identity_cols:
        # for identity columns, accept either the explicit use flag (e.g. 'use_casting') or the column name
        if used_flags.get(c) or used_flags.get(f"use_{c}"):
            provided += 1

    completeness = float(provided) / float(num_total)

    if completeness >= 0.8:
        level = "high"
        msg = "La prédiction utilise la plupart des champs; confiance raisonnable." 
    elif completeness >= 0.5:
        level = "medium"
        msg = "Plusieurs champs manquent; la prédiction peut être moins précise." 
    else:
        level = "low"
        msg = "Peu d'informations fournies; la prédiction est indicative et peu fiable." 

    return {"completeness": completeness, "level": level, "message": msg, "provided": provided, "total": num_total}


class Helpers:
    """OOP facade to import helpers cleanly.

    This keeps pages presentation-focused while allowing a single import:
    `from ml.streamlit_app.helpers import Helpers as H`.

    The underlying module-level functions remain available for backward
    compatibility.
    """

    sanitize_widget_suffix = staticmethod(_sanitize_widget_suffix)
    load_model = staticmethod(_load_model)
    load_schema = staticmethod(_load_schema)
    load_budget_interval = staticmethod(_load_budget_interval)
    format_money = staticmethod(_format_money)
    load_reference_df = staticmethod(_load_reference_df)
    fetch_html = staticmethod(_fetch_html)

    tmdb_api_key = staticmethod(_tmdb_api_key)
    extract_tmdb_movie_id = staticmethod(_extract_tmdb_movie_id)
    tmdb_get_json = staticmethod(_tmdb_get_json)
    tmdb_image_base_url = staticmethod(_tmdb_image_base_url)
    tmdb_profile_url = staticmethod(_tmdb_profile_url)
    tmdb_person_profile_url_by_name = staticmethod(_tmdb_person_profile_url_by_name)
    render_selected_people_thumbnails = staticmethod(_render_selected_people_thumbnails)
    tmdb_movie_cast = staticmethod(_tmdb_movie_cast)

    extract_letterboxd_poster_url = staticmethod(_extract_letterboxd_poster_url)
    get_letterboxd_poster_url = staticmethod(_get_letterboxd_poster_url)

    load_train_features = staticmethod(_load_train_features)
    safe_median = staticmethod(_safe_median)
    q_range = staticmethod(_q_range)
    flatten_unique_lists = staticmethod(_flatten_unique_lists)

    bootstrap_repo_path = staticmethod(_bootstrap_repo_path)
    apply_letterboxd_theme = staticmethod(_apply_letterboxd_theme)

    suggestion_caption = staticmethod(_suggestion_caption)
    optional_slider = staticmethod(_optional_slider)
    optional_multiselect_count = staticmethod(_optional_multiselect_count)
    optional_multiselect_list = staticmethod(_optional_multiselect_list)
    prediction_quality_info = staticmethod(_prediction_quality_info)
