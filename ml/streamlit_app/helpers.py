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


@st.cache_data(ttl=24 * 60 * 60)
def _tmdb_movie_images(tmdb_movie_id: int) -> dict[str, str] | None:
    """Return best-effort poster/backdrop URLs for a TMDB movie."""
    payload = _tmdb_get_json(f"/movie/{tmdb_movie_id}", language="fr-FR")
    if not isinstance(payload, dict):
        return None
    poster_path = payload.get("poster_path")
    backdrop_path = payload.get("backdrop_path")
    out: dict[str, str] = {}
    if isinstance(poster_path, str) and poster_path.strip():
        out["poster_url"] = _tmdb_profile_url(poster_path.strip(), size="w342")
        out["poster_url_large"] = _tmdb_profile_url(poster_path.strip(), size="w780")
    if isinstance(backdrop_path, str) and backdrop_path.strip():
        out["backdrop_url"] = _tmdb_profile_url(backdrop_path.strip(), size="w780")
        out["backdrop_url_large"] = _tmdb_profile_url(backdrop_path.strip(), size="original")
    return out or None


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

                /* Global layout: closer to a Letterboxd-like centered column */
                .stApp, .block-container {{
                    background-color: var(--lb-bg) !important;
                    color: var(--lb-text) !important;
                }}
                .block-container {{
                    max-width: 1100px !important;
                    padding-top: 1.6rem !important;
                    padding-bottom: 2.0rem !important;
                }}

                /* Head/toolbar transparency */
                [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {{
                    background: transparent !important;
                }}

                /* Sidebar: subtle panel */
                [data-testid="stSidebar"] {{
                    background: rgba(255, 255, 255, 0.02) !important;
                    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
                }}
                [data-testid="stSidebar"] img {{
                    border-radius: 10px !important;
                }}

                /* Typography accents */
                .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
                    color: var(--lb-accent) !important;
                }}
                .stMarkdown p, .stMarkdown li {{
                    color: var(--lb-text) !important;
                }}
                .stCaption {{
                    color: rgba(230, 230, 230, 0.72) !important;
                }}

                /* Buttons */
                .stButton > button {{
                    background-color: var(--lb-accent) !important;
                    color: #ffffff !important;
                    border: none !important;
                    border-radius: 10px !important;
                    font-weight: 600 !important;
                    padding: 0.6rem 1rem !important;
                }}
                .stButton > button:hover {{
                    background-color: var(--lb-accent-dark) !important;
                }}

                /* Inputs: darker fields with subtle borders */
                input, textarea {{
                    background: rgba(255, 255, 255, 0.03) !important;
                    color: var(--lb-text) !important;
                    border: 1px solid rgba(255, 255, 255, 0.10) !important;
                    border-radius: 10px !important;
                }}
                input:focus, textarea:focus {{
                    outline: none !important;
                    border-color: rgba(42, 180, 75, 0.60) !important;
                    box-shadow: 0 0 0 2px rgba(42, 180, 75, 0.18) !important;
                }}

                /* Cards: make common containers feel like panels */
                [data-testid="stVerticalBlockBorderWrapper"] {{
                    background: rgba(255, 255, 255, 0.02) !important;
                    border: 1px solid rgba(255, 255, 255, 0.08) !important;
                    border-radius: 12px !important;
                }}

                /* Film hero (Letterboxd-like header) */
                .lb-hero {{
                    position: relative;
                    overflow: hidden;
                    border-radius: 12px;
                    border: 1px solid rgba(255, 255, 255, 0.10);
                    background: rgba(255, 255, 255, 0.02);
                    margin: 0.5rem 0 1rem 0;
                }}
                .lb-hero-bg {{
                    position: absolute;
                    inset: 0;
                    background-position: center;
                    background-size: cover;
                    filter: blur(14px);
                    transform: scale(1.06);
                    opacity: 0.35;
                }}
                .lb-hero-overlay {{
                    position: absolute;
                    inset: 0;
                    background: linear-gradient(90deg, rgba(11, 11, 11, 0.92) 0%, rgba(11, 11, 11, 0.70) 55%, rgba(11, 11, 11, 0.85) 100%);
                }}
                .lb-hero-content {{
                    position: relative;
                    display: flex;
                    gap: 1rem;
                    padding: 1rem;
                    align-items: flex-start;
                }}
                .lb-hero-poster {{
                    width: 140px;
                    flex: 0 0 140px;
                    border-radius: 10px;
                    border: 1px solid rgba(255, 255, 255, 0.14);
                    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
                }}
                .lb-hero-title {{
                    font-size: 1.4rem;
                    font-weight: 800;
                    line-height: 1.2;
                    color: var(--lb-text);
                    margin: 0;
                }}
                .lb-hero-subtitle {{
                    margin-top: 0.25rem;
                    color: rgba(230, 230, 230, 0.72);
                    font-size: 0.95rem;
                }}
                .lb-hero-link a {{
                    color: var(--lb-accent);
                    text-decoration: none;
                    font-weight: 600;
                }}
                .lb-hero-link a:hover {{
                    text-decoration: underline;
                }}

                .lb-hero-cast {{
                    margin-top: 0.75rem;
                    display: flex;
                    gap: 0.6rem;
                    flex-wrap: wrap;
                    align-items: flex-start;
                }}
                .lb-cast-item {{
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    width: 78px;
                }}
                .lb-cast-img {{
                    width: 62px;
                    height: 62px;
                    border-radius: 999px;
                    object-fit: cover;
                    border: 1px solid rgba(255, 255, 255, 0.14);
                    background: rgba(255, 255, 255, 0.04);
                }}
                .lb-cast-name {{
                    margin-top: 0.25rem;
                    font-size: 0.78rem;
                    line-height: 1.1;
                    text-align: center;
                    color: rgba(230, 230, 230, 0.82);
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


def _render_reference_film_hero(
    *,
    title: str,
    year: object | None,
    poster_url: str,
    background_url: str | None = None,
    facts: list[str] | None = None,
    cast: list[dict[str, object]] | None = None,
    letterboxd_url: str | None = None,
) -> None:
    if not poster_url:
        return
    safe_poster = str(poster_url).replace("'", "%27").strip()
    bg = str(background_url or poster_url).replace("'", "%27").strip()
    safe_title = str(title or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    safe_year = ""
    try:
        if year is not None and str(year).strip():
            year_s: str = str(year)
            safe_year = str(int(float(year_s)))
    except Exception:
        safe_year = str(year) if year is not None else ""

    link_html = ""
    if isinstance(letterboxd_url, str) and letterboxd_url.strip().startswith("http"):
        url = letterboxd_url.strip().replace("\"", "%22")
        link_html = (
            f"<div class='lb-hero-link'><a href='{url}' target='_blank' rel='noreferrer'>"
            "Voir sur Letterboxd</a></div>"
        )

    facts_html = ""
    if isinstance(facts, list) and facts:
        safe_facts: list[str] = []
        for f in facts[:6]:
            s = str(f).strip()
            if not s:
                continue
            s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            safe_facts.append(s)
        if safe_facts:
            facts_html = "<div class='lb-hero-subtitle'>" + " • ".join(safe_facts) + "</div>"

    cast_html = ""
    if isinstance(cast, list) and cast:
        items: list[str] = []
        for member in cast[:5]:
            if not isinstance(member, dict):
                continue
            name = str(member.get("name") or "").strip()
            if not name:
                continue
            name_safe = name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            url = member.get("profile_url")
            if isinstance(url, str) and url.strip().startswith("http"):
                u = url.strip().replace("'", "%27")
                img_html = f"<img class='lb-cast-img' src='{u}' alt='{name_safe}'/>"
            else:
                img_html = "<div class='lb-cast-img'></div>"
            items.append(
                "<div class='lb-cast-item'>" + img_html + f"<div class='lb-cast-name'>{name_safe}</div></div>"
            )
        if items:
            cast_html = "<div class='lb-hero-cast'>" + "".join(items) + "</div>"

    st.markdown(
        f"""
        <div class="lb-hero">
          <div class="lb-hero-bg" style="background-image:url('{bg}');"></div>
          <div class="lb-hero-overlay"></div>
          <div class="lb-hero-content">
            <img class="lb-hero-poster" src="{safe_poster}" alt="Poster" />
            <div>
              <div class="lb-hero-title">{safe_title}{f" <span style='color:rgba(230,230,230,0.72); font-weight:700'>({safe_year})</span>" if safe_year else ""}</div>
              {facts_html if facts_html else '<div class="lb-hero-subtitle">Film de référence sélectionné</div>'}
              {link_html}
                            {cast_html}
            </div>
          </div>
        </div>
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
                if isinstance(sugg, (list, dict)) or isinstance(prefill, (list, dict)):
                    return False
                if isinstance(sugg, (int, float, np.number)) and isinstance(prefill, (int, float, np.number)):
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


def _explain_numeric_deltas(
    *,
    values: dict[str, object],
    used_flags: dict[str, bool],
    medians: dict[str, float],
    labels: dict[str, str],
    top_n: int = 5,
    money_features: set[str] | None = None,
) -> list[dict[str, object]]:
    """Explain which *numeric* inputs deviated most from neutral (median) values.

    This is not feature importance: it only highlights how far the user-provided
    inputs are from the dataset median, as a quick sanity-check.
    """

    if money_features is None:
        money_features = {"budget", "revenue"}

    rows: list[dict[str, object]] = []
    for feature, label in labels.items():
        if not used_flags.get(feature):
            continue

        raw_val: Any = values.get(feature)
        try:
            v = float(raw_val) if raw_val is not None else float("nan")  # type: ignore[arg-type]
        except Exception:
            continue
        if not np.isfinite(v):
            continue

        m = float(medians.get(feature, 0.0))
        d = float(v - m)

        def _fmt(x: float) -> str:
            if feature in money_features:
                return _format_money(float(x))
            if float(x).is_integer():
                return f"{int(x):,}".replace(",", " ")
            return f"{x:,.2f}".replace(",", " ")

        fmt_v = _fmt(v)
        fmt_m = _fmt(m)
        if feature in money_features:
            sign = "+" if d >= 0 else "−"
            fmt_d = f"{sign}{_format_money(abs(d))}"
        else:
            sign = "+" if d >= 0 else "−"
            if float(abs(d)).is_integer():
                fmt_d = f"{sign}{int(abs(d)):,}".replace(",", " ")
            else:
                fmt_d = f"{sign}{abs(d):,.2f}".replace(",", " ")

        rows.append(
            {
                "feature": feature,
                "label": label,
                "value": v,
                "median": m,
                "delta": d,
                "abs_delta": float(abs(d)),
                "value_display": fmt_v,
                "median_display": fmt_m,
                "delta_display": fmt_d,
            }
        )

    rows.sort(key=lambda r: float(r["abs_delta"]), reverse=True)  # type: ignore[arg-type]
    return rows[: max(0, int(top_n))]


def _numeric_correlations(
    df: pd.DataFrame,
    *,
    features: list[str],
    target: str = "rating",
    min_n: int = 30,
) -> pd.DataFrame:
    """Compute Pearson/Spearman correlations vs target for each numeric feature.

    Returns a DataFrame with columns: feature, pearson, spearman, n.
    """
    if df is None or df.empty or target not in df.columns:
        return pd.DataFrame(columns=["feature", "pearson", "spearman", "n"])  # type: ignore[return-value]

    out_rows: list[dict[str, object]] = []
    tgt = pd.to_numeric(df[target], errors="coerce")

    for f in features:
        if f not in df.columns:
            continue
        x = pd.to_numeric(df[f], errors="coerce")
        mask = np.isfinite(x.to_numpy(dtype=float, na_value=np.nan)) & np.isfinite(tgt.to_numpy(dtype=float, na_value=np.nan))
        n = int(mask.sum())
        if n < int(min_n):
            continue
        try:
            pearson = float(pd.Series(x[mask]).corr(pd.Series(tgt[mask]), method="pearson"))
        except Exception:
            pearson = float("nan")
        try:
            spearman = float(pd.Series(x[mask]).corr(pd.Series(tgt[mask]), method="spearman"))
        except Exception:
            spearman = float("nan")
        out_rows.append({"feature": f, "pearson": pearson, "spearman": spearman, "n": n})

    out = pd.DataFrame(out_rows)
    if out.empty:
        return out
    out["abs_spearman"] = out["spearman"].abs()
    out = out.sort_values(["abs_spearman", "n"], ascending=[False, False]).drop(columns=["abs_spearman"])  # type: ignore[arg-type]
    return out


def _altair_scatter_with_regression(
    df: pd.DataFrame,
    *,
    feature: str,
    target: str = "rating",
    user_x: float | None = None,
    user_y: float | None = None,
    width: int = 520,
    height: int = 280,
):
    """Return an Altair chart: scatter + regression line + user marker/rule."""
    try:
        import altair as alt  # type: ignore
    except Exception:
        return None

    if df is None or df.empty or feature not in df.columns or target not in df.columns:
        return None

    d = df[[feature, target]].copy()
    d[feature] = pd.to_numeric(d[feature], errors="coerce")
    d[target] = pd.to_numeric(d[target], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 10:
        return None

    base = alt.Chart(d).properties(width=width, height=height)

    pts = base.mark_circle(size=22, opacity=0.18).encode(
        x=alt.X(f"{feature}:Q", title=feature),
        y=alt.Y(f"{target}:Q", title=target),
        tooltip=[alt.Tooltip(f"{feature}:Q"), alt.Tooltip(f"{target}:Q")],
    )
    reg = base.transform_regression(feature, target).mark_line(opacity=0.8).encode(
        x=alt.X(f"{feature}:Q"),
        y=alt.Y(f"{target}:Q"),
    )

    layers = [pts, reg]

    if user_x is not None and np.isfinite(float(user_x)):
        rule = alt.Chart(pd.DataFrame({"x": [float(user_x)]})).mark_rule(opacity=0.7).encode(x="x:Q")
        layers.append(rule)

        if user_y is not None and np.isfinite(float(user_y)):
            user_pt = alt.Chart(pd.DataFrame({"x": [float(user_x)], "y": [float(user_y)]})).mark_point(size=120, filled=True).encode(
                x="x:Q",
                y="y:Q",
                tooltip=[alt.Tooltip("x:Q", title=feature), alt.Tooltip("y:Q", title="note prédite")],
            )
            layers.append(user_pt)

    return alt.layer(*layers)


def _altair_histogram_with_rule(
    df: pd.DataFrame,
    *,
    feature: str,
    user_x: float | None = None,
    width: int = 520,
    height: int = 160,
    max_bins: int = 40,
):
    """Return an Altair histogram of feature distribution, with an optional rule at user_x."""
    try:
        import altair as alt  # type: ignore
    except Exception:
        return None

    if df is None or df.empty or feature not in df.columns:
        return None

    d = pd.DataFrame({feature: pd.to_numeric(df[feature], errors="coerce")})
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 10:
        return None

    hist = alt.Chart(d).mark_bar(opacity=0.7).encode(
        x=alt.X(f"{feature}:Q", bin=alt.Bin(maxbins=int(max_bins)), title=None),
        y=alt.Y("count():Q", title="N"),
        tooltip=[alt.Tooltip("count():Q", title="N")],
    ).properties(width=width, height=height)

    if user_x is not None and np.isfinite(float(user_x)):
        rule = alt.Chart(pd.DataFrame({"x": [float(user_x)]})).mark_rule(opacity=0.8).encode(x="x:Q")
        return alt.layer(hist, rule)

    return hist


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
    tmdb_movie_images = staticmethod(_tmdb_movie_images)

    extract_letterboxd_poster_url = staticmethod(_extract_letterboxd_poster_url)
    get_letterboxd_poster_url = staticmethod(_get_letterboxd_poster_url)

    load_train_features = staticmethod(_load_train_features)
    safe_median = staticmethod(_safe_median)
    q_range = staticmethod(_q_range)
    flatten_unique_lists = staticmethod(_flatten_unique_lists)

    bootstrap_repo_path = staticmethod(_bootstrap_repo_path)
    apply_letterboxd_theme = staticmethod(_apply_letterboxd_theme)

    render_reference_film_hero = staticmethod(_render_reference_film_hero)

    suggestion_caption = staticmethod(_suggestion_caption)
    optional_slider = staticmethod(_optional_slider)
    optional_multiselect_count = staticmethod(_optional_multiselect_count)
    optional_multiselect_list = staticmethod(_optional_multiselect_list)
    prediction_quality_info = staticmethod(_prediction_quality_info)
    explain_numeric_deltas = staticmethod(_explain_numeric_deltas)
    numeric_correlations = staticmethod(_numeric_correlations)
    altair_scatter_with_regression = staticmethod(_altair_scatter_with_regression)
    altair_histogram_with_rule = staticmethod(_altair_histogram_with_rule)
