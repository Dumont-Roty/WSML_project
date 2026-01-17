from __future__ import annotations

import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
import math

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Bootstrap paths so `src.*` is importable (IdentityHasher is pickled as src.utils.identity_hasher.IdentityHasher)
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parent.parent  # project root
ML_DIR = REPO_ROOT / "ml"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.utils.identity_hasher  # noqa: F401

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


@st.cache_data
def _load_test_rmse(metrics_path: Path) -> float | None:
    """Load test RMSE from a training metrics.json file.

    Expected formats:
    - {"test_metrics": {"rmse": ...}}
    - {"rmse": ...}
    """

    if not metrics_path.exists():
        return None
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    test_metrics = payload.get("test_metrics")
    if isinstance(test_metrics, dict):
        rmse_val = test_metrics.get("rmse")
        if rmse_val is not None:
            try:
                v = float(rmse_val)
                return v if np.isfinite(v) else None
            except Exception:
                return None

    rmse_val = payload.get("rmse")
    if rmse_val is not None:
        try:
            v = float(rmse_val)
            return v if np.isfinite(v) else None
        except Exception:
            return None

    return None


def _z_value_for_central_coverage(coverage: float) -> float:
    """Return a z multiplier for a central normal interval.

    coverage is the desired central probability mass (e.g. 0.8 for an 80% interval).
    """

    try:
        c = float(coverage)
    except Exception:
        c = 0.8

    # Common defaults; keep explicit mapping to avoid scipy dependency.
    if abs(c - 0.80) < 1e-6:
        return 1.2815515655446004
    if abs(c - 0.90) < 1e-6:
        return 1.6448536269514722
    if abs(c - 0.95) < 1e-6:
        return 1.959963984540054
    # Fallback: use 80% if an unknown value is provided.
    return 1.2815515655446004


def _approx_prediction_interval_from_rmse(
    y_pred: float,
    rmse: float,
    *,
    coverage: float = 0.8,
    clip: tuple[float, float] = (0.0, 5.0),
) -> dict[str, float] | None:
    """Approximate a prediction interval using RMSE and a normal-error assumption."""

    try:
        y = float(y_pred)
        r = float(rmse)
    except Exception:
        return None

    if not (np.isfinite(y) and np.isfinite(r)):
        return None
    if r < 0:
        return None

    z = _z_value_for_central_coverage(coverage)
    half = float(z * r)
    low = float(y - half)
    high = float(y + half)

    try:
        cmin, cmax = float(clip[0]), float(clip[1])
        if np.isfinite(cmin) and np.isfinite(cmax):
            low = float(np.clip(low, cmin, cmax))
            high = float(np.clip(high, cmin, cmax))
    except Exception:
        pass

    if high < low:
        low, high = high, low

    return {
        "low": low,
        "high": high,
        "coverage": float(coverage),
        "half_width": half,
        "width": float(high - low),
    }


def _ood_numeric_info(
    values: dict[str, object],
    used_flags: dict[str, bool],
    stats_df: pd.DataFrame,
    numeric_cols: list[str],
    *,
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> dict[str, object]:
    """Detect out-of-distribution numeric inputs using empirical quantiles.

    This is intentionally simple: for each used numeric feature, we compare the
    user value against the [q_low, q_high] quantile range computed on stats_df
    (train.parquet if available, otherwise reference JSON).
    """

    if stats_df is None or getattr(stats_df, "empty", True):
        return {"n_checked": 0, "n_outside": 0, "outside": [], "level": "unknown"}

    try:
        ql = float(q_low)
        qh = float(q_high)
    except Exception:
        ql, qh = 0.01, 0.99
    if not (0.0 < ql < qh < 1.0):
        ql, qh = 0.01, 0.99

    outside: list[dict[str, object]] = []
    n_checked = 0

    for c in numeric_cols:
        if not used_flags.get(c):
            continue
        if c not in stats_df.columns:
            continue

        raw_v: Any = values.get(c)
        try:
            v = float(raw_v) if raw_v is not None else float("nan")
        except Exception:
            continue
        if not np.isfinite(v):
            continue

        s = pd.to_numeric(stats_df[c], errors="coerce")
        s = s[np.isfinite(s)]
        if s.empty:
            continue

        try:
            lo = float(s.quantile(ql))
            hi = float(s.quantile(qh))
        except Exception:
            continue
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        if lo >= hi:
            continue

        n_checked += 1
        if v < lo or v > hi:
            outside.append({"feature": str(c), "value": v, "q_low": lo, "q_high": hi})

    n_outside = int(len(outside))
    if n_checked == 0:
        level = "unknown"
        msg = ""
    elif n_outside == 0:
        level = "high"
        msg = "Valeurs numériques dans les plages usuelles du dataset."
    elif n_outside == 1:
        level = "medium"
        msg = "Une valeur semble atypique vs le dataset."
    else:
        level = "low"
        msg = "Plusieurs valeurs semblent atypiques vs le dataset."

    return {
        "n_checked": n_checked,
        "n_outside": n_outside,
        "outside": outside,
        "level": level,
        "message": msg,
        "q_low": ql,
        "q_high": qh,
    }


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
    st.caption(f"{title} (aperçu)")


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
    lo = float(fallback_min)
    hi = float(fallback_max)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return float(fallback_min), float(fallback_max)
    return float(max(fallback_min, lo)), float(min(fallback_max, hi))


def _build_actor_photos_cache(df: pd.DataFrame) -> dict[str, str]:
    """
    Construit un dictionnaire {nom_acteur: profile_url} depuis la base de données.
    
    Utilise le champ `cast_profiles` si présent dans les films, sinon fait une
    recherche TMDB par nom (fallback).
    
    Returns:
        dict mappant nom d'acteur → URL de photo TMDB
    """
    cache: dict[str, str] = {}
    
    if df.empty or "casting" not in df.columns:
        return cache
    
    # Extraire tous les cast_profiles depuis les films
    if "cast_profiles" in df.columns:
        for idx, row in df.iterrows():
            profiles = row.get("cast_profiles")
            if isinstance(profiles, dict):
                for name, profile_path in profiles.items():
                    if name and profile_path and isinstance(name, str) and isinstance(profile_path, str):
                        name_clean = str(name).strip()
                        if name_clean and name_clean not in cache:
                            # Construire l'URL complète
                            cache[name_clean] = _tmdb_profile_url(profile_path.strip(), size="w185")
    
    return cache


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


def _similarity_weight(col_name: str) -> float:
    """Retourne le poids d'un critère pour la similarité globale.
    
    Pondération par catégories (1 = plus fort) :
    1) genres / themes : 2.0
    2) rating : 1.8
    3) directors / actors / casting / cast : 1.5
    4) budget / revenue : 1.2
    5) reste : 1.0
    """
    c = str(col_name).lower().strip()
    # Cat 1 : genres / themes
    if c in {"genres", "genre", "themes", "theme"}:
        return 2.0

    # Cat 2 : rating
    if c == "rating":
        return 1.8

    # Cat 3 : directors / actors / casting
    if c in {"director", "directors", "actors", "casting", "cast"}:
        return 1.5

    # Cat 4 : budget / revenue
    if c in {"budget", "revenue"}:
        return 1.2

    # Cat 5 : le reste
    return 1.0


def _similar_movies(
    ref_df: pd.DataFrame,
    user_values: dict[str, object],
    numeric_cols: list[str],
    identity_cols: list[str],
    *,
    top_n: int = 3,
) -> list[dict[str, object]]:
    """Retourne les films les plus proches des saisies utilisateur.

    - Numériques : similarité = 1 - diff/span, span = (max-min) sur le dataset, bornée à [0,1].
    - Identités (listes) : similarité intersection / max(user_count, ref_count).
    - Score global = moyenne pondérée des similarités.
    """

    if ref_df is None or getattr(ref_df, "empty", True):
        return []

    # Infer columns from JSON to cover all criteria
    def _infer_numeric_cols(df: pd.DataFrame) -> set[str]:
        out: set[str] = set()
        for c in df.columns:
            # Exclure le titre et autres champs non pertinents pour la similarité
            if str(c).lower() in ("title", "url", "letterboxd_url", "letterboxd", "link", "id", "idx"):
                continue
            try:
                if pd.api.types.is_numeric_dtype(df[c]):
                    out.add(str(c))
                    continue
                s = pd.to_numeric(df[c], errors="coerce")
                if s.notna().any():
                    out.add(str(c))
            except Exception:
                pass
        return out

    def _infer_list_cols(df: pd.DataFrame) -> set[str]:
        out: set[str] = set()
        for c in df.columns:
            try:
                if df[c].apply(lambda v: isinstance(v, list)).any():
                    out.add(str(c))
            except Exception:
                pass
        return out

    num_cols = set(numeric_cols or []) | _infer_numeric_cols(ref_df)
    id_cols = set(identity_cols or []) | _infer_list_cols(ref_df)

    spans: dict[str, float] = {}
    for col in num_cols:
        if col in ref_df.columns:
            s = pd.to_numeric(ref_df[col], errors="coerce")
            s = s[np.isfinite(s)]
            if not s.empty:
                span = float(s.max() - s.min())
                spans[col] = span if span > 0 else 1.0

    def _num_sim(col: str, uval: object, rval: object) -> float | None:
        if not isinstance(uval, (int, float)) or not isinstance(rval, (int, float)):
            return None
        if col not in spans:
            return None
        span = spans[col]
        d = abs(float(uval) - float(rval))
        sim = 1.0 - min(d / span, 1.0)
        return float(sim)

    actor_cols = {"casting", "actors", "cast"}

    def _list_sim(col: str, uval: object, rval: object) -> float | None:
        if not isinstance(uval, list) or not isinstance(rval, list):
            return None

        u_list = [str(x).strip().lower() for x in uval if str(x).strip()]
        r_list = [str(x).strip().lower() for x in rval if str(x).strip()]

        # Ne comparer que les 5 premiers acteurs pour limiter le bruit
        if col in actor_cols:
            u_list = u_list[:5]
            r_list = r_list[:5]

        u = set(u_list)
        r = set(r_list)
        if not u or not r:
            return None
        inter = len(u & r)
        max_count = max(len(u), len(r))
        if max_count == 0:
            return None
        return float(inter / max_count)

    def _list_contrib(col: str, uval: object, rval: object) -> dict | None:
        if not isinstance(uval, list) or not isinstance(rval, list):
            return {
                "name": col,
                "type": "identity",
                "similarity": 0.0,
                "status": "missing",
                "details": {"reason": "Données manquantes ou format invalide"},
            }
        u = [str(x).strip() for x in uval if x is not None and str(x).strip()]
        r = [str(x).strip() for x in rval if x is not None and str(x).strip()]

        # Limiter aux 5 premiers acteurs pour réduire le bruit
        if col in {"casting", "actors", "cast"}:
            u = u[:5]
            r = r[:5]

        if not u or not r:
            return {
                "name": col,
                "type": "identity",
                "similarity": 0.0,
                "status": "empty",
                "details": {"reason": "Liste vide après nettoyage", "user_count": 0, "ref_count": 0},
            }
        us = set([s.lower() for s in u])
        rs = set([s.lower() for s in r])
        inter = us & rs
        max_count = max(len(us), len(rs))
        if max_count == 0:
            return {
                "name": col,
                "type": "identity",
                "similarity": 0.0,
                "status": "no_overlap",
                "details": {"reason": "Aucun recoupement", "user_count": len(us), "ref_count": len(rs)},
            }
        sim = float(len(inter) / max_count)
        return {
            "name": col,
            "type": "identity",
            "similarity": round(100.0 * sim, 1),
            "status": "ok",
            "details": {
                "matched": sorted(list(inter)),
                "user_count": len(us),
                "ref_count": len(rs),
            },
        }

    rows: list[dict[str, object]] = []
    for idx, row in ref_df.iterrows():
        total_weighted_sim = 0.0
        total_weight = 0.0

        for col in num_cols:
            if col not in ref_df.columns:
                continue
            sim = _num_sim(col, user_values.get(col), row.get(col))
            if sim is not None:
                weight = _similarity_weight(col)
                total_weighted_sim += sim * weight
                total_weight += weight

        for col in id_cols:
            if col not in ref_df.columns:
                continue
            sim = _list_sim(col, user_values.get(col), row.get(col))
            if sim is not None:
                weight = _similarity_weight(col)
                total_weighted_sim += sim * weight
                total_weight += weight

        if total_weight == 0:
            continue
        score = float(total_weighted_sim / total_weight)
        rows.append(
            {
                "_score": score,
                "title": row.get("title"),
                "year": row.get("year"),
                "url": row.get("url") or row.get("letterboxd_url"),
                "idx": idx,
            }
        )

    if not rows:
        return []

    rows = sorted(rows, key=lambda r: float(r.get("_score") or 0.0), reverse=True)[: int(top_n)]
    for r in rows:
        score = float(r.get("_score", 0.0))
        # Remplacer NaN par 0.0 et s'assurer que le résultat est valide
        if not np.isfinite(score):
            score = 0.0
        r["similarity_pct"] = round(100.0 * score, 1)
    return rows


def _letterboxd_url_from_row(row: dict) -> str | None:
    for k in ("url", "letterboxd_url", "letterboxd", "link"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _similar_movies_with_explanations(
    ref_df: pd.DataFrame,
    user_values: dict[str, object],
    numeric_cols: list[str],
    identity_cols: list[str],
    *,
    top_n: int = 3,
) -> list[dict[str, object]]:
    """Top-N films similaires avec explications par feature.

    Renvoie une liste de dict: {
      title, year, url, poster_url, similarity_pct, features: [ {name, type, similarity, details} ]
    }.
    """
    if ref_df is None or getattr(ref_df, "empty", True):
        return []

    # Infer full sets of columns from JSON and precompute spans
    def _infer_numeric_cols(df: pd.DataFrame) -> set[str]:
        out: set[str] = set()
        for c in df.columns:
            # Exclure le titre et autres champs non pertinents pour la similarité
            if str(c).lower() in ("title", "url", "letterboxd_url", "letterboxd", "link", "id", "idx"):
                continue
            try:
                if pd.api.types.is_numeric_dtype(df[c]):
                    out.add(str(c))
                    continue
                s = pd.to_numeric(df[c], errors="coerce")
                if s.notna().any():
                    out.add(str(c))
            except Exception:
                pass
        return out

    def _infer_list_cols(df: pd.DataFrame) -> set[str]:
        out: set[str] = set()
        for c in df.columns:
            try:
                if df[c].apply(lambda v: isinstance(v, list)).any():
                    out.add(str(c))
            except Exception:
                pass
        return out

    num_cols = set(numeric_cols or []) | _infer_numeric_cols(ref_df)
    id_cols = set(identity_cols or []) | _infer_list_cols(ref_df)

    # Reuse base ranking from _similar_movies
    base = _similar_movies(ref_df, user_values, numeric_cols, identity_cols, top_n=max(top_n * 3, 10))
    if not base:
        return []

    spans: dict[str, float] = {}
    for col in num_cols:
        if col in ref_df.columns:
            s = pd.to_numeric(ref_df[col], errors="coerce")
            s = s[np.isfinite(s)]
            if not s.empty:
                span = float(s.max() - s.min())
                spans[col] = span if span > 0 else 1.0

    def _num_contrib(col: str, uval: object, rval: object) -> dict | None:
        if not isinstance(uval, (int, float)) or not isinstance(rval, (int, float)):
            # Données manquantes = malus de 0%
            return {
                "name": col,
                "type": "numeric",
                "similarity": 0.0,
                "status": "missing",
                "details": {"reason": "Données manquantes ou non numériques", "user": uval, "ref": rval},
            }
        u = float(uval)
        r = float(rval)
        span = spans.get(col)
        if not span:
            return {
                "name": col,
                "type": "numeric",
                "similarity": 0.0,
                "status": "no_span",
                "details": {"reason": "Aucun écart détectable dans le dataset", "user": u, "ref": r},
            }
        d = abs(u - r)
        sim = 1.0 - min(d / span, 1.0)
        return {
            "name": col,
            "type": "numeric",
            "similarity": round(100.0 * float(sim), 1),
            "status": "ok",
            "details": {"user": u, "ref": r, "diff": d, "span": span},
        }

    def _list_contrib(col: str, uval: object, rval: object) -> dict | None:
        if not isinstance(uval, list) or not isinstance(rval, list):
            return {
                "name": col,
                "type": "identity",
                "similarity": 0.0,
                "status": "missing",
                "details": {"reason": "Données manquantes ou format invalide"},
            }
        u = [str(x).strip() for x in uval if x is not None and str(x).strip()]
        r = [str(x).strip() for x in rval if x is not None and str(x).strip()]
        if not u or not r:
            return {
                "name": col,
                "type": "identity",
                "similarity": 0.0,
                "status": "empty",
                "details": {"reason": "Liste vide après nettoyage", "user_count": 0, "ref_count": 0},
            }
        us = set([s.lower() for s in u])
        rs = set([s.lower() for s in r])
        inter = us & rs
        
        # Similarité basée sur l'identité : intersection / max(user_count, ref_count)
        # Mesure la proportion d'éléments communs sans pénaliser les différences de taille
        max_count = max(len(us), len(rs))
        if not inter:
            return {
                "name": col,
                "type": "identity",
                "similarity": 0.0,
                "status": "no_overlap",
                "details": {"reason": "Aucun élément en commun", "user_count": len(us), "ref_count": len(rs)},
            }
        j = float(len(inter)) / float(max_count)
        return {
            "name": col,
            "type": "identity",
            "similarity": round(100.0 * j, 1),
            "status": "ok",
            "details": {"matched": sorted(list(inter))[:6], "user_count": len(us), "ref_count": len(rs), "intersection_count": len(inter)},
        }

    out: list[dict[str, object]] = []
    for item in base:
        idx = item.get("idx")
        if idx is None:
            continue
        try:
            row = ref_df.loc[int(idx)]
        except Exception:
            continue
        features_info: list[dict[str, object]] = []
        # numeric
        for col in num_cols:
            if col in ref_df.columns:
                contrib = _num_contrib(col, user_values.get(col), row.get(col))
                if contrib:
                    features_info.append(contrib)
        # identities
        for col in id_cols:
            if col in ref_df.columns:
                contrib = _list_contrib(col, user_values.get(col), row.get(col))
                if contrib:
                    features_info.append(contrib)

        lb_url = _letterboxd_url_from_row(dict(row))
        poster_url = None
        if lb_url:
            try:
                poster_url = _get_letterboxd_poster_url(lb_url)
            except Exception:
                poster_url = None
            # Fallback to TMDB official poster if Letterboxd poster is missing
            if not poster_url:
                try:
                    lb_html = _fetch_html(lb_url)
                    tmdb_id = _extract_tmdb_movie_id(lb_html)
                    if isinstance(tmdb_id, int):
                        tmdb_imgs = _tmdb_movie_images(tmdb_id)
                        if isinstance(tmdb_imgs, dict):
                            poster_url = tmdb_imgs.get("poster_url") or tmdb_imgs.get("poster_url_large")
                except Exception:
                    pass

        out.append(
            {
                "title": row.get("title"),
                "year": row.get("year"),
                "url": lb_url,
                "poster_url": poster_url,
                "similarity_pct": item.get("similarity_pct"),
                "features": features_info,
            }
        )

    # Keep top_n with most features info and highest similarity
    # Nettoyer les NaN dans similarity_pct
    for item in out:
        sim_pct = item.get("similarity_pct")
        if isinstance(sim_pct, (int, float)):
            if not np.isfinite(float(sim_pct)):
                item["similarity_pct"] = 0.0
        else:
            item["similarity_pct"] = 0.0

    def _sim_sort_key(r: dict[str, object]) -> tuple[float, int]:
        sim_val = r.get("similarity_pct")
        sim_float = float(sim_val) if isinstance(sim_val, (int, float)) else 0.0
        feats = r.get("features")
        feat_count = len(feats) if isinstance(feats, list) else 0
        return (sim_float, feat_count)

    out = sorted(out, key=_sim_sort_key, reverse=True)[: int(top_n)]
    return out


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
    if background_url:
        bg = str(background_url).replace("'", "%27").strip()
    else:
        bg = ""
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

    html_parts = [
        '<div class="lb-hero">',
        f'<div class="lb-hero-bg" style="background-image:url(\'{bg}\');"></div>',
        '<div class="lb-hero-overlay"></div>',
        '<div class="lb-hero-content">',
        f'<img class="lb-hero-poster" src="{safe_poster}" alt="Poster" />',
        '<div>',
        f'<div class="lb-hero-title">{safe_title}',
        f' <span style="color:rgba(230,230,230,0.72); font-weight:700">({safe_year})</span>' if safe_year else '',
        '</div>',
        facts_html if facts_html else '<div class="lb-hero-subtitle">Film de référence sélectionné</div>',
        link_html,
        cast_html,
        '</div>',
        '</div>',
        '</div>',
    ]
    st.markdown("".join(html_parts), unsafe_allow_html=True)


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

    # compute safe float bounds and default
    try:
        lo_f = float(lo)
    except Exception:
        lo_f = float(fallback_min)
    try:
        hi_f = float(hi)
    except Exception:
        hi_f = float(fallback_max)
    if not np.isfinite(lo_f):
        lo_f = float(fallback_min)
    if not np.isfinite(hi_f):
        hi_f = float(fallback_max)
    if hi_f < lo_f:
        hi_f = lo_f + 1.0

    try:
        default_f = float(default)
    except Exception:
        default_f = float(lo_f)
    if not np.isfinite(default_f):
        default_f = float(lo_f)

    # integer-safe bounds
    min_i = int(math.floor(lo_f))
    max_i = int(math.ceil(hi_f))
    if max_i <= min_i:
        max_i = min_i + 1
    default_i = int(round(default_f))
    if default_i < min_i:
        default_i = min_i
    if default_i > max_i:
        default_i = max_i

    def _fmt_fr_number(x: float, *, decimals: int = 2) -> str:
        s = f"{float(x):,.{int(decimals)}f}"
        return s.replace(",", " ").replace(".", ",")

    def _fmt_fr_int(x: float) -> str:
        try:
            return f"{int(round(float(x))):,}".replace(",", " ")
        except Exception:
            return "0"

    if is_int:
        if feature in ("budget", "revenue"):
            scale_key = f"log_{feature}_{state_suffix}"
            raw_key = f"{val_key}__raw"
            log_key = f"{val_key}__log"

            use_log = st.checkbox("Échelle log (log1p)", value=False, key=scale_key)

            current_val = None
            for k in (raw_key, log_key):
                try:
                    if k in st.session_state:
                        current_val = float(st.session_state[k])
                        break
                except Exception:
                    current_val = None
            if current_val is None or not np.isfinite(float(current_val)):
                current_val = float(default_i)

            if use_log:
                lo_money = max(0.0, float(min_i))
                hi_money = max(lo_money + 1.0, float(max_i))
                lo_log = float(np.log1p(lo_money))
                hi_log = float(np.log1p(hi_money))
                d_log = float(np.log1p(max(0.0, current_val)))
                d_log = float(np.clip(d_log, lo_log, hi_log))

                v_log = st.slider(label, lo_log, hi_log, d_log, key=log_key, format="%.2f")
                v = float(np.expm1(float(v_log)))
                v_i = int(round(float(np.clip(v, float(min_i), float(max_i)))))
            else:
                v_i = int(
                    st.number_input(
                        label,
                        min_value=int(min_i),
                        max_value=int(max_i),
                        value=int(round(current_val)),
                        key=raw_key,
                        step=1,
                    )
                )

            try:
                st.caption(f"Valeur sélectionnée : {_format_money(float(v_i))}")
            except Exception:
                pass
            return True, float(v_i)

        val = st.slider(label, min_i, max_i, default_i, key=val_key, step=1)
        try:
            st.caption(f"Valeur sélectionnée : {_fmt_fr_int(float(val))}")
        except Exception:
            pass
        return True, float(val)

    if feature in ("budget", "revenue"):
        scale_key = f"log_{feature}_{state_suffix}"
        raw_key = f"{val_key}__raw"
        log_key = f"{val_key}__log"

        use_log = st.checkbox("Échelle log (log1p)", value=False, key=scale_key)

        current_val = None
        for k in (raw_key, log_key):
            try:
                if k in st.session_state:
                    current_val = float(st.session_state[k])
                    break
            except Exception:
                current_val = None
        if current_val is None or not np.isfinite(float(current_val)):
            current_val = float(default_i)

        if use_log:
            lo_money = max(0.0, float(min_i))
            hi_money = max(lo_money + 1.0, float(max_i))
            lo_log = float(np.log1p(lo_money))
            hi_log = float(np.log1p(hi_money))
            d_log = float(np.log1p(max(0.0, current_val)))
            d_log = float(np.clip(d_log, lo_log, hi_log))

            v_log = st.slider(label, lo_log, hi_log, d_log, key=log_key, format="%.2f")
            v = float(np.expm1(float(v_log)))
            v_i = int(round(float(np.clip(v, float(min_i), float(max_i)))))
        else:
            v_i = int(
                st.number_input(
                    label,
                    min_value=int(min_i),
                    max_value=int(max_i),
                    value=int(round(current_val)),
                    key=raw_key,
                    step=1,
                )
            )

        try:
            st.caption(f"Valeur sélectionnée : {_format_money(float(v_i))}")
        except Exception:
            pass
        return True, float(v_i)

    val = st.slider(label, lo_f, hi_f, default_f, key=val_key, format="%.2f")
    try:
        st.caption(f"Valeur sélectionnée : {_fmt_fr_number(float(val), decimals=2)}")
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
    actor_photos_cache: dict[str, str] | None = None,
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
        # Start with cache from database
        if actor_photos_cache:
            known.update(actor_photos_cache)
        # Merge with TMDB cast data if reference movie is provided
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
    actor_photos_cache: dict[str, str] | None = None,
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
        # Start with cache from database
        if actor_photos_cache:
            known.update(actor_photos_cache)
        # Merge with TMDB cast data if reference movie is provided
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
    int_features: set[str] | None = None,
) -> list[dict[str, object]]:
    """Explain which *numeric* inputs deviated most from neutral (median) values.

    This is not feature importance: it only highlights how far the user-provided
    inputs are from the dataset median, as a quick sanity-check.
    """

    if money_features is None:
        money_features = {"budget", "revenue"}
    if int_features is None:
        int_features = set()

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

        # For integer-like features, force a stable integer baseline to avoid
        # confusing .5 medians (even N) and fractional deltas.
        if feature in int_features and feature not in money_features:
            v = float(int(round(v)))

        m = float(medians.get(feature, 0.0))
        if feature in int_features and feature not in money_features:
            m = float(int(round(m)))

        d = float(v - m)

        def _fmt(x: float) -> str:
            if feature in money_features:
                return _format_money(float(x))
            if (feature in int_features) or float(x).is_integer():
                return f"{int(x):,}".replace(",", " ")
            return f"{x:,.2f}".replace(",", " ").replace(".", ",")

        fmt_v = _fmt(v)
        fmt_m = _fmt(m)
        if feature in money_features:
            sign = "+" if d >= 0 else "−"
            fmt_d = f"{sign}{_format_money(abs(d))}"
        else:
            sign = "+" if d >= 0 else "−"
            if (feature in int_features) or float(abs(d)).is_integer():
                fmt_d = f"{sign}{int(abs(d)):,}".replace(",", " ")
            else:
                fmt_d = f"{sign}{abs(d):,.2f}".replace(",", " ").replace(".", ",")

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


def _numeric_deltas_table_with_mean(
    deltas: list[dict[str, object]],
    *,
    stats_df: pd.DataFrame,
    money_features: set[str] | None = None,
    int_features: set[str] | None = None,
) -> pd.DataFrame:
    """Build a display-ready table for numeric deltas, including mean comparison.

    Input `deltas` is expected to be the output of `_explain_numeric_deltas`.
    Output columns are user-facing French labels.
    """

    if not deltas:
        return pd.DataFrame(columns=["Variable", "Valeur", "Médiane", "Moyenne", "Écart", "Écart vs moyenne"]) 

    if money_features is None:
        money_features = {"budget", "revenue"}
    if int_features is None:
        int_features = set()

    ddf = pd.DataFrame(deltas)
    if ddf.empty:
        return pd.DataFrame(columns=["Variable", "Valeur", "Médiane", "Moyenne", "Écart", "Écart vs moyenne"]) 

    means: dict[str, float] = {}
    try:
        if stats_df is not None and (not stats_df.empty):
            for f in ddf.get("feature", pd.Series(dtype=object)).astype(str).unique().tolist():
                if isinstance(f, str) and f in stats_df.columns:
                    means[f] = float(pd.to_numeric(stats_df[f], errors="coerce").mean())
    except Exception:
        means = {}

    def _fmt_value(feature: str, x: float) -> str:
        if feature in money_features:
            return _format_money(float(x))
        if (feature in int_features) or float(x).is_integer():
            return f"{int(x):,}".replace(",", " ")
        return f"{x:,.2f}".replace(",", " ").replace(".", ",")

    def _fmt_delta(feature: str, d: float) -> str:
        sign = "+" if d >= 0 else "−"
        if feature in money_features:
            return f"{sign}{_format_money(abs(float(d)))}"
        if (feature in int_features) or float(abs(d)).is_integer():
            return f"{sign}{int(abs(d)):,}".replace(",", " ")
        return f"{sign}{abs(d):,.2f}".replace(",", " ").replace(".", ",")

    try:
        ddf["mean"] = ddf["feature"].map(lambda f: means.get(str(f), float("nan")))

        # Align mean comparison with integer-like features
        def _aligned_mean(row: pd.Series) -> float:
            try:
                feat = str(row.get("feature"))
                m_raw = row.get("mean")
                if m_raw is None:
                    return float("nan")
                m = float(m_raw)
            except Exception:
                return float("nan")
            if not np.isfinite(m):
                return float("nan")
            if feat in int_features and feat not in money_features:
                return float(int(round(m)))
            return float(m)

        ddf["mean_aligned"] = ddf.apply(_aligned_mean, axis=1)
        ddf["mean_display"] = ddf.apply(
            lambda r: _fmt_value(str(r["feature"]), float(r["mean_aligned"]))
            if np.isfinite(float(r["mean_aligned"]))
            else "—",
            axis=1,
        )
        ddf["delta_mean"] = ddf.apply(
            lambda r: float(r["value"]) - float(r["mean_aligned"])
            if np.isfinite(float(r["mean_aligned"]))
            else float("nan"),
            axis=1,
        )
        ddf["delta_mean_display"] = ddf.apply(
            lambda r: _fmt_delta(str(r["feature"]), float(r["delta_mean"])) if np.isfinite(float(r["delta_mean"])) else "—",
            axis=1,
        )
    except Exception:
        ddf["mean_display"] = "—"
        ddf["delta_mean_display"] = "—"

    show = ddf[["label", "value_display", "median_display", "mean_display", "delta_display", "delta_mean_display"]].rename(
        columns={
            "label": "Variable",
            "value_display": "Valeur",
            "median_display": "Médiane",
            "mean_display": "Moyenne",
            "delta_display": "Écart",
            "delta_mean_display": "Écart vs moyenne",
        }
    )
    return show


def _local_feature_effects(
    model: Any,
    *,
    features: list[str],
    values: dict[str, object],
    used_flags: dict[str, bool],
    medians: dict[str, float],
    identity_cols: list[str],
    clip: tuple[float, float] = (0.0, 5.0),
    max_rows: int = 20,
) -> pd.DataFrame:
    """Compute a simple local explanation by feature neutralization.

    For each *used* feature, we replace it with a neutral value (median for numeric,
    ['__MISSING__'] for identity/list-like columns) and re-run the model.

    Returned columns: feature, base_pred, neutral_pred, delta (base - neutral).
    """

    def _predict(row_values: dict[str, object]) -> float | None:
        try:
            X = pd.DataFrame([row_values], columns=features)
            y = float(np.asarray(model.predict(X)).reshape(-1)[0])
            y = float(np.clip(y, float(clip[0]), float(clip[1])))
            return y
        except Exception:
            return None

    base_pred = _predict(values)
    if base_pred is None:
        return pd.DataFrame(columns=["feature", "base_pred", "neutral_pred", "delta"]) 

    rows: list[dict[str, object]] = []
    id_set = set([str(x) for x in (identity_cols or [])])

    for f in features:
        if not used_flags.get(f):
            continue

        neutral: object
        if f in id_set or isinstance(values.get(f), list):
            neutral = ["__MISSING__"]
        else:
            neutral = float(medians.get(f, 0.0))

        alt_values = dict(values)
        alt_values[f] = neutral
        neutral_pred = _predict(alt_values)
        if neutral_pred is None:
            continue

        delta = float(base_pred - neutral_pred)
        if not np.isfinite(delta):
            continue
        rows.append({"feature": str(f), "base_pred": float(base_pred), "neutral_pred": float(neutral_pred), "delta": delta})

    if not rows:
        return pd.DataFrame(columns=["feature", "base_pred", "neutral_pred", "delta"]) 

    df = pd.DataFrame(rows)
    df["abs_delta"] = df["delta"].abs()
    df = df.sort_values("abs_delta", ascending=False)
    df = df.drop(columns=["abs_delta"], errors="ignore")
    if int(max_rows) > 0:
        df = df.head(int(max_rows))
    return df


def _altair_local_effects_bar(
    effects_df: pd.DataFrame,
    *,
    feature_labels: dict[str, str] | None = None,
    width: int = 520,
    height: int = 360,
) -> Any:
    """Altair horizontal bar chart for local feature effects."""

    try:
        import altair as alt  # type: ignore
    except Exception:
        return None

    if effects_df is None or effects_df.empty:
        return None

    d = effects_df.copy()
    d["feature"] = d["feature"].astype(str)
    if isinstance(feature_labels, dict) and feature_labels:
        d["label"] = d["feature"].map(lambda x: feature_labels.get(str(x), str(x)))
    else:
        d["label"] = d["feature"]
    d["abs_delta"] = d["delta"].abs()

    chart = (
        alt.Chart(d)
        .mark_bar()
        .encode(
            y=alt.Y("label:N", sort=alt.SortField(field="abs_delta", order="descending"), title=""),
            x=alt.X("delta:Q", title="Effet local sur la note (Δ)", axis=alt.Axis(format=".2f")),
            tooltip=[
                alt.Tooltip("label:N", title="Variable"),
                alt.Tooltip("delta:Q", title="Δ", format=".3f"),
                alt.Tooltip("base_pred:Q", title="Prédiction", format=".3f"),
                alt.Tooltip("neutral_pred:Q", title="Sans cette info", format=".3f"),
            ],
        )
        .properties(width=width, height=height)
    )
    rule = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(opacity=0.6).encode(x="x:Q")
    return alt.layer(chart, rule)


def _numeric_correlations(
    df: pd.DataFrame,
    *,
    features: list[str],
    target: str = "rating",
    min_n: int = 30,
    clip_quantiles: tuple[float, float] | None = None,
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
        if clip_quantiles is not None:
            try:
                ql, qh = clip_quantiles
                lo = float(pd.Series(x[mask]).quantile(float(ql)))
                hi = float(pd.Series(x[mask]).quantile(float(qh)))
                mask = mask & (x >= lo) & (x <= hi)
            except Exception:
                pass
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


def _bootstrap_mean_ci(values: np.ndarray | pd.Series, *, n_boot: int = 1000, alpha: float = 0.05):
    """Compute bootstrap CI for the mean. Returns (mean, lo, hi).

    Values must be a 1-D numeric array or Series (NaNs dropped).
    """
    try:
        arr = np.asarray(values)
    except Exception:
        return float("nan"), float("nan"), float("nan")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    # For very small samples, reduce bootstrap repetitions to avoid heavy CPU
    B = int(max(200, min(int(n_boot), 5000)))
    boots = []
    for _ in range(B):
        sample = np.random.choice(arr, size=arr.size, replace=True)
        boots.append(float(np.mean(sample)))
    lo = float(np.percentile(boots, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boots, 100.0 * (1.0 - alpha / 2.0)))
    return mean, lo, hi


def _altair_token_effect_chart(
    df: pd.DataFrame,
    *,
    token_col: str,
    token_name: str,
    target: str = "rating",
    title_col: str | None = "title",
    width: int = 640,
    height: int = 320,
    n_boot: int = 1000,
):
    """Create an Altair strip/points chart comparing films with vs without a token.

    `token_col` is expected to contain a list of names (e.g. casting/directors) or a single string.
    Returns (chart, stats_dict).
    """

    try:
        import altair as alt  # type: ignore
    except Exception:
        return None, {}

    if df is None or df.empty or target not in df.columns:
        return None, {}

    name = str(token_name or "").strip()
    if not name:
        return None, {}

    def _has_token(cell: object, wanted: str) -> bool:
        if isinstance(cell, list):
            return any(str(x).strip() == wanted for x in cell)
        if isinstance(cell, str):
            return str(cell).strip() == wanted
        return False

    series = df.get(token_col, pd.Series(dtype=object))
    mask_has = series.apply(lambda c: _has_token(c, name))

    df_yes = df.loc[mask_has].copy()
    df_no = df.loc[~mask_has].copy()

    vals_yes = pd.to_numeric(df_yes[target], errors="coerce").dropna().to_numpy(dtype=float)
    vals_no = pd.to_numeric(df_no[target], errors="coerce").dropna().to_numpy(dtype=float)

    n_yes = int(len(vals_yes))
    n_no = int(len(vals_no))
    mean_yes = float(np.mean(vals_yes)) if n_yes > 0 else float("nan")
    mean_no = float(np.mean(vals_no)) if n_no > 0 else float("nan")
    diff = float(mean_yes - mean_no) if (np.isfinite(mean_yes) and np.isfinite(mean_no)) else float("nan")

    ci_yes = _bootstrap_mean_ci(vals_yes, n_boot=n_boot) if n_yes > 0 else (float("nan"), float("nan"), float("nan"))

    overall_sd = float(pd.to_numeric(df[target], errors="coerce").std())
    cohen_d = float(diff / overall_sd) if (np.isfinite(diff) and np.isfinite(overall_sd) and overall_sd > 0) else float("nan")

    # prepare plotting DataFrame with jittered x coordinates and poster if available
    rows: list[dict[str, object]] = []
    rng = np.random.RandomState(42)

    for dsub, grp_x, grp_label in ((df_no, 0.0, "Sans"), (df_yes, 1.0, "Avec")):
        for _, row in dsub.iterrows():
            raw_r = row.get(target)
            if raw_r is None:
                continue
            try:
                r = float(raw_r)
            except Exception:
                continue
            if not np.isfinite(r):
                continue
            title = str(row.get(title_col)) if title_col and title_col in row.index else ""
            poster = None
            for c in ("poster_url", "poster", "poster_path", "poster_url_large", "poster_url_small"):
                if c in row.index and row.get(c):
                    poster = row.get(c)
                    break
            jitter = float(rng.normal(loc=0.0, scale=0.06))
            rows.append({"x": grp_x + jitter, "group": grp_label, "rating": r, "title": title, "poster": poster})

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return None, {"n_yes": n_yes, "n_no": n_no, "mean_yes": mean_yes, "mean_no": mean_no}

    # CI/mean rows for plotting
    ci_rows = [
        {"x": 0.0, "group": "Sans", "mean": mean_no, "lo": float("nan"), "hi": float("nan")},
        {"x": 1.0, "group": "Avec", "mean": mean_yes, "lo": float(ci_yes[1]), "hi": float(ci_yes[2])},
    ]
    ci_df = pd.DataFrame(ci_rows)

    base = alt.Chart(plot_df).properties(width=width, height=height)

    pts = base.mark_circle(size=32, opacity=0.6).encode(
        x=alt.X("x:Q", title=""),
        y=alt.Y("rating:Q", title="Note (rating)"),
        color=alt.Color("group:N", legend=None),
        tooltip=[
            alt.Tooltip("title:N", title="Titre"),
            alt.Tooltip("group:N", title="Groupe"),
            alt.Tooltip("rating:Q", title="Note"),
            alt.Tooltip("poster:N", title="Poster URL"),
        ],
    )

    if "poster" in plot_df.columns and plot_df["poster"].notna().any():
        try:
            img = base.mark_image(width=36, height=48, opacity=0.95).encode(
                x=alt.X("x:Q"),
                y=alt.Y("rating:Q"),
                url=alt.Url("poster:N"),
            )
            pts = alt.layer(img, pts)
        except Exception:
            pass

    mean_pts = alt.Chart(ci_df).mark_point(size=120, filled=True, shape="square").encode(
        x="x:Q",
        y=alt.Y("mean:Q", title=""),
        color=alt.Color("group:N", legend=None),
        tooltip=[alt.Tooltip("group:N", title="Groupe"), alt.Tooltip("mean:Q", title="Moyenne")],
    )

    error = alt.Chart(ci_df).mark_rule(strokeWidth=3, opacity=0.8).encode(
        x="x:Q",
        y=alt.Y("lo:Q", title=""),
        y2=alt.Y2("hi:Q"),
        color=alt.Color("group:N", legend=None),
    )

    if target == "rating":
        y_scale = alt.Scale(domain=[0, 5])
        pts = pts.encode(y=alt.Y("rating:Q", scale=y_scale, title="Note (rating)"))
        mean_pts = mean_pts.encode(y=alt.Y("mean:Q", scale=y_scale))
        error = error.encode(y=alt.Y("lo:Q", scale=y_scale), y2=alt.Y2("hi:Q"))

    layered = alt.layer(pts, mean_pts, error).configure_axis(grid=False)
    stats = {
        "token_col": str(token_col),
        "token_name": name,
        "n_yes": n_yes,
        "n_no": n_no,
        "mean_yes": mean_yes,
        "mean_no": mean_no,
        "diff": diff,
        "cohen_d": cohen_d,
        "ci_yes": ci_yes,
    }
    return layered, stats


def _altair_actor_effect_chart(
    df: pd.DataFrame,
    *,
    actor_name: str,
    target: str = "rating",
    title_col: str | None = "title",
    width: int = 640,
    height: int = 320,
    n_boot: int = 1000,
):
    """Create an Altair strip/points chart comparing films with vs without an actor.

    Shows raw points (with jitter), small poster images when available, the mean and bootstrap CI for the
    "Avec" group, and tooltips containing the film title and poster URL.
    Returns (chart, stats_dict) where stats_dict contains n_yes, n_no, mean_yes, mean_no, ci_yes.
    """
    return _altair_token_effect_chart(
        df,
        token_col="casting",
        token_name=str(actor_name),
        target=target,
        title_col=title_col,
        width=width,
        height=height,
        n_boot=n_boot,
    )


def _altair_scatter_with_regression(
    df: pd.DataFrame,
    *,
    feature: str,
    target: str = "rating",
    user_x: float | None = None,
    user_y: float | None = None,
    width: int = 520,
    height: int = 280,
    clip_quantiles: tuple[float, float] | None = None,
    transform_x: str | None = None,
    title_col: str | None = None,
):
    """Return an Altair chart: scatter + regression line + user marker/rule."""
    try:
        import altair as alt  # type: ignore
    except Exception:
        return None

    if df is None or df.empty or feature not in df.columns or target not in df.columns:
        return None

    cols = [feature, target]
    if title_col and title_col in df.columns:
        cols = [title_col] + cols
    d = df[cols].copy()
    d[feature] = pd.to_numeric(d[feature], errors="coerce")
    d[target] = pd.to_numeric(d[target], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna()

    if clip_quantiles is not None and not d.empty:
        try:
            ql, qh = clip_quantiles
            lo = float(d[feature].quantile(float(ql)))
            hi = float(d[feature].quantile(float(qh)))
            d = d[(d[feature] >= lo) & (d[feature] <= hi)]
        except Exception:
            pass

    x_field = feature
    x_title = feature
    if transform_x == "log1p":
        x_field = f"{feature}__log1p"
        try:
            d[x_field] = np.log1p(d[feature].astype(float))
        except Exception:
            d[x_field] = np.log1p(pd.to_numeric(d[feature], errors="coerce"))
        x_title = f"log(1+{feature})"

    if len(d) < 10:
        return None

    base = alt.Chart(d).properties(width=width, height=height)

    tooltip_fields = [alt.Tooltip(f"{feature}:Q", title=feature), alt.Tooltip(f"{target}:Q", title=target)]
    if title_col and title_col in d.columns:
        tooltip_fields.insert(0, alt.Tooltip(f"{title_col}:N", title="title"))
    # try to include a poster image if present
    poster_col = None
    for c in ("poster_url", "poster", "poster_path", "poster_url_large", "poster_url_small"):
        if c in d.columns:
            poster_col = c
            break
    if poster_col:
        # include poster URL in tooltip and add a small image layer
        tooltip_fields.insert(0, alt.Tooltip(f"{poster_col}:N", title="poster"))

    pts = base.mark_circle(size=22, opacity=0.18).encode(
        x=alt.X(f"{x_field}:Q", title=x_title),
        y=alt.Y(f"{target}:Q", title=target),
        tooltip=tooltip_fields,
    )
    # if poster exists, render tiny images behind points
    if poster_col:
        try:
            img = base.mark_image(width=36, height=48, opacity=0.95).encode(
                x=alt.X(f"{x_field}:Q"),
                y=alt.Y(f"{target}:Q"),
                url=alt.Url(f"{poster_col}:N"),
            )
            pts = alt.layer(img, pts)
        except Exception:
            pass
    reg = base.transform_regression(x_field, target).mark_line(opacity=0.8).encode(
        x=alt.X(f"{x_field}:Q"),
        y=alt.Y(f"{target}:Q"),
    )

    layers = [pts, reg]

    # enforce rating axis limits when target is rating
    if target == "rating":
        y_scale = alt.Scale(domain=[0, 5])
        pts = pts.encode(y=alt.Y(f"{target}:Q", scale=y_scale))
        reg = reg.encode(y=alt.Y(f"{target}:Q", scale=y_scale))

    if feature == "year":
        try:
            min_y = float(d[feature].min())
            max_y = float(d[feature].max())
            x_scale = alt.Scale(domain=[min_y - 5.0, max_y + 5.0])
            pts = pts.encode(x=alt.X(f"{x_field}:Q", title=x_title, scale=x_scale))
            reg = reg.encode(x=alt.X(f"{x_field}:Q", scale=x_scale))
        except Exception:
            pass

    if user_x is not None and np.isfinite(float(user_x)):
        ux = float(user_x)
        if transform_x == "log1p":
            try:
                ux = float(np.log1p(ux))
            except Exception:
                pass
        rule = alt.Chart(pd.DataFrame({"x": [ux]})).mark_rule(opacity=0.7).encode(x="x:Q")
        layers.append(rule)

        if user_y is not None and np.isfinite(float(user_y)):
            user_tooltip = [alt.Tooltip("x:Q", title=feature), alt.Tooltip("y:Q", title="note prédite")]
            if title_col and title_col in d.columns:
                user_tooltip.insert(0, alt.Tooltip(f"{title_col}:N", title="title"))
            user_pt = alt.Chart(pd.DataFrame({"x": [ux], "y": [float(user_y)]})).mark_point(size=120, filled=True).encode(
                x="x:Q",
                y="y:Q",
                tooltip=user_tooltip,
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
    clip_quantiles: tuple[float, float] | None = None,
    transform_x: str | None = None,
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

    if clip_quantiles is not None and not d.empty:
        try:
            ql, qh = clip_quantiles
            lo = float(d[feature].quantile(float(ql)))
            hi = float(d[feature].quantile(float(qh)))
            d = d[(d[feature] >= lo) & (d[feature] <= hi)]
        except Exception:
            pass

    x_field = feature
    x_title = feature
    if transform_x == "log1p":
        x_field = f"{feature}__log1p"
        try:
            d[x_field] = np.log1p(d[feature].astype(float))
        except Exception:
            d[x_field] = np.log1p(pd.to_numeric(d[feature], errors="coerce"))
        x_title = f"log(1+{feature})"

    if len(d) < 10:
        return None

    hist = alt.Chart(d).mark_bar(opacity=0.7).encode(
        x=alt.X(f"{x_field}:Q", bin=alt.Bin(maxbins=int(max_bins)), title=x_title),
        y=alt.Y("count():Q", title="N"),
        tooltip=[alt.Tooltip("count():Q", title="N")],
    ).properties(width=width, height=height)

    if user_x is not None and np.isfinite(float(user_x)):
        ux = float(user_x)
        if transform_x == "log1p":
            try:
                ux = float(np.log1p(ux))
            except Exception:
                pass
        rule = alt.Chart(pd.DataFrame({"x": [ux]})).mark_rule(opacity=0.8).encode(x="x:Q")
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
    load_test_rmse = staticmethod(_load_test_rmse)
    approx_prediction_interval_from_rmse = staticmethod(_approx_prediction_interval_from_rmse)
    ood_numeric_info = staticmethod(_ood_numeric_info)
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
    build_actor_photos_cache = staticmethod(_build_actor_photos_cache)
    flatten_unique_lists = staticmethod(_flatten_unique_lists)
    similar_movies = staticmethod(_similar_movies)
    similar_movies_with_explanations = staticmethod(_similar_movies_with_explanations)

    bootstrap_repo_path = staticmethod(_bootstrap_repo_path)
    apply_letterboxd_theme = staticmethod(_apply_letterboxd_theme)

    render_reference_film_hero = staticmethod(_render_reference_film_hero)

    suggestion_caption = staticmethod(_suggestion_caption)
    optional_slider = staticmethod(_optional_slider)
    optional_multiselect_count = staticmethod(_optional_multiselect_count)
    optional_multiselect_list = staticmethod(_optional_multiselect_list)
    prediction_quality_info = staticmethod(_prediction_quality_info)
    explain_numeric_deltas = staticmethod(_explain_numeric_deltas)
    numeric_deltas_table_with_mean = staticmethod(_numeric_deltas_table_with_mean)
    local_feature_effects = staticmethod(_local_feature_effects)
    altair_local_effects_bar = staticmethod(_altair_local_effects_bar)
    numeric_correlations = staticmethod(_numeric_correlations)
    altair_scatter_with_regression = staticmethod(_altair_scatter_with_regression)
    altair_histogram_with_rule = staticmethod(_altair_histogram_with_rule)
    bootstrap_mean_ci = staticmethod(_bootstrap_mean_ci)
    altair_token_effect_chart = staticmethod(_altair_token_effect_chart)
    altair_actor_effect_chart = staticmethod(_altair_actor_effect_chart)
