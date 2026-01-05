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


def _sanitize_widget_suffix(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "manual"
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", s)
    return s[:80] if len(s) > 80 else s


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


def _tmdb_api_key() -> str | None:
    # Prefer Streamlit secrets, then environment variable
    try:
        key = st.secrets.get("TMDB_API_KEY")  # type: ignore[attr-defined]
        if isinstance(key, str) and key.strip():
            return key.strip()
    except Exception:
        pass
    key = os.environ.get("TMDB_API_KEY")
    return key.strip() if isinstance(key, str) and key.strip() else None


def _extract_tmdb_movie_id(letterboxd_html: str) -> int | None:
    # Common patterns on Letterboxd pages
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
    for row in cast:
        if not isinstance(row, dict):
            continue
        profile_path = row.get("profile_path")
        if not isinstance(profile_path, str) or not profile_path.strip():
            continue
        name = row.get("name")
        character = row.get("character")
        out.append(
            {
                "name": str(name) if name is not None else "",
                "character": str(character) if character is not None else "",
                "profile_url": _tmdb_profile_url(profile_path.strip(), size="w185"),
            }
        )
        if len(out) >= int(top_n):
            break
    return out


def _extract_letterboxd_poster_url(html: str) -> str | None:
    # Letterboxd meta tags can sometimes point to the page header/backdrop.
    # We want the actual poster (usually a.ltrbxd.com/resized/film-poster/...).

    _CROP_DIMS_RE = re.compile(r"-(\d+)-(\d+)-(\d+)-(\d+)-crop", flags=re.IGNORECASE)

    def _looks_like_poster_url(url: str) -> bool:
        u = (url or "").strip().lower()
        if not u.startswith("http"):
            return False
        if "empty-poster" in u:
            return False
        # Real poster/CDN URLs are served from a.ltrbxd.com; placeholders/icons often aren't.
        if "a.ltrbxd.com" not in u:
            return False

        m = _CROP_DIMS_RE.search(u)
        if not m:
            # If we can't infer dimensions, be conservative.
            return "film-poster" in u

        try:
            a, b, c, d = (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
        except Exception:
            return False

        width = max(a, b)
        height = max(c, d)
        if width <= 0 or height <= 0:
            return False

        # Reject tiny images (flags/icons)
        if width < 150 or height < 200:
            return False

        # Posters are portrait (~2:3 => height/width ~ 1.5). Backdrops are landscape (~16:9).
        if height <= width:
            return False
        ratio = height / width
        return 1.25 <= ratio <= 1.85

    def _img_tag_to_url(tag: str) -> str | None:
        # Prefer the largest in srcset (usually last)
        m_srcset = re.search(r"\bsrcset=\"([^\"]+)\"", tag, flags=re.IGNORECASE)
        if m_srcset:
            parts = [p.strip() for p in m_srcset.group(1).split(",") if p.strip()]
            if parts:
                last = parts[-1]
                last_url = last.split(" ")[0].strip()
                if last_url.startswith("http"):
                    if "empty-poster" in last_url:
                        return None
                    return last_url

        m_src = re.search(r"\bsrc=\"([^\"]+)\"", tag, flags=re.IGNORECASE)
        if m_src:
            u = m_src.group(1).strip()
            if u.startswith("http"):
                if "empty-poster" in u:
                    return None
                return u
        return None

    def _pick_best(urls: list[str]) -> str | None:
        urls = [u.strip() for u in urls if isinstance(u, str) and u.strip()]
        urls = [u for u in urls if u.startswith("http")]
        urls = [u for u in urls if "empty-poster" not in u]
        if not urls:
            return None

        # Filter to plausible poster URLs (avoids backdrops and small icons)
        urls = [u for u in urls if _looks_like_poster_url(u)]
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

        # Otherwise, accept the best poster-like URL (typically /resized/sm/upload/...)
        for u in urls:
            if "-230-" in u and "-345-" in u:
                return u
        return urls[0]

    def _extract_candidates_from_text(text: str) -> list[str]:
        # Capture both classic film-poster URLs and sm/upload poster crops.
        return re.findall(
            r"https?://a\.ltrbxd\.com/resized/(?:film-poster|sm/upload)/[^\"'\s<>]+",
            text,
            flags=re.IGNORECASE,
        )

    # 0) Best: extract poster from the actual poster column (never the backdrop)
    m_poster_col = re.search(r"\bid\s*=\s*(['\"])js-poster-col\1", html, flags=re.IGNORECASE)
    if m_poster_col:
        start = m_poster_col.start()
        # The HTML can be large; keep a generous window to include the poster markup.
        window = html[start : start + 150000]
        # Prefer <img alt="Poster for ..."> if present in that window
        poster_col_candidates: list[str] = []
        for img in re.finditer(r"<img\b[^>]*>", window, flags=re.IGNORECASE | re.DOTALL):
            tag = img.group(0)
            if re.search(r"\balt=\"Poster\s+for\b", tag, flags=re.IGNORECASE):
                u = _img_tag_to_url(tag)
                if u:
                    poster_col_candidates.append(u)
        best = _pick_best(poster_col_candidates)
        if best:
            return best
        # Fallback: any image inside poster col
        poster_col_candidates = []
        for img in re.finditer(r"<img\b[^>]*>", window, flags=re.IGNORECASE | re.DOTALL):
            u = _img_tag_to_url(img.group(0))
            if u:
                poster_col_candidates.append(u)
        best = _pick_best(poster_col_candidates)
        if best:
            return best

        # Still nothing? Directly scan this region for crop URLs.
        best = _pick_best(_extract_candidates_from_text(window))
        if best:
            return best

    # 1) Strongest remaining: direct crop URLs anywhere in the HTML
    best = _pick_best(_extract_candidates_from_text(html))
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
def _get_letterboxd_poster_url(letterboxd_url: str, _cache_bust: str = "v6") -> str | None:
    try:
        html = _fetch_html(letterboxd_url)
        poster = _extract_letterboxd_poster_url(html)
        if poster:
            return poster

        def _looks_like_poster_final_url(url: str) -> bool:
            u = (url or "").strip().lower()
            if not u.startswith("http"):
                return False
            if "empty-poster" in u:
                return False
            if "a.ltrbxd.com" not in u:
                return False
            if "backdrop" in u:
                return False

            m = re.search(r"-(\d+)-(\d+)-(\d+)-(\d+)-crop", u)
            if not m:
                return "film-poster" in u
            try:
                a, b, c, d = (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))
            except Exception:
                return False
            width = max(a, b)
            height = max(c, d)
            if width < 150 or height < 200:
                return False
            if height <= width:
                return False
            ratio = height / width
            return 1.25 <= ratio <= 1.85

        def _pick_best_candidate(urls: list[str]) -> str | None:
            urls = [u.strip() for u in urls if isinstance(u, str) and u.strip()]
            urls = [u for u in urls if _looks_like_poster_final_url(u)]
            if not urls:
                return None
            for u in urls:
                if "-0-230-0-345-crop" in u:
                    return u
            return urls[0]

        def _extract_candidate_urls(text: str) -> list[str]:
            return re.findall(
                r"https?://a\.ltrbxd\.com/resized/(?:film-poster|sm/upload)/[^\"'\s<>]+",
                text,
                flags=re.IGNORECASE,
            )

        # Fallback A: use the JSON details endpoint embedded in the HTML.
        # This is often present even when the poster <img> is JS-injected.
        details_endpoints: list[str] = []
        for m in re.finditer(r"data-details-endpoint\s*=\s*(['\"])(.*?)\1", html, flags=re.IGNORECASE | re.DOTALL):
            p = (m.group(2) or "").strip()
            if p:
                details_endpoints.append(p)
        # If not present, try the conventional /json/ endpoint.
        if not details_endpoints:
            details_endpoints.append("json/")

        for ep in details_endpoints[:2]:
            try:
                details_url = urllib.parse.urljoin(letterboxd_url, ep)
                req = urllib.request.Request(
                    details_url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                        "Accept": "application/json,text/javascript,*/*;q=0.8",
                        "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
                        "X-Requested-With": "XMLHttpRequest",
                    },
                    method="GET",
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    raw = resp.read()
                text = raw.decode("utf-8", errors="ignore")
                best = _pick_best_candidate(_extract_candidate_urls(text))
                if best:
                    return best
            except Exception:
                continue

        # If the poster is loaded dynamically, the HTML often contains a resolver endpoint like:
        # data-poster-url="/film/<slug>/image-150/" (which may redirect to the real image).
        poster_paths: list[str] = []
        for m in re.finditer(r"data-poster-url\s*=\s*(['\"])(.*?)\1", html, flags=re.IGNORECASE | re.DOTALL):
            p = (m.group(2) or "").strip()
            if not p:
                continue
            if "empty-poster" in p:
                continue
            poster_paths.append(p)

        for p in poster_paths[:3]:
            try:
                resolved_url = urllib.parse.urljoin(letterboxd_url, p)
                req = urllib.request.Request(
                    resolved_url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                        "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
                    },
                    method="GET",
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    final_url = resp.geturl()
                    content_type = (resp.headers.get("Content-Type") or "").lower()
                    raw = resp.read()

                # If we landed on an actual image, the final URL is the one we want.
                if final_url.startswith("http") and (
                    content_type.startswith("image/")
                    or ".jpg" in final_url.lower()
                    or ".jpeg" in final_url.lower()
                    or ".png" in final_url.lower()
                ):
                    # Only accept real posters (avoid flags/icons/backdrops)
                    if _looks_like_poster_final_url(final_url):
                        return final_url

                # Otherwise, try to extract a film-poster URL from the returned HTML.
                try:
                    text = raw.decode("utf-8", errors="ignore")
                except Exception:
                    text = ""
                poster2 = _extract_letterboxd_poster_url(text)
                if poster2:
                    return poster2
            except Exception:
                continue

        return None
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError):
        return None


@st.cache_data
def _load_train_features(path: Path, features: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    requested = [str(c) for c in features if c is not None and str(c).strip()]
    requested = list(dict.fromkeys(requested))  # stable de-dup

    # Prefer column-projection when possible to avoid loading the whole parquet.
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
    # Fallback only (some environments may not ship final_results_28.json at runtime).
    ref_df = _load_reference_df(MERGED_RESULTS_PATH)

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
train_df = _load_train_features(TRAIN_PATH, train_cols)
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
                
            # Option A — TMDB: actor headshots (requires TMDB_API_KEY)
            api_key = _tmdb_api_key()
            if api_key:
                try:
                    lb_html = _fetch_html(lb_url)
                    tmdb_id = _extract_tmdb_movie_id(lb_html)
                except Exception:
                    tmdb_id = None

                ref_tmdb_id = tmdb_id

                if tmdb_id:
                    cast = _tmdb_movie_cast(tmdb_id, top_n=3)
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
WIDGET_STATE_SUFFIX = _sanitize_widget_suffix(_ref_token)


# Note: train.parquet contains engineered "*_count" features, not the original name lists.
# So we build the option lists from the reference JSON.
names_df = ref_df

name_options = {
    # top_k=0 => "all" (users expect to be able to type-search any known entry).
    "directors": _flatten_unique_lists(names_df, "directors", top_k=0),
    "casting": _flatten_unique_lists(names_df, "casting", top_k=0),
    "producers": _flatten_unique_lists(names_df, "producers", top_k=0),
    "writers": _flatten_unique_lists(names_df, "writers", top_k=0),
    "composer": _flatten_unique_lists(names_df, "composer", top_k=0),
    "studio": _flatten_unique_lists(names_df, "studio", top_k=0),
    "languages": _flatten_unique_lists(names_df, "languages", top_k=0),
    "genres": _flatten_unique_lists(names_df, "genres", top_k=0),
    "themes": _flatten_unique_lists(names_df, "themes", top_k=0),
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
    state_suffix = globals().get("WIDGET_STATE_SUFFIX", "manual")
    use_key = f"use_{feature}_{state_suffix}"
    val_key = f"val_{feature}_{state_suffix}"

    pre_has = bool(preset_values and feature in preset_values)

    prefill_value = None
    prefill_used = False
    if bool(globals().get("prefill_from_ref")) and isinstance(globals().get("ref_row"), dict):
        rr = globals().get("ref_row")
        if isinstance(rr, dict) and feature in rr and rr.get(feature) is not None:
            prefill_value = rr.get(feature)
            prefill_used = True

    use = st.checkbox(f"Utiliser {label}", value=(prefill_used or pre_has), key=use_key)
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

    # Clamp to slider range to avoid StreamlitValueAboveMaxError
    if np.isfinite(default):
        default = float(min(max(default, float(lo)), float(hi)))

    if is_int:
        val = st.slider(label, int(lo), int(hi), int(round(default)), key=val_key)
        return True, float(val)

    val = st.slider(label, float(lo), float(hi), float(default), key=val_key, format="%.2f")
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
    state_suffix = globals().get("WIDGET_STATE_SUFFIX", "manual")
    use_key = f"use_{feature}_{state_suffix}"
    val_key = f"val_{feature}_{state_suffix}"

    prefill_list: list[str] = []
    prefill_used = False
    if bool(globals().get("prefill_from_ref")) and isinstance(globals().get("ref_row"), dict):
        rr = globals().get("ref_row")
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

    # TMDB-enhanced UX for people-like lists: show small photos for selected names.
    if selected and _tmdb_api_key() and ref_col in {"directors", "casting", "producers", "writers", "composer"}:
        known: dict[str, str] = {}
        # For casting, prefer exact photos from the reference movie cast when available.
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
    *,
    feature: str,
    label: str,
    ref_col: str,
    preset_values: dict | None = None,
) -> tuple[bool, list[str]]:
    state_suffix = globals().get("WIDGET_STATE_SUFFIX", "manual")
    use_key = f"use_{feature}_{state_suffix}"
    val_key = f"val_{feature}_{state_suffix}"

    prefill_list: list[str] = []
    prefill_used = False
    if bool(globals().get("prefill_from_ref")) and isinstance(globals().get("ref_row"), dict):
        rr = globals().get("ref_row")
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

    # TMDB-enhanced UX for people-like lists: show small photos for selected names.
    if selected and _tmdb_api_key() and ref_col in {"directors", "casting", "producers", "writers", "composer"}:
        known: dict[str, str] = {}
        # For casting, prefer exact photos from the reference movie cast when available.
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
            u, v = _optional_multiselect_list(
                feature=feature,
                label=label,
                ref_col=ref_col,
                preset_values=preset_values,
            )
        else:
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
