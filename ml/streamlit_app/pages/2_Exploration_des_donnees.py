from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


def _bootstrap_repo_path() -> Path:
    here = Path(__file__).resolve()
    repo_root = next(p for p in here.parents if p.name == "ml").parent
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
        </style>
        """,
        unsafe_allow_html=True,
    )


def _flatten_unique_lists(df: pd.DataFrame, col: str, top_k: int = 250) -> list[str]:
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


def _has_any_token(cell: object, tokens: set[str]) -> bool:
    if not tokens:
        return True
    if isinstance(cell, list):
        return any(str(x) in tokens for x in cell)
    if isinstance(cell, str):
        return cell in tokens
    return False


@st.cache_data
def _load_reference_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path)


_bootstrap_repo_path()

# Paths
here = Path(__file__).resolve()
ml_dir = next(p for p in here.parents if p.name == "ml")
repo_root = ml_dir.parent
ref_data_path = ml_dir / "data" / "final_results_28.json"
merged_results_path = repo_root / "merged_results.json"

st.set_page_config(page_title="Exploration des données — Letterboxd", layout="wide")
_apply_letterboxd_theme()

st.title("Exploration des données")

ref_df = _load_reference_df(ref_data_path)
if ref_df.empty:
    ref_df = _load_reference_df(merged_results_path)

if ref_df.empty:
    st.error("Impossible de charger les données (final_results_28.json / merged_results.json).")
    st.stop()

# Normalize common columns
for col in ["year", "duration", "budget", "revenue", "rating"]:
    if col in ref_df.columns:
        ref_df[col] = pd.to_numeric(ref_df[col], errors="coerce")

# Sidebar filters
st.sidebar.subheader("Filtres")

year_min = int(np.nanmin(ref_df["year"]) if "year" in ref_df.columns and ref_df["year"].notna().any() else 1900)
year_max = int(np.nanmax(ref_df["year"]) if "year" in ref_df.columns and ref_df["year"].notna().any() else 2025)
year_range = st.sidebar.slider("Année", min_value=year_min, max_value=year_max, value=(year_min, year_max))

rating_range = None
if "rating" in ref_df.columns and ref_df["rating"].notna().any():
    rmin = float(np.nanmin(ref_df["rating"]))
    rmax = float(np.nanmax(ref_df["rating"]))
    rating_range = st.sidebar.slider("Note (rating)", min_value=float(max(0.0, rmin)), max_value=float(min(5.0, rmax)), value=(0.0, 5.0))

# Categorical filters (list-like)
director_options = _flatten_unique_lists(ref_df, "directors", top_k=300)
theme_options = _flatten_unique_lists(ref_df, "themes", top_k=400)
genre_options = _flatten_unique_lists(ref_df, "genres", top_k=250)

selected_directors = st.sidebar.multiselect("Réalisateurs", options=director_options, default=[])
selected_themes = st.sidebar.multiselect("Thèmes", options=theme_options, default=[])
selected_genres = st.sidebar.multiselect("Genres", options=genre_options, default=[])

# Apply filters
filtered = ref_df.copy()
if "year" in filtered.columns:
    filtered = filtered[(filtered["year"] >= year_range[0]) & (filtered["year"] <= year_range[1])]

if rating_range is not None and "rating" in filtered.columns:
    filtered = filtered[(filtered["rating"].fillna(-1) >= rating_range[0]) & (filtered["rating"].fillna(-1) <= rating_range[1])]

if selected_directors and "directors" in filtered.columns:
    tokens = set(selected_directors)
    filtered = filtered[filtered["directors"].apply(lambda x: _has_any_token(x, tokens))]

if selected_themes and "themes" in filtered.columns:
    tokens = set(selected_themes)
    filtered = filtered[filtered["themes"].apply(lambda x: _has_any_token(x, tokens))]

if selected_genres and "genres" in filtered.columns:
    tokens = set(selected_genres)
    filtered = filtered[filtered["genres"].apply(lambda x: _has_any_token(x, tokens))]

st.caption(f"Films sélectionnés: {len(filtered)} / {len(ref_df)}")

# Summary metrics
cols = st.columns(4)
if "rating" in filtered.columns:
    cols[0].metric("Note moyenne", f"{filtered['rating'].mean(skipna=True):.2f}" if filtered['rating'].notna().any() else "—")
else:
    cols[0].metric("Note moyenne", "—")

if "budget" in filtered.columns:
    cols[1].metric("Budget médian", f"{filtered['budget'].median(skipna=True):,.0f}".replace(",", " ") if filtered['budget'].notna().any() else "—")
else:
    cols[1].metric("Budget médian", "—")

if "revenue" in filtered.columns:
    cols[2].metric("Revenu médian", f"{filtered['revenue'].median(skipna=True):,.0f}".replace(",", " ") if filtered['revenue'].notna().any() else "—")
else:
    cols[2].metric("Revenu médian", "—")

if "year" in filtered.columns:
    cols[3].metric("Année médiane", f"{int(filtered['year'].median(skipna=True))}" if filtered['year'].notna().any() else "—")
else:
    cols[3].metric("Année médiane", "—")

st.divider()

# Correlations and plots
st.subheader("Corrélations et graphiques")

numeric_candidates = [c for c in ["budget", "revenue", "duration", "year", "nbr_watched", "nbr_likes", "fans_favoris"] if c in filtered.columns]

if "rating" in filtered.columns and numeric_candidates:
    x_var = st.selectbox("Variable à comparer à la note", options=numeric_candidates, index=0)

    xy = filtered[[x_var, "rating"]].copy()
    xy = xy[np.isfinite(xy[x_var]) & np.isfinite(xy["rating"])].dropna()
    if len(xy) >= 3:
        pearson = float(xy[x_var].corr(xy["rating"], method="pearson"))
        spearman = float(xy[x_var].corr(xy["rating"], method="spearman"))
        st.caption(f"Corrélation Pearson: {pearson:.3f} | Spearman: {spearman:.3f}")
        st.scatter_chart(xy.rename(columns={x_var: "x", "rating": "y"}), x="x", y="y")
    else:
        st.info("Pas assez de données numériques valides pour calculer une corrélation.")
else:
    st.info("Corrélations indisponibles (colonnes manquantes).")

# Average ratings by genre
if "genres" in filtered.columns and "rating" in filtered.columns:
    st.subheader("Note moyenne par genre")
    tmp = filtered[["genres", "rating"]].dropna()
    tmp = tmp[tmp["genres"].apply(lambda x: isinstance(x, list))]
    if not tmp.empty:
        exploded = tmp.explode("genres")
        grp = exploded.groupby("genres", as_index=False)["rating"].mean().sort_values("rating", ascending=False).head(30)
        st.bar_chart(grp.set_index("genres"))
        
# Table + export
st.subheader("Table")
show_cols = [c for c in ["title", "year", "rating", "budget", "revenue", "duration", "directors", "genres", "themes"] if c in filtered.columns]
if not show_cols:
    show_cols = list(filtered.columns[:10])

st.dataframe(filtered[show_cols], width='stretch', height=420)

csv = filtered[show_cols].to_csv(index=False).encode("utf-8")
st.download_button("Télécharger CSV (sélection)", data=csv, file_name="letterboxd_selection.csv", mime="text/csv")
