from __future__ import annotations

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Bootstrap paths so `ml.*` is importable under Streamlit.
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]  # project root
ML_DIR = REPO_ROOT / "ml"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from streamlit_app.helpers import Helpers as H

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


H.bootstrap_repo_path()

# Paths
ref_data_path = ML_DIR / "data" / "final_results_28.json"
merged_results_path = REPO_ROOT / "merged_results.json"

st.set_page_config(page_title="Exploration des données — Letterboxd", layout="wide")
H.apply_letterboxd_theme()

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
# Build initial options from the dataset already constrained by year/rating
pre_filtered = ref_df.copy()
if "year" in pre_filtered.columns:
    pre_filtered = pre_filtered[(pre_filtered["year"] >= year_range[0]) & (pre_filtered["year"] <= year_range[1])]
if rating_range is not None and "rating" in pre_filtered.columns:
    pre_filtered = pre_filtered[(pre_filtered["rating"].fillna(-1) >= rating_range[0]) & (pre_filtered["rating"].fillna(-1) <= rating_range[1])]

# If an actor was already chosen in the session state, narrow the primary options
actor_state = st.session_state.get("Acteurs")
if actor_state and actor_state != "Aucun" and "casting" in pre_filtered.columns:
    _mask_actor = pre_filtered[pre_filtered["casting"].apply(lambda c: isinstance(c, list) and str(actor_state) in [str(x) for x in c])]
    director_options = _flatten_unique_lists(_mask_actor, "directors", top_k=300)
    theme_options = _flatten_unique_lists(_mask_actor, "themes", top_k=400)
    genre_options = _flatten_unique_lists(_mask_actor, "genres", top_k=250)
else:
    director_options = _flatten_unique_lists(pre_filtered, "directors", top_k=300)
    theme_options = _flatten_unique_lists(pre_filtered, "themes", top_k=400)
    genre_options = _flatten_unique_lists(pre_filtered, "genres", top_k=250)

# Primary selectors
selected_directors = st.sidebar.multiselect("Réalisateurs", options=director_options, default=[])
selected_themes = st.sidebar.multiselect("Thèmes", options=theme_options, default=[])
selected_genres = st.sidebar.multiselect("Genres", options=genre_options, default=[])

# Narrow actor options to values present after applying primary selectors
mask_for_actors = pre_filtered.copy()
if selected_directors and "directors" in mask_for_actors.columns:
    tokens = set(selected_directors)
    mask_for_actors = mask_for_actors[mask_for_actors["directors"].apply(lambda x: _has_any_token(x, tokens))]
if selected_themes and "themes" in mask_for_actors.columns:
    tokens = set(selected_themes)
    mask_for_actors = mask_for_actors[mask_for_actors["themes"].apply(lambda x: _has_any_token(x, tokens))]
if selected_genres and "genres" in mask_for_actors.columns:
    tokens = set(selected_genres)
    mask_for_actors = mask_for_actors[mask_for_actors["genres"].apply(lambda x: _has_any_token(x, tokens))]

casting_options = _flatten_unique_lists(mask_for_actors, "casting", top_k=600)

# Actor selector placed under "Réalisateurs" as requested; remove the "(filtre)" suffix
selected_actor = None
if casting_options:
    selected_actor = st.sidebar.selectbox("Acteurs", options=["Aucun"] + casting_options, index=0)

# (Réalisateurs associés will be computed after building the final `filtered` set)

# Apply filters
filtered = ref_df.copy()
if "year" in filtered.columns:
    filtered = filtered[(filtered["year"] >= year_range[0]) & (filtered["year"] <= year_range[1])]

if rating_range is not None and "rating" in filtered.columns:
    filtered = filtered[(filtered["rating"].fillna(-1) >= rating_range[0]) & (filtered["rating"].fillna(-1) <= rating_range[1])]

if selected_directors and "directors" in filtered.columns:
    tokens = set(selected_directors)
    filtered = filtered[filtered["directors"].apply(lambda x: _has_any_token(x, tokens))]

if selected_actor and selected_actor != "Aucun" and "casting" in filtered.columns:
    # filter to films containing the selected actor
    filtered = filtered[filtered["casting"].apply(lambda c: isinstance(c, list) and (str(selected_actor) in [str(x) for x in c]))]

# (No associated-directors selector — removed per user request)

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
        chart = H.altair_scatter_with_regression(filtered, feature=str(x_var), target="rating", title_col="title")
        if chart is None:
            st.scatter_chart(xy.rename(columns={x_var: "x", "rating": "y"}), x="x", y="y")
        else:
            st.altair_chart(chart)
    else:
        st.info("Pas assez de données numériques valides pour calculer une corrélation.")
else:
    st.info("Corrélations indisponibles (colonnes manquantes).")

# Actor impact visualization (bootstrap + raw points) — actor chosen via sidebar
if "casting" in filtered.columns and selected_actor and selected_actor != "Aucun":
    chosen_actor = selected_actor
    try:
        chart, stats = H.altair_actor_effect_chart(
            filtered,
            actor_name=str(chosen_actor),
            target="rating",
            title_col="title",
            width=700,
            height=320,
            n_boot=1000,
        )
        if chart is None:
            st.info("Visualisation indisponible (Altair manquant ou pas assez de données).")
        else:
            st.altair_chart(chart)

        if stats:
            n_yes = int(stats.get("n_yes", 0))
            n_no = int(stats.get("n_no", 0))
            mean_yes = stats.get("mean_yes")
            mean_no = stats.get("mean_no")
            ci = stats.get("ci_yes")
            st.write(f"Films avec l'acteur: **{n_yes}**, sans: **{n_no}**")
            if n_yes > 0:
                if isinstance(ci, (list, tuple)) and len(ci) >= 3 and np.isfinite(ci[1]) and np.isfinite(ci[2]):
                    st.write(f"Moyenne (avec): **{mean_yes:.2f}** — IC bootstrap 95%: **{ci[1]:.2f} – {ci[2]:.2f}**")
                else:
                    st.write(f"Moyenne (avec): **{mean_yes:.2f}**")
            if n_no > 0:
                st.write(f"Moyenne (sans): **{mean_no:.2f}**")

            try:
                df_yes_titles = filtered[filtered.get("casting", pd.Series(dtype=object)).apply(lambda c: isinstance(c, list) and str(chosen_actor) in [str(x) for x in c])]["title"]
                if not df_yes_titles.empty:
                    st.caption("Exemples de films avec l'acteur (échantillon):")
                    st.write(list(df_yes_titles.head(12).astype(str).tolist()))
            except Exception:
                pass
    except Exception as e:
        st.warning(f"Impossible de calculer l'impact de l'acteur: {e}")

# Average ratings by genre
if "genres" in filtered.columns and "rating" in filtered.columns:
    st.subheader("Note moyenne par genre")
    tmp = filtered[["genres", "rating"]].dropna()
    tmp = tmp[tmp["genres"].apply(lambda x: isinstance(x, list))]
    if not tmp.empty:
        exploded = tmp.explode("genres")
        grp = exploded.groupby("genres", as_index=False)["rating"].mean()
        try:
            s = grp["rating"].sort_values(ascending=False).head(30)
            grp = grp.loc[s.index]
        except Exception:
            grp = grp.head(30)
        st.bar_chart(grp.set_index("genres"))
        
# Table + export
st.subheader("Table")
show_cols = [c for c in ["title", "year", "rating", "budget", "revenue", "duration", "directors", "genres", "themes"] if c in filtered.columns]
if not show_cols:
    show_cols = list(filtered.columns[:10])

st.dataframe(filtered[show_cols], width='stretch', height=420)

csv = filtered[show_cols].to_csv(index=False).encode("utf-8")
st.download_button("Télécharger CSV (sélection)", data=csv, file_name="letterboxd_selection.csv", mime="text/csv")
