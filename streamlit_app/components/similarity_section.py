from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def render_similarity_section(
    *,
    H: Any,
    ref_df: pd.DataFrame,
    last_values: dict[str, object],
    numeric_cols: list[str],
    identity_cols: list[str],
    y_pred: float,
    ref_idx: int | None,
    ref_row: dict[str, Any] | None,
    top_n: int = 4,
) -> None:
    """Rendu Streamlit de la section Similarités.

    Objectif: éviter d'alourdir la page principale; toute la logique d'affichage
    (tabs, posters, tableau détaillé, exclusions) vit ici.
    """

    with st.expander("Similarités (top 3)", expanded=False):
        values_with_rating = dict(last_values)
        values_with_rating["rating"] = y_pred

        sims = H.similar_movies_with_explanations(ref_df, values_with_rating, numeric_cols, identity_cols, top_n=top_n)

        # Exclure le film de référence des résultats
        if ref_idx is not None:
            sims = [s for s in sims if s.get("idx") != ref_idx]
        elif ref_row is not None:
            ref_title = str(ref_row.get("title") or "").strip().lower()
            ref_year = str(ref_row.get("year") or "").strip()
            sims = [
                s
                for s in sims
                if not (
                    str(s.get("title") or "").strip().lower() == ref_title
                    and str(s.get("year") or "").strip() == ref_year
                )
            ]

        if not sims:
            st.info("Aucune similarité calculable (données insuffisantes).")
            return

        tab_visual, tab_details = st.tabs(["🎬 Vue visuelle", "📋 Détails complets"])

        with tab_visual:
            st.markdown("##### Aperçu visuel des films similaires")
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
                            st.image(poster, width=160, caption="Affiche")
                        except Exception:
                            pass
                    if url:
                        st.link_button("Fiche Letterboxd", url, use_container_width=True)

        with tab_details:
            st.markdown("##### Films similaires à votre saisie")

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

                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.markdown(f"### #{i+1} — {label}")
                    with col2:
                        st.metric("Similarité", f"{sim_pct}%")
                    with col3:
                        if url:
                            st.link_button("Letterboxd", url)

                    if feats:
                        st.markdown("##### Comparaison détaillée")
                        table_data: list[dict[str, str]] = []
                        for f in feats:
                            name = str(f.get("name") or "")
                            typ = str(f.get("type") or "")
                            sim_val = f.get("similarity")
                            det = f.get("details") or {}
                            status = f.get("status", "unknown")

                            if status == "missing":
                                user_value = "Manquant"
                                film_value = "Manquant"
                            elif typ == "numeric":
                                user_value = det.get("user")
                                film_value = det.get("ref")
                            else:
                                user_count = det.get("user_count", 0)
                                ref_count = det.get("ref_count", 0)
                                user_value = f"{user_count} item(s)"
                                film_value = f"{ref_count} item(s)"

                            diff_display = "—"
                            if status == "ok" and typ == "numeric":
                                user_v = det.get("user")
                                ref_v = det.get("ref")
                                if user_v is not None and ref_v is not None and ref_v != 0:
                                    diff_pct = abs(user_v - ref_v) / abs(ref_v) * 100
                                    diff_display = f"{diff_pct:.1f}%"
                            elif status == "ok" and typ == "identity" and isinstance(sim_val, (int, float)):
                                diff_display = f"{100 - float(sim_val):.1f}%"

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

                            table_data.append(
                                {
                                    "Critère": name,
                                    "Votre valeur": str(user_value) if user_value is not None else "—",
                                    "Valeur film": str(film_value) if film_value is not None else "—",
                                    "Similarité": status_display,
                                    "Différence (%)": diff_display,
                                }
                            )

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
