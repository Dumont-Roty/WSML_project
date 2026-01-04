from __future__ import annotations

import sys
from pathlib import Path

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


_bootstrap_repo_path()

st.set_page_config(page_title="Présentation — Estimateur Letterboxd", layout="wide")
_apply_letterboxd_theme()

st.title("Présentation")

st.markdown(
    """
Ce projet vise à **estimer la note Letterboxd (0–5)** d’un film à partir de caractéristiques (année, durée, budget, revenu…) et d’informations d’**identité** (réalisateurs, casting, thèmes…).

- La **note prédite** est l’objectif principal.
- La **suggestion de budget** est une **aide** pour contextualiser l’estimation (indication, pas une vérité).

Utilise la navigation Streamlit (menu à gauche) pour accéder aux pages :
- **Estimateur Letterboxd** : saisie des paramètres + prédictions
- **Exploration des données** : filtres, stats, corrélations, moyennes par thème/réal…
- **Outils / Jeu** : (à venir)
- **À propos** : méthodologie, choix de modèles, limites
"""
)

with st.expander("Données et limites"):
    st.markdown(
        """
- Les données sont issues de scraping / agrégation et peuvent contenir des biais, valeurs manquantes ou incohérences.
- Les prédictions sont **indicatives** et dépendent de la couverture et de la qualité des données.
"""
    )
