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

st.set_page_config(page_title="À propos — Estimateur Letterboxd", layout="wide")
_apply_letterboxd_theme()

st.title("À propos")

st.markdown(
    """
## Objectif
Estimer la **note Letterboxd (0–5)** d’un film, en s’appuyant sur des attributs numériques et des informations d’identités (réalisateurs, casting, thèmes…).

## Données
Les données sont issues d’un pipeline de scraping/agrégation (selon le dataset présent dans `ml/data/`). Elles peuvent contenir : valeurs manquantes, erreurs de saisie, biais de popularité.

## Choix de modélisation
- **Pipeline scikit-learn** avec séparation:
  - Variables numériques (année, durée, budget, revenu…)
  - Variables d’identité (réalisateurs/casting/thèmes/genres…)
- **Identités** : encodage par hashing via un transformeur `IdentityHasher` (basé sur `FeatureHasher`).
  - Intérêt: gérer des catégories très nombreuses, éviter un one-hot géant.
  - Limite: collisions possibles (dépend de la dimension de hashing).

## Budget (aide)
- Le modèle budget est entraîné sur `log1p(budget)` puis inversé via `expm1`.
- Une **fourchette** peut être estimée à partir de quantiles des résidus (approximatif).

## Limites
- Les prédictions sont **indicatives**.
- Les corrélations ne sont pas des causalités.
- Les estimations peuvent être instables pour des combinaisons rares (identités peu vues).
"""
)
