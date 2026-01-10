from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap paths so `ml.*` is importable under Streamlit.
HERE = Path(__file__).resolve()
ML_DIR = next(p for p in HERE.parents if p.name == "ml")
REPO_ROOT = ML_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from ml.streamlit_app.helpers import Helpers as H

st.set_page_config(page_title="Présentation — Estimateur Letterboxd", layout="wide")
H.apply_letterboxd_theme()

st.title("Présentation", text_alignment="center")

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
