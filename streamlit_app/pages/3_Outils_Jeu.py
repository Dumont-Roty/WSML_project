from __future__ import annotations

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

st.set_page_config(page_title="Outils / Jeu — Letterboxd", layout="wide")
H.apply_letterboxd_theme()

st.title("Outils / Jeu")
st.info("Page en construction — on ajoutera ici des mini-outils/jeux basés sur les données Letterboxd.")

st.markdown(
    """
Idées (à implémenter plus tard) :
- **What-if**: comparer deux configurations de film (casting/genres) et voir l'impact sur la note.
- **Deviner le budget**: proposition + score selon l'écart.
- **Templates**: appliquer des presets et comparer.
"""
)
