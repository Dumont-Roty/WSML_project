from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from ml.streamlit_app.helpers import Helpers as H


def _bootstrap_repo_path() -> Path:
    here = Path(__file__).resolve()
    repo_root = next(p for p in here.parents if p.name == "ml").parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root




_bootstrap_repo_path()

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
