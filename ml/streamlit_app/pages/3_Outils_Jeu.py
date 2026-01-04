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

st.set_page_config(page_title="Outils / Jeu — Letterboxd", layout="wide")
_apply_letterboxd_theme()

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
