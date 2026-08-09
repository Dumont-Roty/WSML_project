from __future__ import annotations

import streamlit as st


st.set_page_config(page_title="WSML", layout="wide")

pages = [
    st.Page("streamlit_app/pages/0_Presentation.py", title="Présentation"),
    st.Page("streamlit_app/pages/1_Estimateur_Letterboxd.py", title="Estimateur Letterboxd"),
    st.Page("streamlit_app/pages/2_Exploration_des_donnees.py", title="Exploration des données"),
    st.Page("streamlit_app/pages/3_Outils_Jeu.py", title="Outils / Jeu"),
    st.Page("streamlit_app/pages/4_A_propos.py", title="À propos"),
]

# Prefer the newer navigation API (it hides app.py from the nav entirely).
if hasattr(st, "navigation") and hasattr(st, "Page"):
    nav = st.navigation(pages)
    nav.run()
else:
    # Fallback: redirect (app.py may still appear in the old multipage sidebar)
    try:
        st.switch_page("pages/0_Presentation.py")
    except Exception:
        st.title("WSML")
        st.info(
            "Navigation multipage non supportée par cette version de Streamlit. "
            "Utilise le menu à gauche pour ouvrir une page."
        )
