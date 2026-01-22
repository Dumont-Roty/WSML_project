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

st.set_page_config(page_title="Présentation — Estimateur Letterboxd", layout="wide")
H.apply_letterboxd_theme()

def show_presentation():

    # Titre principal 
    st.title("🎬 Letterboxd Insights", text_alignment="center")
    st.subheader("Le Machine Learning au service du 7ème Art")

    st.markdown("---")

    # Section 1 : Le Pitch 
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
            ### ✨ Est-il possible d'anticiper le succès d'un film ?
            Bienvenue sur **Letterboxd Insights**, une plateforme analytique conçue pour décoder les secrets des notes cinématographiques. 
            
            En croisant les données sociales de **Letterboxd** avec les indicateurs économiques de **TMDB**, notre algorithme de Machine Learning identifie les facteurs qui font d'un film une œuvre culte ou un succès populaire.
            """)

    with col2:
        st.info("💡 **Le Saviez-vous ?** \n\n Le budget n'est pas toujours le premier facteur de succès sur Letterboxd. L'influence du réalisateur et le genre pèsent parfois bien plus lourd !")

    st.markdown("---")

    # Section 2 : Les fonctionnalités 
    st.write("### 🚀 Explorez les fonctionnalités")
        
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### 🔮 **Prédire**")
        st.write("Utilisez l'**Estimateur** pour simuler la réception d'un film en fonction de son casting, son budget et son équipe technique.")
            
    with c2:
        st.markdown("#### 📊 **Analyser**")
        st.write("Plongez dans l'**Exploration de données** pour visualiser les corrélations entre revenus, genres et notations communautaires.")

    with c3:
        st.markdown("#### 🧠 **Comprendre**")
        st.write("Consultez l'onglet **À propos** pour découvrir les coulisses de notre modèle de Machine Learning et ses performances.")

    st.markdown("---")

    # Section 3 : Note de méthodologie 
    with st.expander("📌 Note méthodologique & Données"):
        st.write("""Ce projet repose sur un dataset unique combinant scraping et API. \n **Source principale :** Letterboxd (données sociales et critiques).\n **Source secondaire :** TMDB (budgets, revenus, visuels).\n **Précision :** Les estimations fournies sont indicatives. Elles reflètent les tendances statistiques observées sur plus de 3 000 films analysés.""")

# Appel de la fonction
show_presentation()