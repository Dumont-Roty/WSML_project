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

# --- BANDEAU EN COURS DE DÉVELOPPEMENT ---
st.warning("🧪 **Espace Laboratoire** : Cette page est actuellement en cours de développement. Le module 'Duel' est en version Bêta fonctionnelle, d'autres outils arriveront prochainement.")

st.title("⚔️ Le Duel des Studios", text_alignment="center")
st.subheader("Quel projet remportera les faveurs de la communauté ?")

# --- LOGIQUE DU DUEL ---
col_a, col_b = st.columns(2)

with col_a:
    st.header("🎬 Projet A")
    title_a = st.text_input("Nom du film A", "Projet Alpha")
    genre_a = st.selectbox("Genre A", ["Drama", "Horror", "Comedy", "Sci-Fi", "Action"], key="g_a")
    affichage_budget_a = st.empty()
    budget_a = st.slider("Budget prévisionnel", 1, 300, 50, key="b_a")
    affichage_budget_a.write(f"💰 **Budget : {budget_a} M$**")

with col_b:
    st.header("🎬 Projet B")
    title_b = st.text_input("Nom du film B", "Projet Beta")
    genre_b = st.selectbox("Genre B", ["Drama", "Horror", "Comedy", "Sci-Fi", "Action"], key="g_b")
    affichage_budget_b = st.empty()
    budget_b = st.slider("Budget prévisionnel", 1, 300, 50, key="b_b")
    affichage_budget_b.write(f"💰 **Budget : {budget_b} M$**")

st.markdown("---")

if st.button("🚀 Lancer le Duel !", use_container_width=True):
    
    # Simulation simplifiée des notes
    bonus_genre = {"Drama": 0.5, "Sci-Fi": 0.4, "Horror": 0.3, "Comedy": 0.2, "Action": 0.1}
    
    # Simulation Note A
    note_a = 2.8 + bonus_genre.get(genre_a, 0) + (budget_a / 1000)
    # Simulation Note B
    note_b = 2.8 + bonus_genre.get(genre_b, 0) + (budget_b / 1000)
    
    note_a = round(min(note_a, 4.9), 2)
    note_b = round(min(note_b, 4.9), 2)

    st.balloons()
    
    res_a, res_b = st.columns(2)
    with res_a:
        st.metric(label=f"Note estimée pour {title_a}", value=f"{note_a}/5")
    with res_b:
        st.metric(label=f"Note estimée pour {title_b}", value=f"{note_b}/5")

    if note_a > note_b:
        st.success(f"🏆 Victoire pour **{title_a}** !")
    elif note_b > note_a:
        st.success(f"🏆 Victoire pour **{title_b}** !")
    else:
        st.warning("🤝 Égalité parfaite !")

