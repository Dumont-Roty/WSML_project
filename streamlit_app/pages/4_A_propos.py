from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

from streamlit_app.helpers import Helpers as H

# Bootstrap paths so `ml.*` is importable.
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]  # project root
ML_DIR = REPO_ROOT / "ml"
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

metrics_path = ML_DIR / "models" / "metrics.json"
budget_metrics_path = ML_DIR / "models" / "budget_metrics.json"


metrics = H.load_json(metrics_path)
budget_metrics = H.load_json(budget_metrics_path)


st.set_page_config(page_title="À propos — Estimateur Letterboxd", layout="wide")

# Theme (kept local to avoid importing heavy deps here)
LETTERBOXD_COLORS = {
    "bg": "#0b0b0b",
    "text": "#E6E6E6",
    "accent": "#2AB44B",
    "accent_dark": "#1B5E20",
    "muted": "#9aa0a6",
}
st.markdown(
    f"""
    <style>
    :root {{
      --lb-bg: {LETTERBOXD_COLORS['bg']};
      --lb-text: {LETTERBOXD_COLORS['text']};
      --lb-muted: {LETTERBOXD_COLORS['muted']};
      --lb-accent: {LETTERBOXD_COLORS['accent']};
      --lb-accent-dark: {LETTERBOXD_COLORS['accent_dark']};
    }}
    .stApp, .block-container {{
      background-color: var(--lb-bg) !important;
      color: var(--lb-text) !important;
    }}
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
      color: var(--lb-accent) !important;
    }}

    /* Subtle default text tones */
    .stMarkdown p, .stMarkdown li {{
      color: var(--lb-text) !important;
    }}
    .stMarkdown small, .stCaption, .stMarkdown .st-emotion-cache-1wmy9hl {{
      color: var(--lb-muted) !important;
    }}

    /* Buttons */
    .stButton>button {{
      background-color: var(--lb-accent) !important;
      color: #fff !important;
      border: none !important;
    }}

    /* Tables (st.table) */
    .stTable table {{
      color: var(--lb-text) !important;
      border-collapse: collapse !important;
    }}
    .stTable thead tr th {{
      background: var(--lb-accent-dark) !important;
      color: #fff !important;
      border-bottom: 1px solid rgba(255,255,255,0.12) !important;
    }}
    .stTable tbody tr td {{
      background: rgba(255,255,255,0.03) !important;
      border-bottom: 1px solid rgba(255,255,255,0.08) !important;
    }}

    /* Dataframes (st.dataframe) */
    .stDataFrame {{
      background: rgba(255,255,255,0.02) !important;
      border: 1px solid rgba(255,255,255,0.08) !important;
      border-radius: 8px !important;
    }}

    /* Code blocks */
    code {{
      color: var(--lb-accent) !important;
    }}
    pre {{
      background: rgba(255,255,255,0.03) !important;
      border: 1px solid rgba(255,255,255,0.08) !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)
####################################################

st.title("ℹ️ À propos du Projet", text_alignment="center")

tab_concept, tab_methode, tab_data = st.tabs([
    "🎯 Concept & Objectifs", 
    "🧠 Méthodologie ML", 
    "📊 Source des Données"
])

with tab_concept:
    st.markdown("""
    ### Le Machine Learning au service du Cinéma
    L'objectif de cette application est de décoder la réception d'une œuvre par la communauté **Letterboxd**. 
    En utilisant des algorithmes de régression, nous estimons la note moyenne (0-5) qu'un film pourrait obtenir.
    
    **Ce que le modèle analyse :**
    - La "signature" des réalisateurs et acteurs (notoriété statistique).
    - Les indicateurs économiques (budget, revenus).
    - Les caractéristiques techniques (genres, thèmes, durée).
    - Etc.
    """)

with tab_methode:
    st.markdown("""
    ### Choix de Modélisation
    Nous utilisons un pipeline **Scikit-Learn** sophistiqué :
    - **Identity Hashing** : Pour gérer des milliers de noms (acteurs/réalisateurs) sans créer des fichiers géants, nous utilisons le *Hashing Trick*. Cela permet de capturer l'influence d'un individu même dans un casting choral.
    """)
# **Modèle Budget** : Un modèle secondaire aide à prédire ou contextualiser le budget via une transformation logarithmique (`log1p`) pour stabiliser les écarts extrêmes entre blockbusters et films indépendants.

with tab_data:
    st.info("⚠️ **Note sur la fiabilité** : Les données sont issues d'un pipeline de scraping. Elles reflètent des corrélations statistiques et non des vérités absolues.")
    st.markdown("""
    - **Origine** : Scraping hybride Letterboxd + Enrichissement TMDB.
    - **Traitement** : Les valeurs manquantes sont imputées par la médiane pour garantir que le modèle puisse toujours répondre.
    """)

st.divider()

# --- Section Performance (Dynamique) ---
st.header("📈 Performance du Modèle", text_alignment="center")

if not metrics:
    st.error("📊 Données de performance (`metrics.json`) indisponibles.")
else:
    # Affichage des métriques clés dans des colonnes avec design
    test_metrics = metrics.get("test_metrics", {})
    r2 = test_metrics.get("r2", 0)
    rmse = test_metrics.get("rmse", 0)
    acc050 = test_metrics.get("acc_within_0_50", 0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Précision (R²)", f"{float(r2):.3f}", help="Plus proche de 1 est le mieux.")
    c2.metric("Erreur (RMSE)", f"{float(rmse):.3f}", help="Écart moyen en points de note.")
    c3.metric("Fiabilité à ±0.5", f"{float(acc050)*100:.1f}%", help="% de films prédits avec moins de 0.5 point d'écart.")

    # Expander pour les détails techniques plus profonds
    with st.expander("🔍 Voir les détails techniques complets"):
        col_l, col_r = st.columns(2)
        with col_l:
            st.write(f"**Modèle sélectionné :** `{metrics.get('selected_model', 'N/A')}`")
            st.write(f"**Cible :** `{metrics.get('target', 'rating')}`")
        with col_r:
            input_schema = metrics.get("input_schema", {})
            st.write(f"**Variables numériques :** {len(input_schema.get('numeric_cols', []))}")
            st.write(f"**Colonnes d'identité :** {len(input_schema.get('identity_cols', []))}")
        
        cv_results = metrics.get("cv_results", [])
        if cv_results:
            st.write("**Comparaison des modèles testés (Cross-Validation) :**")
            st.table(cv_results)

st.divider()

# --- Section Pédagogique (Expliquer le fonctionnement) ---
st.header("🛠️ Comment ça marche ?", text_alignment="center")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("👥 L'effet 'Star System'")
    st.markdown("""
    Le modèle ne compte pas simplement le nombre d'acteurs. Il reconnaît les **identités**. 
    Un acteur historiquement associé à des films bien notés apportera un "poids" positif à la prédiction. 
    *Ce n'est pas un jugement de talent, mais un constat statistique sur vos données.*
    """)

with col_b:
    st.subheader("📉 Score de Confiance")
    st.markdown("""
    Chaque prédiction est accompagnée d'un indice de complétude :
    - **Élevé (80%+)** : Tous les champs sont remplis.
    - **Moyen (50%+)** : Certains champs manquent (médiane utilisée).
    - **Faible (-50%)** : Données insuffisantes pour une prédiction fiable.
    """)

# --- Footer ---
st.caption("Projet Master MECEN - Web Scraping et Machine Learning", text_alignment="center")

