from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# Bootstrap paths so `ml.*` is importable.
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]  # project root
ML_DIR = REPO_ROOT / "ml"
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

metrics_path = ML_DIR / "models" / "metrics.json"
budget_metrics_path = ML_DIR / "models" / "budget_metrics.json"

try:
  metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
except Exception:
  metrics = {}

try:
  budget_metrics = json.loads(budget_metrics_path.read_text(encoding="utf-8")) if budget_metrics_path.exists() else {}
except Exception:
  budget_metrics = {}

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

st.title("ℹ️ À propos du Projet")

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
st.header("📈 Performance du Modèle")

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
st.header("🛠️ Comment ça marche ?")

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
st.caption("Projet Master MECEN - Web Scraping et Machine Learning")


# --- Section Budget (Désactivée) ---

# st.subheader("Modèle de budget (aide)")
# if not budget_metrics:
#   st.info("Fichier `ml/models/budget_metrics.json` introuvable ou illisible.")
# else:
#   target = str(budget_metrics.get("target") or "budget")
#   selected_model = budget_metrics.get("selected_model")
#   scoring = budget_metrics.get("scoring")
#   feature_mode = budget_metrics.get("feature_mode")
#   input_schema = budget_metrics.get("input_schema") if isinstance(budget_metrics.get("input_schema"), dict) else {}
#   numeric_cols = input_schema.get("numeric_cols") if isinstance(input_schema.get("numeric_cols"), list) else []
#   identity_cols = input_schema.get("identity_cols") if isinstance(input_schema.get("identity_cols"), list) else []
#   hash_dim = input_schema.get("hash_dim")

#   left, right = st.columns(2)
#   with left:
#     st.markdown(
#       "\n".join(
#         [
#           f"- **Cible**: `{target}`",
#           f"- **Modèle sélectionné**: `{selected_model}`" if selected_model else "- **Modèle sélectionné**: —",
#           f"- **Scoring**: `{scoring}`" if scoring else "- **Scoring**: —",
#           f"- **Mode des features**: `{feature_mode}`" if feature_mode else "- **Mode des features**: —",
#         ]
#       )
#     )
#   with right:
#     st.markdown("**Schéma d'entrée**")
#     st.markdown(f"- `numeric_cols`: {len(numeric_cols)}")
#     st.markdown(f"- `identity_cols`: {len(identity_cols)}")
#     if hash_dim is not None:
#       st.markdown(f"- `hash_dim`: `{hash_dim}`")

#   test_metrics = budget_metrics.get("test_metrics") if isinstance(budget_metrics.get("test_metrics"), dict) else {}
#   if test_metrics:
#     st.markdown("**Qualité (test)**")
#     r2 = test_metrics.get("r2")
#     mae = test_metrics.get("mae")
#     rmse = test_metrics.get("rmse")
#     acc025 = test_metrics.get("acc_within_0_25")
#     acc050 = test_metrics.get("acc_within_0_50")

#     c1, c2, c3, c4, c5 = st.columns(5)
#     with c1:
#       st.metric("R²", f"{float(r2):.3f}" if r2 is not None else "—")
#     with c2:
#       st.metric("MAE", f"{float(mae):.3f}" if mae is not None else "—")
#     with c3:
#       st.metric("RMSE", f"{float(rmse):.3f}" if rmse is not None else "—")
#     with c4:
#       st.metric("Acc@0.25", f"{100.0*float(acc025):.1f}%" if acc025 is not None else "—")
#     with c5:
#       st.metric("Acc@0.50", f"{100.0*float(acc050):.1f}%" if acc050 is not None else "—")

#     if acc025 is not None or acc050 is not None:
#       st.caption(
#         "Acc@0.25/0.50 = proportion de prédictions à moins de 0.25/0.5 unité de la vraie valeur."
#       )

#   cv_results = budget_metrics.get("cv_results") if isinstance(budget_metrics.get("cv_results"), list) else []
#   if cv_results:
#     st.markdown("**Comparaison modèles (CV)**")
#     st.table(cv_results)

#   interval = budget_metrics.get("prediction_interval") if isinstance(budget_metrics.get("prediction_interval"), dict) else None
#   if interval:
#     st.markdown("**Intervalle de prédiction (approx.)**")
#     st.write(interval)