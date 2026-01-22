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


st.divider()
st.header("Détails techniques")


st.subheader("Modèle de note (rating)")
if not metrics:
  st.info("Fichier `ml/models/metrics.json` introuvable ou illisible.")
else:
  target = str(metrics.get("target") or "rating")
  selected_model = metrics.get("selected_model")
  scoring = metrics.get("scoring")
  feature_mode = metrics.get("feature_mode")
  input_schema = metrics.get("input_schema") if isinstance(metrics.get("input_schema"), dict) else {}
  numeric_cols = input_schema.get("numeric_cols") if isinstance(input_schema.get("numeric_cols"), list) else []
  identity_cols = input_schema.get("identity_cols") if isinstance(input_schema.get("identity_cols"), list) else []
  hash_dim = input_schema.get("hash_dim")

  left, right = st.columns(2)
  with left:
    st.markdown(
      "\n".join(
        [
          f"- **Cible**: `{target}`",
          f"- **Modèle sélectionné**: `{selected_model}`" if selected_model else "- **Modèle sélectionné**: —",
          f"- **Scoring**: `{scoring}`" if scoring else "- **Scoring**: —",
          f"- **Mode des features**: `{feature_mode}`" if feature_mode else "- **Mode des features**: —",
        ]
      )
    )
  with right:
    st.markdown("**Schéma d'entrée**")
    st.markdown(f"- `numeric_cols`: {len(numeric_cols)}")
    st.markdown(f"- `identity_cols`: {len(identity_cols)}")
    if hash_dim is not None:
      st.markdown(f"- `hash_dim`: `{hash_dim}`")

  test_metrics = metrics.get("test_metrics") if isinstance(metrics.get("test_metrics"), dict) else {}
  if test_metrics:
    st.markdown("**Qualité (test)**")

    r2 = test_metrics.get("r2")
    mae = test_metrics.get("mae")
    rmse = test_metrics.get("rmse")
    acc025 = test_metrics.get("acc_within_0_25")
    acc050 = test_metrics.get("acc_within_0_50")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
      st.metric("R²", f"{float(r2):.3f}" if r2 is not None else "—")
    with c2:
      st.metric("MAE", f"{float(mae):.3f}" if mae is not None else "—")
    with c3:
      st.metric("RMSE", f"{float(rmse):.3f}" if rmse is not None else "—")
    with c4:
      st.metric("Acc@0.25", f"{100.0*float(acc025):.1f}%" if acc025 is not None else "—")
    with c5:
      st.metric("Acc@0.50", f"{100.0*float(acc050):.1f}%" if acc050 is not None else "—")

    st.caption(
      "Acc@0.25/0.50 = proportion de prédictions à moins de 0.25/0.5 point de la vraie note. "
      "MAE/RMSE sont en points de note (sur 0–5)."
    )

  cv_results = metrics.get("cv_results") if isinstance(metrics.get("cv_results"), list) else []
  if cv_results:
    st.markdown("**Comparaison modèles (CV)**")
    st.table(cv_results)


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


st.subheader("Comment sont utilisées les personnes (casting, réalisateurs, etc.)")
st.markdown(
  """
Dans ce projet, les personnes/catégories (ex: `casting`, `directors`, `genres`, `themes`…) sont fournies au modèle comme **listes d'identités**.

- Le modèle ne reçoit pas simplement “le nombre d'acteurs”, mais les **noms** (identités) encodés par hashing.
- Conséquence: **1 acteur “top” vs 5 acteurs inconnus** peut changer la prédiction à cause des identités elles‑mêmes, pas seulement à cause de la taille du casting.
- Attention: ce n'est pas une mesure de “qualité réelle” d'un acteur, mais des **corrélations apprises** sur les données d'entraînement.
"""
)


st.subheader("Qualité estimée des prédictions selon les champs fournis")
st.markdown(
  """
La précision d'une prédiction dépend de la quantité et de la pertinence des champs fournis.

- Si tous les champs numériques et d'identité présents dans le schéma d'entrée sont renseignés, la prédiction a une confiance plus élevée.
- Si plusieurs champs manquent, le modèle retombe sur des valeurs médianes ou jeux d'identités vides, ce qui réduit la fiabilité.

Règle simple utilisée dans l'interface :

1. On compte le nombre de champs attendus (`numeric_cols` + `identity_cols`).
2. On compte combien d'entre eux l'utilisateur a explicitement activés/complétés.
3. `completeness = fournis / total` → niveau :
   - `>= 0.8` : confiance élevée
   - `>= 0.5` : confiance moyenne
   - `<  0.5` : confiance faible

Concrètement, lorsqu'un champ n'est pas renseigné :

- Les variables numériques manquantes sont remplacées par leur médiane calculée sur l'ensemble d'entraînement.
- Les listes d'identités non fournies sont traitées comme vides (ou par défaut), ce qui supprime les effets d'identités spécifiques.
- Pour le budget, si absent, le modèle utilise une prédiction de budget (modèle d'aide) ou la médiane selon la configuration.

La page principale affichera, pour chaque prédiction, un indicateur de complétude et un message (ex : "Confiance : Moyenne — Plusieurs champs manquent").
"""
)
