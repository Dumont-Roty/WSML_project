# WSML_project

Ce dépôt contient un projet autour de données Letterboxd (scraping + modèle ML + app Streamlit).

## Fusionner les JSON (agrandir la base)

### 1) Où mettre les fichiers

Tu peux ajouter de nouveaux exports JSON à l’un de ces emplacements :

- à la racine du repo : `results_*.json` (ex: `results_81_90.json`)
- dans le dossier `results/` : `results/*.json`

Le script sait lire :

- des fichiers JSON contenant une **liste** d’objets
- ou un **objet** JSON unique (il sera automatiquement converti en liste)

### 2) Construire le fichier `partial_result_<date>.json`

Le script centralisé est : `scripts/merge_results.py`.

Commande (PowerShell) :

- `./.venv/Scripts/python ./scripts/merge_results.py`

Par défaut :

- la sortie est écrite dans `ml/data/partial_result_<date>.json` (ex: `partial_result_2026-01-10.json`)
- la déduplication est **activée** (pour éviter de gonfler le dataset avec des doublons)

### 3) Vérifier les doublons potentiels

Le script déduplique sur le champ `url` (quand il existe). Pour auditer les doublons, tu peux générer un rapport :

- `./.venv/Scripts/python ./scripts/merge_results.py --report-duplicates duplicates_report.tsv`

Le rapport inclut : `url`, `year`, `directors`, `title`, et `source_file`.

### 4) Étape recommandée “doublons” (à faire plus tard)

L’objectif est de vérifier s’il existe des doublons (même film) avant de figer une version finale de dataset :

- comparer par `url` (facile)
- éventuellement compléter avec une heuristique `title + year` si certaines URLs manquent

Notes :

- `ml/data/` est ignoré par git (voir `ml/.gitignore`) : c’est normal que `partial_result_<date>.json` ne soit pas versionné.

## Relancer la prédiction avec les nouvelles données

Si tu ajoutes de nouvelles données scrapées (puis fusionnées), tu as deux niveaux de “mise à jour” possibles :

- **Mettre à jour le dataset de référence** (statistiques, options acteurs/réals, exploration) : tu changes le fichier JSON utilisé comme référence.
- **Mettre à jour le modèle** (les prédictions elles-mêmes) : tu ré-entraînes et tu remplaces les artefacts dans `ml/models/`.

### A) Mettre à jour le modèle (recommandé)

1) Fusionner les JSON (génère un `partial_result_<date>.json`) :

- `./.venv/Scripts/python ./scripts/merge_results.py`

Le fichier généré ressemble à `ml/data/partial_result_2026-01-10.json`. Dans les commandes ci-dessous, remplace `<date>` par la date réelle.

2) (Optionnel mais utile) Recalculer un `train.parquet`/`test.parquet` à partir du nouveau JSON (sert aux stats, quantiles, etc.) :

- `./.venv/Scripts/python ./ml/src/cleaning_movies.py --input ml/data/partial_result_<date>.json --output ml/data/cleaned_data.parquet`
- `./.venv/Scripts/python ./ml/src/prepare_dataset.py --data ml/data/cleaned_data.parquet --out-train ml/data/train.parquet --out-test ml/data/test.parquet`

3) Ré-entraîner le modèle `rating` à partir du JSON enrichi (inclut les identités: réalisateurs/casting/etc.) :

- `./.venv/Scripts/python ./ml/src/optimize_model.py --train ml/data/partial_result_<date>.json --test-size 0.2 --target rating --clip-predictions --use-identities`

Sorties attendues (utilisées par Streamlit) :

- `ml/models/best_model.joblib`
- `ml/models/metrics.json`

4) Relancer l’app Streamlit (important : Streamlit met en cache le modèle ; il faut redémarrer l’app pour recharger un nouveau `.joblib`) :

- `./.venv/Scripts/python -m streamlit run ./ml/streamlit_app/app.py`

### B) Mettre à jour le dataset de référence (sans ré-entraîner)

Si tu veux seulement que l’app (exploration + stats) s’appuie sur le dataset fusionné, sans changer le modèle :

- Utilise le nouveau `ml/data/partial_result_<date>.json` comme fichier de référence dans les pages Streamlit.

Actuellement, les pages utilisent surtout `ml/data/final_results_28.json` et en fallback `merged_results.json`. La manière la plus simple est de **pointer** ces chemins vers ton nouveau fichier (sans le committer). Si tu veux, je peux faire une petite amélioration pour que l’app détecte automatiquement le dernier `partial_result_*.json` dans `ml/data/`.

'