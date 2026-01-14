Projet ML minimal utilisant `merged_results.json` comme jeu de données de départ.

Structure:
- data/: données brutes et transformées (ne pas committer les gros fichiers)
- notebooks/: notebooks Jupyter pour exploration (si applicable)
- ../src/ml/: code ML réutilisable (préprocessing, features, modèles)
- models/: modèles sérialisés
- streamlit_app/: application Streamlit

Voir `requirements.txt` pour dépendances.

## Début du traitement ML (comme vu en cours)

Objectif: passer d'un JSON faiblement structuré (listes) à un dataset numérique propre sérialisé en parquet.

1) Nettoyage et sérialisation:

	python ../src/ml/cleaning_movies.py --input ml/data/final_results_28.json --output ml/data/cleaned_data.parquet

2) Exploration des valeurs manquantes (optionnel):

	python ../src/ml/preprocess.py --input ml/data/merged_results.json --save-heatmap ml/data/missing_heatmap.png
