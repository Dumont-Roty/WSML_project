# WSML_project

Projet Letterboxd : scraping (Playwright), préparation de données, entraînement de modèles et app Streamlit.

## Installer / activer l’environnement

- Python 3.14 recommandé
- Créer le venv et installer les dépendances :
```powershell
python -m venv .venv
./.venv/Scripts/activate
./.venv/Scripts/python -m pip install -r ml/requirements.txt
./.venv/Scripts/python -m playwright install chromium  # requis pour les scrapers
```

## Scraper les données Letterboxd

Activer l’environnement puis exporter le chemin source pour les modules (PowerShell) :
```powershell
./.venv/Scripts/activate
$env:PYTHONPATH = 'src'
```

- Grilles populaires (parallèle) :
```powershell
./.venv/Scripts/python - <<'PY'
from src.scraping.scrapers.list_scraper import list_scrape_parallel
list_scrape_parallel(max_pages=40, workers=4, headless=True, output_path='results/results_parallel.json')
PY
```
	- Paramètres utiles : `start_page`, `end_page` pour un sous-intervalle ; `preserve_page_order=True` pour garder l’ordre ; `headless=False` pour voir le navigateur.
- Grilles populaires (séquentiel) : remplacer par `list_scrape(...)` si besoin.
- Films à genre unique :
```powershell
./.venv/Scripts/python ./scripts/scrape_single_genre_grids.py --max-pages 40 --start 1 --end 40 --output single_genre_movies.json
```
- Mise à jour ciblée TMDB (budget/revenue) depuis un CSV :
```powershell
./.venv/Scripts/python ./scripts/targeted_update_tmdb.py --input missing_tmdb.csv --output missing_tmdb_updated.csv --delay 0.2
```

Les sorties JSON/CSV peuvent ensuite être fusionnées.

## Fusionner les JSON (agrandir la base)

Sources acceptées :
- racine : `results_*.json`
- dossier `results/` : `results/*.json`

```powershell
./.venv/Scripts/python ./scripts/merge_results.py
# ou avec rapport de doublons
./.venv/Scripts/python ./scripts/merge_results.py --report-duplicates duplicates_report.tsv
```
Sortie par défaut : `ml/data/partial_result_<date>.json` (ignoré par git, cf. `ml/.gitignore`).

## Préparer le dataset pour le modèle

1) Nettoyage/enrichissement :
```powershell
./.venv/Scripts/python ./src/ml/cleaning_movies.py --input ml/data/partial_result_<date>.json --output ml/data/cleaned_data.parquet
```
2) Split train/test :
```powershell
./.venv/Scripts/python ./src/ml/prepare_dataset.py --data ml/data/cleaned_data.parquet --out-train ml/data/train.parquet --out-test ml/data/test.parquet
```

## Entraîner / réentraîner le modèle

Exemple pour la cible `rating` avec identités + clipping :
```powershell
./.venv/Scripts/python ./src/ml/optimize_model.py --train ml/data/partial_result_<date>.json --test-size 0.2 --target rating --clip-predictions --use-identities
```
Sorties : `ml/models/best_model.joblib` et `ml/models/metrics.json`.

## Lancer l’app Streamlit

```powershell
./.venv/Scripts/activate
./.venv/Scripts/python -m streamlit run streamlit_app/app.py --server.port 8501
```
Ouvrir http://localhost:8501. Après réentraînement, redémarrer l’app pour recharger le modèle.

## Tests et couverture

- Tests : `./.venv/Scripts/python -m pytest`
- Couverture : `./.venv/Scripts/python -m pytest --cov=src --cov-report=term-missing`
État actuel : 92 tests, ~82% de couverture (scrapers et ML principaux couverts).

## Récap commandes clés

- Scraper (populaire parallèle) : voir bloc `list_scrape_parallel` ci-dessus
- Scraper single-genre : `./.venv/Scripts/python ./scripts/scrape_single_genre_grids.py --max-pages ...`
- Mise à jour TMDB ciblée : `./.venv/Scripts/python ./scripts/targeted_update_tmdb.py --input ...`
- Merge JSON : `./.venv/Scripts/python ./scripts/merge_results.py`
- Nettoyage → parquet : `./.venv/Scripts/python ./src/ml/cleaning_movies.py --input ... --output ...`
- Split train/test : `./.venv/Scripts/python ./src/ml/prepare_dataset.py --data ...`
- Train modèle : `./.venv/Scripts/python ./src/ml/optimize_model.py --train ... --target rating --clip-predictions --use-identities`
- Lancer l’app : `./.venv/Scripts/python -m streamlit run ml/streamlit_app/app.py --server.port 8501`
- Tests : `./.venv/Scripts/python -m pytest`
- Couverture : `./.venv/Scripts/python -m pytest --cov=src --cov=ml/src --cov-report=term-missing`