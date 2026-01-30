# 📚 Structure du projet WSML - Documentation Technique

## 🎯 Vue d'ensemble

Ce projet implémente un pipeline complet de **Web Scraping** et **Machine Learning** pour prédire les notes Letterboxd des films.

### Architecture en 3 phases

```
1. SCRAPING (src/scraping)     → Récupération des données Letterboxd/TMDB
2. NETTOYAGE (src/ml)           → Transformation et préparation des données
3. MODÉLISATION (src/ml)        → Entraînement et prédiction
```

---

## 📁 Structure des dossiers

```
src/
├── scraping/           # Phase 1 : Collecte de données
│   ├── scrapers/       # Scraping Letterboxd (films, listes, etc.)
│   ├── services/       # Intégration TMDB API
│   ├── browser/        # Gestion de Playwright (navigateur headless)
│   └── utils/          # Utilitaires de parsing
│
├── ml/                 # Phase 2 & 3 : Pipeline ML
│   ├── cleaning_movies.py     # Nettoyage JSON → Parquet
│   ├── prepare_dataset.py     # Split train/test
│   └── optimize_model.py      # Entraînement et optimisation
│
└── utils/              # Outils communs
    └── identity_hasher.py     # Encodage des features catégorielles
```

---

## 🔄 Pipeline complet (étape par étape)

### **Étape 1 : Scraping** (facultatif si données déjà disponibles)

```bash
# Scraping de films Letterboxd + enrichissement TMDB
# Génère : ml/data/partial_result_<date>.json
python scripts/scrape_movies.py --output ml/data/partial_result_2026-01-23.json
```

**Données récupérées :**
- Letterboxd : titre, année, URL, notes, genres, acteurs, réalisateurs
- TMDB : budget, revenus, compositeurs, langues

---

### **Étape 2 : Nettoyage des données**

```bash
# Transformation JSON → Parquet avec features numériques
python src/ml/cleaning_movies.py \
  --input ml/data/partial_result_2026-01-23.json \
  --output ml/data/cleaned_data.parquet
```

**Transformations effectuées :**
- ✅ Suppression des doublons (même URL = même film)
- ✅ Conversion des listes en compteurs : `["Drama", "Thriller"]` → `genres_count=2`
- ✅ Normalisation des types : toutes les colonnes numériques en `float64`
- ✅ Gestion des valeurs manquantes : `None` → `0` ou `NaN`

**Résultat :** Un fichier Parquet compact et rapide à lire

---

### **Étape 3 : Préparation du dataset ML**

```bash
# Split train/test (80/20 par défaut)
python src/ml/prepare_dataset.py \
  --data ml/data/cleaned_data.parquet \
  --out-train ml/data/train.parquet \
  --out-test ml/data/test.parquet \
  --test-size 0.2 \
  --seed 42
```

**Actions :**
- ✅ Filtrage des films sans `rating` (cible manquante)
- ✅ Split aléatoire avec seed fixe (reproductibilité)
- ✅ Génération de deux fichiers indépendants

**Résultat :** 
- `train.parquet` : ~80% des films (entraînement)
- `test.parquet` : ~20% des films (évaluation)

---

### **Étape 4 : Entraînement du modèle**

```bash
# Entraînement avec optimisation d'hyperparamètres
python src/ml/optimize_model.py \
  --train ml/data/train.parquet \
  --target rating \
  --clip-predictions \
  --use-identities \
  --search grid \
  --cv 5
```

**Options importantes :**
- `--clip-predictions` : Force les prédictions dans [0, 5] (bornes Letterboxd)
- `--use-identities` : Active le hashing des features catégorielles (acteurs, genres, etc.)
- `--search grid` : GridSearchCV (exhaustif) vs `random` (plus rapide)
- `--cv 5` : Validation croisée sur 5 folds

**Résultat :**
- `ml/models/best_model.joblib` : Modèle entraîné (format sklearn)
- `ml/models/metrics.json` : Métriques de performance (R², RMSE, MAE)

---

## 🧠 Détail de la modélisation

### Architecture du pipeline sklearn

```python
Pipeline([
    # Étape 1 : Transformation des features
    ('features', ColumnTransformer([
        # Features numériques : normalisation
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  # Remplissage des NaN
            ('scaler', StandardScaler())                    # Centrage/réduction
        ]), NUMERIC_COLS),
        
        # Features catégorielles : hashing
        ('cat', IdentityHasher(
            columns=IDENTITY_COLS,
            n_features=1024  # Projection dans 1024 dimensions
        ), IDENTITY_COLS)
    ])),
    
    # Étape 2 : Régression
    ('regressor', GradientBoostingRegressor())  # Ou RandomForest, Ridge, etc.
])
```

### Pourquoi IdentityHasher ?

**Problème :** Des milliers de noms d'acteurs/réalisateurs uniques → explosion dimensionnelle avec One-Hot Encoding

**Solution :** Feature Hashing (trick du hachage)
- Projette chaque nom dans un vecteur de taille fixe (ex: 1024)
- Collision possible mais impact faible
- Dimension maîtrisée : 1024 au lieu de 50000+

**Exemple :**
```python
Input:  {"directors": ["Christopher Nolan"], "genres": ["Sci-Fi", "Thriller"]}
Output: array de 1024 floats
```

---

## 📊 Métriques de performance

Le fichier `ml/models/metrics.json` contient :

```json
{
  "test_metrics": {
    "r2": 0.45,           // R² score (coefficient de détermination)
    "rmse": 0.38,         // Erreur quadratique moyenne
    "mae": 0.29           // Erreur absolue moyenne
  },
  "train_metrics": { ... },
  "features": [...],      // Liste des features utilisées
  "selected_model": "GradientBoostingRegressor",
  "best_params": { ... }  // Hyperparamètres optimaux
}
```

**Interprétation :**
- **R² = 0.45** : Le modèle explique 45% de la variance des notes
- **RMSE = 0.38** : Erreur moyenne de ±0.38 étoiles (sur échelle 0-5)
- **MAE = 0.29** : Erreur absolue de 0.29 étoiles en moyenne

---

## 🛠️ Utilitaires clés

### `IdentityHasher` (src/utils/identity_hasher.py)

Transformateur sklearn personnalisé pour features catégorielles de type liste.

**Usage :**
```python
from src.utils.identity_hasher import IdentityHasher

# Création
hasher = IdentityHasher(
    columns=("directors", "casting", "genres"),
    n_features=1024
)

# Transformation
X_hashed = hasher.fit_transform(df[["directors", "casting", "genres"]])
# Shape: (n_films, 1024)
```

---

## 🚀 Application Streamlit

L'interface web est dans `streamlit_app/` :

```bash
streamlit run streamlit_app/app.py
```

**Fonctionnalités :**
- 🔮 **Estimateur** : Prédiction de notes pour un film hypothétique
- 📊 **Exploration** : Visualisation des données (corrélations, distributions)
- 🎮 **Jeu** : Duel de films (quel projet aura la meilleure note ?)

---

## 📝 Checklist pour reproduire le projet

1. ✅ **Scraping** : Récupérer les données (ou utiliser partial_result existant)
2. ✅ **Nettoyage** : `cleaning_movies.py` → cleaned_data.parquet
3. ✅ **Split** : `prepare_dataset.py` → train.parquet + test.parquet
4. ✅ **Entraînement** : `optimize_model.py` → best_model.joblib + metrics.json
5. ✅ **Visualisation** : Lancer Streamlit pour l'interface

---

## 🧪 Tests

```bash
# Tests unitaires du scraping
pytest src/scraping/tests/

# Tests de l'IdentityHasher
pytest src/utils/tests/
```

---

## 📚 Dépendances principales

- **Scraping** : `playwright`, `beautifulsoup4`, `requests`
- **ML** : `scikit-learn`, `pandas`, `polars`, `numpy`
- **Visualisation** : `streamlit`, `altair`, `plotly`

Voir `pyproject.toml` pour la liste complète.

---

## 🤝 Contribution

Pour ajouter une feature ou corriger un bug :

1. Créer une branche : `git checkout -b feature/ma-feature`
2. Modifier le code + ajouter des tests
3. Commit : `git commit -m "feat: ajout de ma feature"`
4. Push : `git push origin feature/ma-feature`
5. Créer une Pull Request

---

## 📞 Support

En cas de problème, vérifier :
- [ ] Python ≥ 3.10 installé
- [ ] Dépendances à jour : `pip install -e .`
- [ ] Fichiers de données présents dans `ml/data/`
- [ ] Modèle entraîné dans `ml/models/`

---

**Projet réalisé dans le cadre du Master MECEN - Web Scraping & Machine Learning**
