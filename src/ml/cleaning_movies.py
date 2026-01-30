
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "polars",
# ]
# ///

"""
Nettoyage du dataset films (Letterboxd/TMDB) -> parquet prêt pour ML.

Ce script prépare les données pour l'apprentissage automatique :
- lit un JSON (liste d'objets films)
- supprime les doublons (même film plusieurs fois)
- convertit les listes (ex: genres, casting) en compteurs numériques
- sérialise le tout au format parquet (plus rapide et compact que CSV/JSON)

Inspiré de la logique du repo du prof (séance 10/11).
Ce script NE fait pas d'entraînement / prédiction.

Usage :
    python src/ml/cleaning_movies.py --input ml/data/partial_result_<date>.json --output ml/data/cleaned_data.parquet

Par défaut, on garde `rating` comme colonne cible potentielle (mais on ne l'utilise pas ici).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import polars as pl


DEFAULT_INPUT = Path("ml/data/partial_result_2026-01-23.json")
DEFAULT_OUTPUT = Path("ml/data/cleaned_data.parquet")


def read_movies_json(path: Path) -> pl.DataFrame:
    """
    Lit un JSON (liste d'objets films) et retourne un DataFrame polars.
    Gère les cas où pl.read_json échoue (format non standard) en fallback sur json.load.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")

    # Polars sait lire certains JSON "array" mais pas tous selon version.
    # On tente d'abord le chemin rapide, sinon fallback json.load.
    try:
        return pl.read_json(str(path))
    except Exception:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Le JSON doit être une liste d'objets")
        return pl.from_dicts(data)


def elimine_doublons(df: pl.DataFrame) -> pl.DataFrame:
    """
    Supprime les doublons de films (même URL = même film).
    Si pas de colonne 'url', dédoublonne sur toutes les colonnes.
    """
    if "url" in df.columns:
        return df.unique(["url"], keep="first")
    return df.unique(keep="first")


def _safe_list_len(expr: pl.Expr) -> pl.Expr:
    """
    Renvoie la taille d'une liste, ou 0 si la valeur est nulle.
    Utile pour compter les genres, acteurs, etc. même si la donnée est manquante.
    """
    return pl.when(expr.is_null()).then(pl.lit(0)).otherwise(expr.list.len())


def construit_features_numeriques(df: pl.DataFrame) -> pl.DataFrame:
    """
    Transforme les colonnes en features numériques utilisables par sklearn.
    
    Transformations appliquées :
    1. Colonnes scalaires (year, duration, budget, etc.) : converties en float64
    2. Colonnes listes (genres, casting, directors, etc.) : comptées (nombre d'éléments)
       - Ex: ["Drama", "Thriller"] -> genres_count = 2
    3. Valeurs manquantes : remplacées par 0 ou None selon le contexte
    
    Cette étape prépare les données pour l'apprentissage supervisé où chaque
    feature doit être un nombre.
    """

    # Colonnes numériques à garder (si elles existent)
    numeric_cols: list[str] = [
        "year",
        "duration",
        "nbr_watched",
        "nbr_appearence",
        "nbr_likes",
        "fans_favoris",
        "budget",
        "revenue",
        "rating",
    ]

    # Colonnes de type liste à convertir en compteurs
    list_cols: list[str] = [
        "directors",
        "casting",
        "producers",
        "writers",
        "composer",
        "studio",
        "languages",
        "genres",
        "themes",
    ]

    # Conversion des colonnes numériques en float (si présentes)
    conversions: list[pl.Expr] = []
    for c in numeric_cols:
        if c in df.columns:
            conversions.append(pl.col(c).cast(pl.Float64, strict=False).alias(c))

    # Pour chaque colonne liste, ajoute une colonne *_count (nb d'éléments)
    counts: list[pl.Expr] = []
    for c in list_cols:
        if c in df.columns:
            counts.append(_safe_list_len(pl.col(c)).cast(pl.Int64).alias(f"{c}_count"))

    # Colonnes "meta" à garder si présentes (identification)
    keep_meta: list[str] = [c for c in ("url", "title") if c in df.columns]

    # Applique les conversions et ajouts de compteurs
    out = df.with_columns(conversions + counts)

    # Remarque : pas d'imputation avancée ici (pas vu dans le repo prof).
    # On remplace juste les nulls numériques par 0 pour obtenir une matrice ML exploitable.
    # On sélectionne les colonnes finales à garder dans le dataset
    select_cols: list[str] = (
        keep_meta
        + [c for c in numeric_cols if c in out.columns]
        + [f"{c}_count" for c in list_cols if f"{c}_count" in out.columns]
    )

    out = out.select(select_cols)

    # Remplit les valeurs manquantes numériques par 0 (sauf la colonne cible)
    target_col = "rating"
    out = out.with_columns([
        pl.when(pl.col(c).is_null()).then(pl.lit(0)).otherwise(pl.col(c)).alias(c)
        for c in out.columns
        if out.schema[c] in (pl.Float64, pl.Int64) and c != target_col
    ])

    return out


def main(argv: Iterable[str] | None = None) -> int:
    """
    Point d'entrée du script (exécuté en ligne de commande).
    Lit le JSON, nettoie, transforme, et écrit le parquet final.
    """
    parser = argparse.ArgumentParser(description="Nettoyage dataset films -> cleaned_data.parquet")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Chemin du JSON source")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Chemin du parquet de sortie")
    args = parser.parse_args(list(argv) if argv is not None else None)

    inp = Path(args.input)
    out = Path(args.output)

    # Lecture du JSON source
    df = read_movies_json(inp)
    # Suppression des doublons
    df = elimine_doublons(df)
    # Transformation en features numériques
    cleaned = construit_features_numeriques(df)

    # Sauvegarde du parquet final
    out.parent.mkdir(parents=True, exist_ok=True)
    cleaned.write_parquet(str(out))

    print(f"OK: {len(cleaned)} lignes, {len(cleaned.columns)} colonnes -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
