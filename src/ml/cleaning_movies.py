# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "polars",
# ]
# ///

"""Nettoyage du dataset films (Letterboxd/TMDB) -> parquet prêt pour ML.

Inspiré de la logique du repo du prof (séance 10/11):
- partir d'un JSON (liste d'objets)
- supprimer doublons
- transformer des colonnes faiblement structurées (listes) en variables numériques
- sérialiser un dataset "propre" au format parquet

Ce script NE fait pas d'entraînement / prédiction.

Usage:
    python src/ml/cleaning_movies.py --input ml/data/merged_results.json --output ml/data/cleaned_data.parquet

Par défaut, on garde `rating` comme colonne cible potentielle (mais on ne l'utilise pas ici).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import polars as pl


DEFAULT_INPUT = Path("ml/data/final_results_28.json")
DEFAULT_OUTPUT = Path("ml/data/cleaned_data.parquet")


def read_movies_json(path: Path) -> pl.DataFrame:
    """Lit un JSON de type liste d'objets et retourne un DataFrame polars."""
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
    if "url" in df.columns:
        return df.unique(["url"], keep="first")
    return df.unique(keep="first")


def _safe_list_len(expr: pl.Expr) -> pl.Expr:
    return pl.when(expr.is_null()).then(pl.lit(0)).otherwise(expr.list.len())


def construit_features_numeriques(df: pl.DataFrame) -> pl.DataFrame:
    """Construit un dataset numérique minimal prêt pour ML.

    - garde quelques colonnes d'identification (url, title)
    - convertit les champs numériques en Float64
    - ajoute des colonnes *_count à partir des listes
    """

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

    # Conversions numériques (si la colonne existe)
    conversions: list[pl.Expr] = []
    for c in numeric_cols:
        if c in df.columns:
            conversions.append(pl.col(c).cast(pl.Float64, strict=False).alias(c))

    # Counts de listes (si la colonne existe)
    counts: list[pl.Expr] = []
    for c in list_cols:
        if c in df.columns:
            counts.append(_safe_list_len(pl.col(c)).cast(pl.Int64).alias(f"{c}_count"))

    # colonnes "meta" conservées si présentes
    keep_meta: list[str] = [c for c in ("url", "title") if c in df.columns]

    out = df.with_columns(conversions + counts)

    # Remarque: on ne fait pas d'imputation "avancée" (pas vu dans le repo prof).
    # On remplace juste les null numériques par 0 pour obtenir une matrice ML exploitable.
    select_cols: list[str] = (
        keep_meta
        + [c for c in numeric_cols if c in out.columns]
        + [f"{c}_count" for c in list_cols if f"{c}_count" in out.columns]
    )

    out = out.select(select_cols)

    # fill nulls for numeric columns (Float/Int) except target
    target_col = "rating"
    out = out.with_columns([
        pl.when(pl.col(c).is_null()).then(pl.lit(0)).otherwise(pl.col(c)).alias(c)
        for c in out.columns
        if out.schema[c] in (pl.Float64, pl.Int64) and c != target_col
    ])

    return out


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Nettoyage dataset films -> cleaned_data.parquet")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Chemin du JSON source")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Chemin du parquet de sortie")
    args = parser.parse_args(list(argv) if argv is not None else None)

    inp = Path(args.input)
    out = Path(args.output)

    df = read_movies_json(inp)
    df = elimine_doublons(df)
    cleaned = construit_features_numeriques(df)

    out.parent.mkdir(parents=True, exist_ok=True)
    cleaned.write_parquet(str(out))

    print(f"OK: {len(cleaned)} lignes, {len(cleaned.columns)} colonnes -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
