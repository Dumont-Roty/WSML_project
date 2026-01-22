
# Préparation du dataset ML : split train/test à partir du parquet nettoyé
# Ce script ne fait pas d'entraînement, il prépare juste les fichiers d'entrée pour la modélisation.

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split



def main() -> int:
    """
    Point d'entrée du script : split train/test à partir du parquet nettoyé.
    - Filtre les lignes où la cible (ex: rating) est manquante
    - Sépare en train/test (proportion paramétrable)
    - Écrit deux fichiers parquet pour la suite du pipeline ML
    """
    parser = argparse.ArgumentParser(
        description=(
            "Prépare un dataset ML à partir de cleaned_data.parquet (sans entraînement).\n"
            "- filtre les lignes où la cible est manquante\n"
            "- split train/test\n"
            "- écrit train.parquet et test.parquet"
        )
    )
    parser.add_argument("--data", default="ml/data/cleaned_data.parquet", help="Chemin du parquet nettoyé")
    parser.add_argument("--target", default="rating", help="Nom de la colonne cible (ex: rating)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion du test set (ex: 0.2 = 20%)")
    parser.add_argument("--seed", type=int, default=42, help="Seed aléatoire pour la reproductibilité")
    parser.add_argument(
        "--out-train",
        default="ml/data/train.parquet",
        help="Chemin de sortie parquet pour le train",
    )
    parser.add_argument(
        "--out-test",
        default="ml/data/test.parquet",
        help="Chemin de sortie parquet pour le test",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {data_path}")

    # Lecture du parquet nettoyé
    df = pd.read_parquet(data_path)
    if args.target not in df.columns:
        raise ValueError(f"Colonne cible '{args.target}' absente. Colonnes: {list(df.columns)}")

    # On ne garde que les lignes où la cible est présente (pas de NaN)
    df = df.copy()
    df[args.target] = pd.to_numeric(df[args.target], errors="coerce")
    df = df[df[args.target].notna()]

    # Split train/test (aléatoire, proportion paramétrable)
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
    )

    # Sauvegarde des fichiers de sortie
    out_train = Path(args.out_train)
    out_test = Path(args.out_test)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_test.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(out_train, index=False)
    test_df.to_parquet(out_test, index=False)

    print(f"OK: {len(df)} lignes utilisables (cible non-NaN)")
    print(f"Train: {len(train_df)} -> {out_train}")
    print(f"Test : {len(test_df)} -> {out_test}")
    print(f"Colonnes: {len(df.columns)} (target='{args.target}')")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
