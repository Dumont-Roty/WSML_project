# Script d'exploration rapide des valeurs manquantes dans un JSON de films
# Affiche le nombre de NaN par colonne et peut générer une heatmap PNG

from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
	parser = argparse.ArgumentParser(description="Exploration rapide des valeurs manquantes")  # Description du script
	parser.add_argument("--input", default="ml/data/final_results_28.json", help="Chemin du JSON")
	parser.add_argument(
		"--save-heatmap",
		default=None,
		help="Chemin PNG de sortie pour la heatmap des NaN (optionnel)",
	)
	args = parser.parse_args()

	inp = Path(args.input)  # Chargement du chemin d'entrée
	df = pd.read_json(inp)

	print("\n--- Missing values (par colonne) ---")
	print(df.isnull().sum().sort_values(ascending=False).head(30))

	print("\n--- DataFrame info ---")
	df.info()  # Affichage des informations du DataFrame

	if args.save_heatmap:
		import matplotlib

		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
		import seaborn as sns

		out = Path(args.save_heatmap)
		out.parent.mkdir(parents=True, exist_ok=True)

		plt.figure(figsize=(12, 6))  # Création de la figure pour la heatmap
		sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
		plt.title("Valeurs manquantes par ligne/colonne")
		plt.tight_layout()
		plt.savefig(out, dpi=150)
		print(f"\nHeatmap sauvegardée: {out}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())