"""Fusionne plusieurs fichiers JSON de résultats en un seul.

Chaque fichier d'entrée doit contenir une liste d'objets JSON (résultats sérialisés).
Par défaut, les enregistrements sont dédupliqués par `url` si présente; sinon ils
sont ajoutés tels quels. Le résultat est écrit dans un fichier JSON.

Usage (PowerShell) :    
    $env:PYTHONPATH='src;src/WSML_code'
    .\.venv\Scripts\python .\scripts\merge_results.py results_1.json results_2.json results_3.json results_4.json --output merged_results.json

Options :
    --no-dedup    Désactive la déduplication par url (concatène tout)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, List


def load_json_list(path: Path) -> List[Any]:
    """Charge un fichier JSON qui doit contenir une liste."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Le fichier {path} ne contient pas une liste JSON")
    return data


def merge_files(paths: Iterable[Path], dedup: bool = True) -> list[Any]:
    merged: list[Any] = []
    seen_urls: set[str] = set()

    for path in paths:
        records = load_json_list(path)
        for rec in records:
            if dedup:
                url = rec.get("url") if isinstance(rec, dict) else None
                if isinstance(url, str):
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
            merged.append(rec)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Fusionne plusieurs fichiers JSON de résultats")
    parser.add_argument("inputs", nargs="+", help="Chemins des fichiers JSON à fusionner")
    parser.add_argument("--output", default="merged_results.json", help="Fichier de sortie")
    parser.add_argument("--no-dedup", dest="dedup", action="store_false", help="Désactive la déduplication par url")
    parser.set_defaults(dedup=True)
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    for p in input_paths:
        if not p.exists():
            raise SystemExit(f"Fichier introuvable: {p}")

    merged = merge_files(input_paths, dedup=args.dedup)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Fusionné {len(input_paths)} fichiers en {out_path} ({len(merged)} enregistrements)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
