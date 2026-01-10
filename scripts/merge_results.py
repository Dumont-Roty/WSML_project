"""Fusionne des exports JSON Letterboxd en un seul fichier.

Objectif
- Centraliser l'outil de fusion des JSON (pour agrandir la base de données).
- Permettre une fusion automatique (sans devoir lister tous les fichiers à la main).
- Conserver l'option de déduplication par `url`.

Entrées supportées
- Chaque fichier peut contenir une liste JSON (cas courant) OU un objet JSON unique.

Auto-détection (si aucun input n'est fourni)
- `results_*.json` à la racine du repo
- `results/*.json` dans le dossier `results/`

Sortie
- Par défaut: `ml/data/partial_result_<date>.json`
  (ce dossier est ignoré par git via `ml/.gitignore`).

Usage (PowerShell)
    ./.venv/Scripts/python ./scripts/merge_results.py

    # Fusion explicite de fichiers
    ./.venv/Scripts/python ./scripts/merge_results.py results_51_70.json results_71_80.json --output ml/data/partial_result_2026-01-10.json

Options
    --dedup              Active la déduplication par `url` (active par défaut)
    --no-dedup           Désactive la déduplication par `url`
    --include-final28    Ajoute aussi `ml/data/final_results_28.json` si présent
    --report-duplicates  Écrit un rapport des doublons (url + réalisateurs + année)
"""
from __future__ import annotations

import argparse
import json
import datetime as _dt
from pathlib import Path
from typing import Any, Iterable, List


def _repo_root_from_here() -> Path:
    # scripts/ -> repo root
    return Path(__file__).resolve().parents[1]


def _discover_inputs(repo_root: Path) -> list[Path]:
    root_candidates = sorted(repo_root.glob("results_*.json"))
    results_dir = repo_root / "results"
    dir_candidates: list[Path] = []
    if results_dir.exists():
        dir_candidates = sorted(results_dir.glob("*.json"))
    paths = [*root_candidates, *dir_candidates]
    return [p for p in paths if p.is_file()]


def load_json_records(path: Path) -> List[Any]:
    """Charge un fichier JSON (liste OU objet unique).

    - Si le fichier contient une liste, elle est retournée telle quelle.
    - Si le fichier contient un objet JSON, il est encapsulé dans une liste.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Le fichier {path} ne contient ni liste ni objet JSON")


def _as_list_of_str(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[str] = []
        for x in value:
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    # fallback
    s = str(value).strip()
    return [s] if s else []


def _to_int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return int(value)
        return int(float(value))
    except Exception:
        return None


def _to_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def normalize_movie_record(rec: dict[str, Any]) -> dict[str, Any] | None:
    """Normalise un enregistrement pour correspondre au schéma `Movie`.

    Champs requis: url, title, year.
    Les autres champs sont optionnels et normalisés (listes, int/float, None).
    """
    url = rec.get("url")
    title = rec.get("title")
    year = rec.get("year")

    if not isinstance(url, str) or not url.strip():
        return None
    if not isinstance(title, str) or not title.strip():
        return None
    y = _to_int_or_none(year)
    if y is None:
        return None

    out: dict[str, Any] = {
        "url": url.strip(),
        "title": title.strip(),
        "year": int(y),
        "directors": _as_list_of_str(rec.get("directors")),
        "casting": _as_list_of_str(rec.get("casting")),
        "duration": _to_int_or_none(rec.get("duration")),
        "nbr_watched": _to_int_or_none(rec.get("nbr_watched")),
        "nbr_appearence": _to_int_or_none(rec.get("nbr_appearence")),
        "nbr_likes": _to_int_or_none(rec.get("nbr_likes")),
        "rating": _to_float_or_none(rec.get("rating")),
        "fans_favoris": _to_int_or_none(rec.get("fans_favoris")),
        "producers": _as_list_of_str(rec.get("producers")),
        "writers": _as_list_of_str(rec.get("writers")),
        "composer": _as_list_of_str(rec.get("composer")),
        "studio": _as_list_of_str(rec.get("studio")),
        "languages": _as_list_of_str(rec.get("languages")),
        "genres": _as_list_of_str(rec.get("genres")),
        "themes": _as_list_of_str(rec.get("themes")),
        "budget": _to_int_or_none(rec.get("budget")),
        "revenue": _to_int_or_none(rec.get("revenue")),
    }
    return out


def merge_files(paths: list[Path], *, dedup: bool) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
    merged: list[dict[str, Any]] = []

    # Dedup key is URL by design (Letterboxd film page).
    seen_urls: set[str] = set()

    # For reporting: keep enough info to audit duplicates.
    dup_rows: list[dict[str, Any]] = []
    n_total = 0
    n_kept = 0
    n_invalid = 0

    for path in paths:
        records = load_json_records(path)
        n_total += len(records)
        for rec in records:
            if not isinstance(rec, dict):
                n_invalid += 1
                continue

            norm = normalize_movie_record(rec)
            if norm is None:
                n_invalid += 1
                continue

            url = norm.get("url")
            if dedup and isinstance(url, str) and url:
                if url in seen_urls:
                    dup_rows.append(
                        {
                            "url": url,
                            "year": norm.get("year"),
                            "directors": norm.get("directors"),
                            "title": norm.get("title"),
                            "source_file": str(path),
                        }
                    )
                    continue
                seen_urls.add(url)

            merged.append(norm)
            n_kept += 1

    stats = {
        "files": len(paths),
        "records_total": n_total,
        "records_kept": n_kept,
        "records_invalid_skipped": n_invalid,
        "records_dropped_as_dupe": len(dup_rows) if dedup else 0,
        "unique_urls": len(seen_urls) if dedup else 0,
    }
    return merged, stats, dup_rows


def main() -> int:
    repo_root = _repo_root_from_here()

    parser = argparse.ArgumentParser(description="Fusionne des fichiers JSON de résultats")
    parser.add_argument("inputs", nargs="*", help="Chemins des fichiers JSON à fusionner (optionnel)")
    parser.add_argument(
        "--output",
        default=None,
        help="Fichier de sortie (défaut: ml/data/partial_result_<date>.json)",
    )

    dedup_group = parser.add_mutually_exclusive_group()
    dedup_group.add_argument(
        "--dedup",
        dest="dedup",
        action="store_true",
        help="Active la déduplication par url (déjà actif par défaut)",
    )
    dedup_group.add_argument(
        "--no-dedup",
        dest="dedup",
        action="store_false",
        help="Désactive la déduplication par url (par défaut: active)",
    )
    parser.set_defaults(dedup=True)

    parser.add_argument(
        "--include-final28",
        action="store_true",
        help="Ajoute aussi ml/data/final_results_28.json si présent",
    )
    parser.add_argument(
        "--report-duplicates",
        default=None,
        help="Si la déduplication est active, écrit un rapport des doublons (url + réalisateurs + année)",
    )
    args = parser.parse_args()

    today = _dt.date.today().isoformat()
    default_out = repo_root / "ml" / "data" / f"partial_result_{today}.json"
    out_path = Path(args.output) if args.output else default_out

    if args.inputs:
        input_paths = [Path(p) for p in args.inputs]
    else:
        input_paths = _discover_inputs(repo_root)

    if args.include_final28:
        p = repo_root / "ml" / "data" / "final_results_28.json"
        if p.exists():
            input_paths.append(p)

    input_paths = [p for p in input_paths if p.exists()]
    if not input_paths:
        raise SystemExit("Aucun fichier JSON d'entrée trouvé (patterns: results_*.json, results/*.json)")

    merged, stats, dup_rows = merge_files(list(input_paths), dedup=bool(args.dedup))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Entrées: {len(input_paths)} fichier(s)")
    print(
        f"Records: {stats['records_total']} -> {stats['records_kept']} (dedup={'on' if args.dedup else 'off'}) "
        f"| invalides ignorés: {stats.get('records_invalid_skipped', 0)}"
    )
    if args.dedup:
        print(f"Doublons url ignorés: {stats['records_dropped_as_dupe']}")
        if args.report_duplicates:
            dup_path = Path(str(args.report_duplicates))
            # TSV simple (lisible dans Excel) : url \t year \t directors \t title \t source_file
            lines = ["url\tyear\tdirectors\ttitle\tsource_file"]
            for row in dup_rows:
                url = str(row.get("url") or "")
                year = str(row.get("year") or "")
                directors = row.get("directors")
                if isinstance(directors, list):
                    directors_s = ", ".join([str(x) for x in directors])
                else:
                    directors_s = str(directors or "")
                title = str(row.get("title") or "")
                src = str(row.get("source_file") or "")
                lines.append(f"{url}\t{year}\t{directors_s}\t{title}\t{src}")
            dup_path.parent.mkdir(parents=True, exist_ok=True)
            dup_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"Rapport doublons: {dup_path}")
    print(f"Écrit: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
