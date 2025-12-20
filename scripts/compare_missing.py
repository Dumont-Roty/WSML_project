"""
Compare old missing CSV (`missing_tmdb.csv`) with a new scraping JSON result (e.g. `results_2pages.json`).
Prints which URLs from the old file are still missing TMDB budget/revenue, and which ones are now present.

Usage (PowerShell):
  $env:PYTHONPATH = 'src;src/WSML_code'
  .\.venv\Scripts\python .\scripts\compare_missing.py --old missing_tmdb.csv --new results_2pages.json

"""
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional


def find_url(record: Dict[str, Any]) -> Optional[str]:
    # Common key
    if 'url' in record and record['url']:
        return record['url']
    # look for any value that looks like a letterboxd film url
    for v in record.values():
        if isinstance(v, str) and 'letterboxd.com/film' in v:
            return v
    return None


def extract_tmdb_values(record: Dict[str, Any]):
    # Try multiple possible shapes: top-level 'budget'/'revenue', or nested 'tmdb'
    budget = None
    revenue = None
    for k in ('budget','tmdb_budget','tmdbBudget','budget_usd'):
        if k in record and record[k] not in (None, ''):
            budget = record[k]
            break
    for k in ('revenue','tmdb_revenue','tmdbRevenue','revenue_usd'):
        if k in record and record[k] not in (None, ''):
            revenue = record[k]
            break
    # nested
    if (budget is None or revenue is None) and 'tmdb' in record and isinstance(record['tmdb'], dict):
        tm = record['tmdb']
        if budget is None:
            for k in ('budget','budget_usd'):
                if k in tm and tm[k] not in (None, ''):
                    budget = tm[k]
                    break
        if revenue is None:
            for k in ('revenue','revenue_usd'):
                if k in tm and tm[k] not in (None, ''):
                    revenue = tm[k]
                    break
    return budget, revenue


def is_present(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        s = value.strip()
        return s != ''
    return True


def main(old_csv: Path, new_json: Path):
    old = list(csv.DictReader(old_csv.open(encoding='utf-8')))
    old_map = {r['url']: r for r in old}
    old_urls = set(old_map.keys())
    print(f'Old missing rows: {len(old)}')

    new_data = json.loads(new_json.read_text(encoding='utf-8'))
    # Expecting list of film dicts
    new_map = {}
    for rec in new_data:
        url = find_url(rec)
        if not url:
            continue
        new_map[url] = rec
    print(f'New results contain {len(new_map)} recognizable film URLs')

    still_missing = []
    now_present = []
    not_found_in_new = []

    for url in old_urls:
        new_rec = new_map.get(url)
        if not new_rec:
            not_found_in_new.append(url)
            continue
        budget, revenue = extract_tmdb_values(new_rec)
        if is_present(budget) or is_present(revenue):
            now_present.append((url, budget, revenue))
        else:
            still_missing.append(url)

    print('\nSummary:')
    print(f' - Old missing total: {len(old_urls)}')
    print(f' - Present now in new results: {len(now_present)}')
    print(f' - Still missing in new results: {len(still_missing)}')
    print(f' - URLs from old not present in new JSON: {len(not_found_in_new)}')

    if now_present:
        print('\nSample now present (up to 10):')
        for u,b,r in now_present[:10]:
            print(f' - {u}  budget={b!r}  revenue={r!r}')

    if still_missing:
        print('\nSample still missing (up to 10):')
        for u in still_missing[:10]:
            print(' -', u)

    if not_found_in_new:
        print('\nSample not in new JSON (up to 10):')
        for u in not_found_in_new[:10]:
            print(' -', u)

    # Exit code could be useful: 0 if none still missing, else 2
    return 0 if len(still_missing) == 0 else 2


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--old', required=True)
    p.add_argument('--new', required=True)
    args = p.parse_args()
    old_csv = Path(args.old)
    new_json = Path(args.new)
    if not old_csv.exists():
        print('Old file not found:', old_csv)
        raise SystemExit(1)
    if not new_json.exists():
        print('New results file not found:', new_json)
        raise SystemExit(1)
    raise SystemExit(main(old_csv, new_json))
