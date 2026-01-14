"""Récupère budget/revenue TMDB manquants en revisitant les pages Letterboxd.

Lit un JSON de résultats et une liste de films manquants (CSV ou txt). Pour chaque
URL, visite la page Letterboxd et récupère budget/revenue via `PageScrap.scrap_tmdb_url`,
puis écrit un JSON mis à jour.

Usage (PowerShell) depuis la racine du repo :
    $env:PYTHONPATH = 'src'
    .\.venv\Scripts\python .\scripts\fix_missing_in_json.py --input results_parallel.json --missing csv/missing_tmdb.csv --output results_parallel_fixed.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# make sure repo imports work when running from repo root
ROOT = Path.cwd()
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

try:
    from scraping.scrapers.page_scraper import PageScrap
    from scraping.services.dismiss_impl import dismiss_overlay
except Exception as e:
    print("Import error: make sure you run this from the repository root with the project's venv active.")
    raise

from playwright.sync_api import sync_playwright


def load_missing_list(path: Path) -> List[str]:
    """Retourne la liste des URLs à tenter de récupérer.

    Accepte un fichier texte (une URL par ligne) ou un CSV avec une colonne `url`
    et optionnellement `manquant` (on sélectionne les lignes où `manquant` == 'manquant').
    Si le CSV n'a pas de colonne `manquant`, toutes les URLs sont renvoyées.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".txt":
        return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    # CSV path
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    urls: List[str] = []
    key = None
    if rows:
        # find url-like column
        for k in rows[0].keys():
            if k and "url" == k.strip().lower():
                key = k
                break
        if key is None:
            for k in rows[0].keys():
                if k and "url" in k.strip().lower():
                    key = k
                    break

    for r in rows:
        if key is None:
            # try first column value
            vals = list(r.values())
            if not vals:
                continue
            u = vals[0]
        else:
            u = r.get(key, "")
        if not u:
            continue
        u = u.strip()
        manq = r.get("manquant")
        if manq is None:
            urls.append(u)
        else:
            if str(manq).strip().lower() == "manquant":
                urls.append(u)
    return urls


def build_url_map(data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Indexe les enregistrements par URL (ou première URL ressemblant à Letterboxd)."""
    m: Dict[str, Dict[str, Any]] = {}
    for rec in data:
        url = rec.get("url")
        if not url:
            # try find any string that looks like letterboxd film url
            for v in rec.values():
                if isinstance(v, str) and "letterboxd.com/film" in v:
                    url = v
                    break
        if url:
            m[url] = rec
    return m


def fix_missing(
    input_json: Path,
    missing_list: Path,
    output_json: Path,
    headless: bool = True,
    delay: float = 0.2,
    timeout_ms: int = 12000,
):
    """Répare budget/revenue manquants en mettant à jour le JSON de résultats."""
    data = json.loads(input_json.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError("Expected top-level JSON array of film dicts")
    url_map = build_url_map(data)

    targets = load_missing_list(missing_list)
    targets = [t for t in targets if t]
    print(f"Loaded {len(targets)} target URLs to attempt recovery")

    found_count = 0
    visited = 0

    with sync_playwright() as playwright:
        chromium = playwright.chromium
        browser = chromium.launch(headless=headless)
        context = browser.new_context()
        # block heavy assets
        context.route(re.compile(r"(\.(?:png|jpg|jpeg|svg|webp|gif|ico|woff|woff2|css|mp4|webm))"), lambda route: route.abort())
        context.route(re.compile(r"(google-analytics\.com|googletagmanager\.com|doubleclick\.net|googlesyndication\.com)"), lambda route: route.abort())

        tmdb_page = context.new_page()
        tmdb_page.set_default_timeout(10000)
        tmdb_page.route(re.compile(r"(\.(?:png|jpg|jpeg|svg|webp|gif|ico|woff|woff2|css|mp4|webm))"), lambda r: r.abort())

        for url in targets:
            visited += 1
            print(f"[{visited}/{len(targets)}] {url}")
            rec = url_map.get(url)
            if rec is None:
                print("  Warning: URL not present in input JSON, skipping")
                continue
            page = context.new_page()
            page.set_default_timeout(timeout_ms)
            try:
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                except Exception as e:
                    print("  goto failed:", e)
                try:
                    dismiss_overlay(page)
                except Exception:
                    pass

                # Use PageScrap helper to get tmdb values
                try:
                    tm = PageScrap.scrap_tmdb_url(page, tmdb_page)
                except Exception as e:
                    print("  scrap_tmdb_url failed:", e)
                    tm = {"budget": None, "revenue": None}

                b = tm.get("budget")
                r = tm.get("revenue")
                if b is not None or r is not None:
                    # update record in-place
                    if b is not None:
                        rec["budget"] = b
                    if r is not None:
                        rec["revenue"] = r
                    rec["_fixed_by_script"] = True
                    found_count += 1
                    print(f"  Recovered: budget={b!r} revenue={r!r}")
                else:
                    print("  No tmdb values found for this film")

            finally:
                try:
                    page.close()
                except Exception:
                    pass

            if delay:
                time.sleep(delay)

        try:
            tmdb_page.close()
        except Exception:
            pass
        context.close()
        browser.close()

    # write updated JSON
    output_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote updated JSON to {output_json} (found {found_count} updates)")


def main():
    """Parse les arguments CLI et lance `fix_missing`."""
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results_parallel.json")
    p.add_argument("--missing", default="csv/missing_tmdb.csv", help="CSV or txt file listing missing URLs")
    p.add_argument("--output", default="results_parallel_fixed.json")
    p.add_argument("--no-headless", dest="headless", action="store_false")
    p.add_argument("--headless", dest="headless", action="store_true")
    p.add_argument("--delay", type=float, default=0.2)
    p.set_defaults(headless=True)
    args = p.parse_args()

    input_json = Path(args.input)
    missing_list = Path(args.missing)
    output_json = Path(args.output)

    if not input_json.exists():
        print("Input JSON not found:", input_json)
        raise SystemExit(1)
    if not missing_list.exists():
        print("Missing list not found:", missing_list)
        raise SystemExit(1)

    fix_missing(input_json, missing_list, output_json, headless=args.headless, delay=args.delay)


if __name__ == "__main__":
    main()
