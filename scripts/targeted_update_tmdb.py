"""Met à jour de façon ciblée budget/revenue TMDB à partir d'un CSV de manquants.

Lit `missing_tmdb.csv`, revisite chaque URL et écrit `missing_tmdb_updated.csv` avec
les nouveaux budgets/revenues et un statut. Conçu pour être lancé depuis la racine du
repo avec le virtualenv actif.

Usage (PowerShell) :
    $env:PYTHONPATH = 'src;src/WSML_code'
    .\.venv\Scripts\python .\scripts\targeted_update_tmdb.py

Ou après activation du venv :
    python .\scripts\targeted_update_tmdb.py --input missing_tmdb.csv --output missing_tmdb_updated.csv

Options :
    --headless / --no-headless : exécution Playwright headless (défaut headless)
    --delay SECONDS : délai optionnel entre pages (défaut 0.2)
"""
import sys
from pathlib import Path
import time
import csv
import argparse
from collections import Counter

# Ensure project modules are importable when running from repo root
ROOT = Path.cwd()
SRC = ROOT / 'src'
SRC_WSML = SRC / 'WSML_code'
sys.path.insert(0, str(SRC_WSML))
sys.path.insert(0, str(SRC))

try:
    from WSML_code.scrapers.page_scraper import PageScrap
    from WSML_code.services.dismiss_impl import dismiss_overlay
except Exception as e:
    print('Import error:', e)
    print('Make sure you run this from the repository root and the venv is available.')
    raise

from playwright.sync_api import sync_playwright


def run(input_path: Path, output_path: Path, headless: bool = True, delay: float = 0.2):
    """Revisite les URLs du CSV et écrit un CSV mis à jour avec budget/revenue/status."""
    rows = list(csv.DictReader(input_path.open(encoding='utf-8')))
    print(f'Loaded {len(rows)} rows from {input_path}')

    updated = []
    start = time.perf_counter()

    with sync_playwright() as playwright:
        chromium = playwright.chromium
        browser = chromium.launch(headless=headless)
        context = browser.new_context()
        import re
        # Block heavy resources
        context.route(re.compile(r"(\.(?:png|jpg|jpeg|svg|webp|gif|ico|woff|woff2|css|mp4|webm))"), lambda route: route.abort())
        context.route(re.compile(r"(google-analytics\.com|googletagmanager\.com|doubleclick\.net|googlesyndication\.com)"), lambda route: route.abort())

        tmdb_page = context.new_page()
        tmdb_page.set_default_timeout(10000)
        tmdb_page.route(re.compile(r"(\.(?:png|jpg|jpeg|svg|webp|gif|ico|woff|woff2|css|mp4|webm))"), lambda r: r.abort())
        tmdb_page.route(re.compile(r"(doubleclick\.net|googlesyndication\.com)"), lambda r: r.abort())

        i = 0
        total = len(rows)
        for r in rows:
            i += 1
            url = r.get('url')
            title = r.get('title') or ''
            year = r.get('year') or ''
            print(f'[{i}/{total}] {title} ({year}) -> {url}')

            page = context.new_page()
            page.set_default_timeout(12000)
            new_budget = None
            new_revenue = None
            status = 'missing'
            try:
                try:
                    page.goto(url, wait_until='domcontentloaded', timeout=12000)
                except Exception as e:
                    print('  goto failed:', e)
                try:
                    dismiss_overlay(page)
                except Exception:
                    pass
                try:
                    res = PageScrap.scrap_tmdb_url(page, tmdb_page=tmdb_page)
                    new_budget = res.get('budget')
                    new_revenue = res.get('revenue')
                    if (new_budget is not None) or (new_revenue is not None):
                        status = 'ok'
                    else:
                        status = 'missing'
                except Exception as e:
                    print('  scrap_tmdb_url error:', e)
                    status = 'error'
            finally:
                try:
                    page.close()
                except Exception:
                    pass

            updated.append({
                'url': url,
                'title': title,
                'year': year,
                'source_page': r.get('source_page',''),
                'old_budget': r.get('budget',''),
                'old_revenue': r.get('revenue',''),
                'new_budget': new_budget if new_budget is not None else '',
                'new_revenue': new_revenue if new_revenue is not None else '',
                'status': status,
            })

            if delay and i < total:
                time.sleep(delay)

        try:
            tmdb_page.close()
        except Exception:
            pass
        context.close()
        browser.close()

    # Write CSV
    fieldnames = ['url','title','year','source_page','old_budget','old_revenue','new_budget','new_revenue','status']
    with output_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in updated:
            writer.writerow(row)

    duration = time.perf_counter() - start
    print(f'Wrote {len(updated)} rows to {output_path} in {duration:.1f}s')
    cnt = Counter(r['status'] for r in updated)
    print('Status counts:', dict(cnt))
    found = [r for r in updated if r['new_budget'] or r['new_revenue']]
    print(f'Found {len(found)} entries with at least one TMDB value')
    if found:
        print('Sample found (up to 10):')
        for x in found[:10]:
            print(' -', x['title'], x['year'], 'new_budget=', x['new_budget'], 'new_revenue=', x['new_revenue'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='missing_tmdb.csv')
    parser.add_argument('--output', default='missing_tmdb_updated.csv')
    parser.add_argument('--no-headless', dest='headless', action='store_false')
    parser.add_argument('--headless', dest='headless', action='store_true')
    parser.add_argument('--delay', type=float, default=0.2)
    parser.set_defaults(headless=True)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f'Input file {in_path} not found. Exit.')
        raise SystemExit(1)

    run(in_path, out_path, headless=args.headless, delay=args.delay)