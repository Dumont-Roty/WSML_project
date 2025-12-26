"""Corrige les notes manquantes en revisitant les pages des films.

Charge un JSON de résultats, identifie les enregistrements sans note (ou note 0),
et peut revisiter chaque page pour récupérer la note via `Scraping.scrap_rate`.

Usage (PowerShell) :
    $env:PYTHONPATH='src;src/WSML_code'
    .\.venv\Scripts\python .\scripts\fix_missing_rate.py --input results_parallel.json --output results_parallel_rates_fixed.json --dry-run

Oter `--dry-run` pour lancer réellement les visites Playwright.
"""
from pathlib import Path
import argparse
import json
import sys
import time
from typing import Optional

# Ensure local package imports when executed from repo root
ROOT = Path.cwd()
SRC = ROOT / 'src'
SRC_WSML = SRC / 'WSML_code'
sys.path.insert(0, str(SRC_WSML))
sys.path.insert(0, str(SRC))

from WSML_code.scrapers.element_scraper import Scraping
from WSML_code.services.dismiss_impl import dismiss_overlay
from playwright.sync_api import sync_playwright, Page


def find_missing(data):
    """Retourne les (index, enregistrement) dont la note est absente ou vaut 0."""
    targets = []
    for i, rec in enumerate(data):
        r = rec.get('rating')
        if r is None:
            targets.append((i, rec))
            continue
        try:
            # Accept numbers; some pipelines may have 0.0 as placeholder
            val = float(r)
        except Exception:
            targets.append((i, rec))
            continue
        if val == 0.0:
            targets.append((i, rec))
    return targets


def run(input_path: Path, output_path: Path, headless: bool = True, delay: float = 0.2, dry_run: bool = True):
    """Optionnellement revisite les films à note manquante pour récupérer la note."""
    data = json.loads(input_path.read_text(encoding='utf-8'))
    targets = find_missing(data)
    print(f'Loaded {len(data)} records; {len(targets)} targets with missing/zero ratings')

    if dry_run:
        # Print a few sample URLs and exit
        for idx, rec in targets[:10]:
            print(f' - [{idx}] {rec.get("title")!s} {rec.get("year")!s} -> {rec.get("url")!s} (rating={rec.get("rating")!r})')
        print('Dry-run mode; no browser started. Rerun without --dry-run to perform fixes.')
        return 0

    updated = 0
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        context = browser.new_context()
        # block heavy resources
        import re
        context.route(re.compile(r"(\.(?:png|jpg|jpeg|svg|webp|gif|ico|woff|woff2|css|mp4|webm))"), lambda r: r.abort())
        context.route(re.compile(r"(google-analytics\.com|googletagmanager\.com|doubleclick\.net|googlesyndication\.com)"), lambda r: r.abort())

        for idx, rec in targets:
            url = rec.get('url')
            print(f'[{idx}] Visiting {url}')
            page: Optional[Page] = None
            try:
                # create page first, then validate URL to avoid calling goto(None)
                page = context.new_page()
                page.set_default_timeout(8000)
                if not isinstance(url, str):
                    print('  Skipping record without valid url')
                    continue
                page.goto(url, wait_until='domcontentloaded', timeout=10000)
                try:
                    dismiss_overlay(page)
                except Exception:
                    pass
                # reuse Scraping.scrap_rate
                new_rate: Optional[float] = Scraping.scrap_rate(page)
                if new_rate is not None:
                    data[idx]['rating'] = new_rate
                    data[idx]['_fixed_rating'] = True
                    updated += 1
                    print(f'  -> recovered rating={new_rate}')
                else:
                    print('  -> rating still missing')
            except Exception as e:
                print('  Error visiting page:', e)
            finally:
                if page is not None:
                    try:
                        page.close()
                    except Exception:
                        pass
            if delay:
                time.sleep(delay)

        try:
            context.close()
        except Exception:
            pass
        try:
            browser.close()
        except Exception:
            pass

    # write output
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {len(data)} records to {output_path} (updated {updated} ratings)')
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='results_parallel.json')
    p.add_argument('--output', default='results_parallel_rates_fixed.json')
    p.add_argument('--no-headless', dest='headless', action='store_false')
    p.add_argument('--headless', dest='headless', action='store_true')
    p.add_argument('--delay', type=float, default=0.2)
    p.add_argument('--dry-run', dest='dry_run', action='store_true')
    p.add_argument('--run', dest='dry_run', action='store_false')
    p.set_defaults(headless=True, dry_run=True)
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        print('Input file not found:', in_path)
        raise SystemExit(1)

    raise SystemExit(run(in_path, out_path, headless=args.headless, delay=args.delay, dry_run=args.dry_run))
