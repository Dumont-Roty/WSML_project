"""Update `fans_favoris` in a JSON dataset by fetching only the fans anchor.

Usage:
  python scripts/update_fans_fast.py --input ml/data/partial_result_2026-01-10.json \
      --output ml/data/partial_result_2026-01-10.fixed.json --workers 8 --threshold 1000

The script updates entries whose current `fans_favoris` is below `threshold` (default 1000).
It performs concurrent HTTP GET requests (requests + BeautifulSoup) and writes a backup.
"""
from __future__ import annotations
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def parse_letterboxd_fans(text: str) -> Optional[int]:
    if not text:
        return None
    s = text.strip().lower().replace('\xa0', ' ').replace(',', '')
    import re
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*([kmb])?\b", s, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        value = float(m.group(1))
    except Exception:
        return None
    suf = (m.group(2) or '').lower()
    if suf == 'k':
        value *= 1_000
    elif suf == 'm':
        value *= 1_000_000
    elif suf == 'b':
        value *= 1_000_000_000
    return int(value)


def fetch_fans_from_url(url: str, session: requests.Session, timeout: int = 10) -> Optional[int]:
    try:
        headers = {"User-Agent": "python-requests/WSML_project (+https://github.com)"}
        r = session.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, 'html.parser')
        # anchor used on Letterboxd pages for fans (matches existing scraper)
        el = soup.select_one('a.all-link.more-link') or soup.select_one("a[href*='/fans/']")
        if not el:
            return None
        return parse_letterboxd_fans(el.get_text())
    except Exception:
        return None


def update_dataset(input_path: str, output_path: str, threshold: int, workers: int, delay: float):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    idxs_to_update = []
    for i, item in enumerate(data):
        cur = item.get('fans_favoris')
        if cur is None or (isinstance(cur, (int, float)) and cur < threshold):
            url = item.get('url')
            if url:
                idxs_to_update.append((i, url))

    if not idxs_to_update:
        print('No entries to update.')
        return

    session = requests.Session()
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_fans_from_url, url, session): idx for idx, url in idxs_to_update}
        for fut in tqdm(as_completed(futures), total=len(futures), desc='fetching'):
            idx = futures[fut]
            try:
                val = fut.result()
            except Exception:
                val = None
            results[idx] = val
            if delay:
                time.sleep(delay)

    updated = 0
    for idx, val in results.items():
        if val is not None:
            data[idx]['fans_favoris'] = val
            updated += 1

    # backup original
    backup_path = input_path + '.bak'
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f'Updated {updated} entries (backup at {backup_path}).')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--threshold', type=int, default=1000, help='Only update entries with fans_favoris < threshold')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--delay', type=float, default=0.0, help='Delay (s) after each fetch to be polite')
    args = p.parse_args()

    update_dataset(args.input, args.output, args.threshold, args.workers, args.delay)


if __name__ == '__main__':
    main()
