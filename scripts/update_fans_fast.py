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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from tqdm import tqdm


def parse_letterboxd_fans(text: str) -> Optional[int]:
    if not text:
        return None
    s = text.strip()
    import re
    m = re.search(r"([0-9]{1,3}(?:[ \u00A0\u202F\u2009\.,][0-9]{3})*(?:[.,][0-9]+)?|[0-9]+(?:[.,][0-9]+)?)\s*([kmb])?",
                  s, flags=re.IGNORECASE)
    if not m:
        return None
    num_str = m.group(1)
    suf = (m.group(2) or '').lower()
    # Normalize numeric string
    if '.' in num_str and ',' in num_str:
        num_str = num_str.replace(',', '')
    elif ',' in num_str and '.' not in num_str:
        num_str = num_str.replace(',', '.')
    num_str = re.sub(r'[ \u00A0\u202F\u2009]', '', num_str)
    try:
        value = float(num_str)
    except Exception:
        return None
    if suf == 'k':
        value *= 1_000
    elif suf == 'm':
        value *= 1_000_000
    elif suf == 'b':
        value *= 1_000_000_000
    return int(value)


def fetch_fans_from_url(url: str, session: requests.Session, timeout: int = 10) -> Optional[int]:
    try:
        # be polite: set a realistic User-Agent and accept headers
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/116.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        r = session.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return None

        text = r.text or ""
        soup = BeautifulSoup(text, 'html.parser')

        # Strategy 1: find the anchor used by the page
        el = soup.select_one('a.all-link.more-link') or soup.select_one("a[href*='/fans/']")
        if el and el.get_text(strip=True):
            v = parse_letterboxd_fans(el.get_text())
            if v is not None:
                return v

        # Strategy 2: search for a nearby link text matching "NNN fans"
        for a in soup.find_all('a'):
            txt = a.get_text(separator=' ', strip=True)
            if txt and 'fans' in txt.lower():
                v = parse_letterboxd_fans(txt)
                if v is not None:
                    return v

        # Strategy 3: regex on full HTML/text (fallback)
        import re
        m = re.search(r"(\d+(?:\.\d{1,3})?)\s*[KkMmBb]?\s*fans", text)
        if m:
            return parse_letterboxd_fans(m.group(0))

        return None
    except Exception:
        return None


def update_dataset(input_path: str, output_path: str, threshold: int, workers: int, delay: float):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # create a backup of the original file before modifying
    backup_path = input_path + '.bak'
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

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

    # Prepare session with retries
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    session.mount('http://', HTTPAdapter(max_retries=retries))
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

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f'Updated {updated} entries (original backed up at {backup_path}).')


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
