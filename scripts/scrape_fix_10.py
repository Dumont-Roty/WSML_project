#!/usr/bin/env python3
"""Scrape up to 10 films (with numeric slug tokens) and repopulate COUNT_FIELDS.

Writes output to a new file with suffix `.patched.json` and prints a summary.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.scraping.scrapers.element_scraper import Scraping

from playwright.sync_api import sync_playwright

COUNT_FIELDS = ("nbr_watched", "nbr_appearence", "nbr_likes")


def has_slug_number(url: str) -> bool:
    if not isinstance(url, str):
        return False
    parts = [p for p in url.split("/") if p]
    for seg in parts:
        for tok in seg.split("-"):
            if tok.isdigit():
                return True
    return False


def full_url(u: str) -> str:
    if not isinstance(u, str):
        return ""
    if u.startswith("http"):
        return u
    # assume letterboxd
    return f"https://letterboxd.com{u}" if u.startswith("/") else f"https://letterboxd.com/{u}"


def main():
    in_path = ROOT / "ml" / "data" / "partial_result_2026-01-19.fixed.json"
    if not in_path.exists():
        print("Input fixed JSON not found:", in_path)
        raise SystemExit(1)

    data = json.loads(in_path.read_text(encoding="utf-8"))
    # select candidates: slug contains number and any COUNT_FIELDS is None
    candidates = []
    for rec in data:
        url = rec.get("url") or rec.get("letterboxd_url") or ""
        if has_slug_number(url):
            # if any field is None, we want to scrape
            needs = any(rec.get(f) is None for f in COUNT_FIELDS)
            if needs:
                candidates.append(rec)
    if not candidates:
        print("No candidate records found.")
        return

    to_process = candidates[:10]
    print(f"Processing {len(to_process)} records...")

    results = {"success": 0, "failed": 0}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        for rec in tqdm(to_process, desc="Scraping films"):
            url = rec.get("url") or rec.get("letterboxd_url") or ""
            u = full_url(url)
            try:
                page.goto(u, timeout=15000, wait_until='domcontentloaded')
                time.sleep(0.5)
                # call Scraping functions which expect Playwright Page
                try:
                    w = Scraping.nbr_watched(page)
                    a = Scraping.scrap_appearence(page)
                    l = Scraping.scrap_like(page)
                except Exception as e:
                    print(f"Error scraping {u}: {e}")
                    results["failed"] += 1
                    continue
                # update record only if we obtained non-zero values
                if isinstance(w, int) and w > 0:
                    rec["nbr_watched"] = w
                if isinstance(a, int) and a > 0:
                    rec["nbr_appearence"] = a
                if isinstance(l, int) and l > 0:
                    rec["nbr_likes"] = l
                results["success"] += 1
            except Exception as e:
                print(f"Failed to load {u}: {e}")
                results["failed"] += 1

        try:
            context.close()
            browser.close()
        except Exception:
            pass

    out_path = in_path.with_name(in_path.stem + ".patched.json")
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Done. summary: {results}. Output: {out_path}")


if __name__ == "__main__":
    main()
