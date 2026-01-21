#!/usr/bin/env python3
"""Fix JSON records where numeric fields were accidentally set to numbers taken from URL slugs.

Heuristic:
- For each record, extract slug segments from the `url` field and collect tokens separated by '-'.
- If any token is purely digits (e.g. '5' in 'final-destination-5') we consider it a slug-number.
- If any of the count fields (`nbr_watched`, `nbr_appearence`, `nbr_likes`) is exactly equal
  to one of these slug-numbers, replace that field with None (null in JSON) and log the change.

Usage:
  python scripts/fix_url_numbers.py --in ml/data/partial_result_2026-01-19.json
    --out ml/data/partial_result_2026-01-19.fixed.json --backup

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import re
import sys

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
import time
import random
import math
from datetime import datetime, timedelta

from playwright.sync_api import sync_playwright


def _ensure_src_on_path() -> None:
    """Make `src/` importable when running this script directly.

    Tests configure `pythonpath = ["src"]`, but ad-hoc script runs usually don't.
    """
    try:
        repo_root = Path(__file__).resolve().parents[1]
        src_dir = repo_root / "src"
        if src_dir.exists() and str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
    except Exception:
        pass


_ensure_src_on_path()

try:
    from scraping.services.dismiss_impl import dismiss_overlay as _lb_dismiss_overlay
    from scraping.scrapers.element_scraper import Scraping as _lb_scraping
except Exception:
    _lb_dismiss_overlay = None
    _lb_scraping = None


def _scrape_counts_with_existing_scraper(page) -> Dict[str, int]:
    """Scrape counts using the same selectors/logic as `src/scraping`.

    Returns only fields that were successfully extracted (values > 0).
    """
    found: Dict[str, int] = {}
    if _lb_scraping is None:
        return found

    try:
        page.wait_for_selector(".production-statistic", timeout=5000)
    except Exception:
        pass

    try:
        watched = int(_lb_scraping.nbr_watched(page) or 0)
        if watched > 0:
            found["nbr_watched"] = watched
    except Exception:
        pass
    try:
        appear = int(_lb_scraping.scrap_appearence(page) or 0)
        if appear > 0:
            found["nbr_appearence"] = appear
    except Exception:
        pass
    try:
        likes = int(_lb_scraping.scrap_like(page) or 0)
        if likes > 0:
            found["nbr_likes"] = likes
    except Exception:
        pass

    return found

def run_with_saved_state(state_path: str, url: str):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(storage_state=state_path)
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=7000)
        # poursuivre le scraping

COUNT_FIELDS = ("nbr_watched", "nbr_appearence", "nbr_likes")


def slug_numbers_from_url(url: str) -> List[int]:
    if not isinstance(url, str):
        return []
    # Remove query and fragment
    url = url.split("?", 1)[0].split("#", 1)[0]
    parts = [p for p in url.split("/") if p]
    nums: List[int] = []
    for seg in parts:
        # consider tokens separated by hyphen as slug words
        for tok in seg.split("-"):
            if tok.isdigit():
                try:
                    nums.append(int(tok))
                except Exception:
                    pass
    return nums


def _parse_int_from_text(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    # find first number with optional K/M/B suffix
    m = re.search(r"([\d\u00A0\u202f\s\.,]+)\s*([KkMmBb])?", text)
    if not m:
        return None
    num_part = m.group(1)
    suffix = (m.group(2) or "").upper()
    # normalize spaces and non-digit separators
    cleaned = num_part.replace("\u202f", "").replace("\u00A0", "").replace(" ", "")
    # remove thousands separators (commas)
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.strip()
    if not cleaned:
        return None
    try:
        val = float(cleaned)
    except Exception:
        return None
    if suffix == "K":
        val *= 1_000
    elif suffix == "M":
        val *= 1_000_000
    elif suffix == "B":
        val *= 1_000_000_000
    return int(val)


def _human_verification_detected(page) -> bool:
    try:
        if page.locator("text=/verify you are human|are you human|captcha|robot/i").count() > 0:
            return True
    except Exception:
        pass
    try:
        if page.locator("iframe[title*='captcha' i]").count() > 0:
            return True
    except Exception:
        pass
    try:
        title = page.title().lower()
        if "verify" in title or "captcha" in title:
            return True
    except Exception:
        pass
    return False


@dataclass
class Config:
    apply_fixes: bool = True
    rescrape: str = "none"  # one of 'none', 'sample', 'all'
    max_rescrape: int = 10
    headless: bool = False
    backup: bool = False
    in_path: Optional[Path] = None
    out_path: Optional[Path] = None
    storage_state: Optional[Path] = None
    save_storage_state: Optional[Path] = None
    # rate limiting / retry config
    min_delay: float = 3.0
    max_delay: float = 6.0
    max_retries: int = 3
    backoff_base: float = 1.0
    respect_retry_after: bool = True
    pause_on_human_check: bool = True
    only_null_counts: bool = False


def _human_delay(cfg: "Config", factor: float = 1.0) -> None:
    """Sleep a human-like randomized interval derived from cfg delays.

    `factor` scales the configured interval so callers can request
    shorter or longer micro-delays (e.g. before a click vs between pages).
    """
    try:
        base = float(cfg.min_delay)
        top = float(cfg.max_delay)
    except Exception:
        base = 0.5
        top = 1.0
    if top < base:
        top = base
    delay = random.uniform(base, top) * float(factor)
    time.sleep(delay)


def fix_records(records: List[Dict[str, Any]], cfg: Config, console: Console) -> Dict[str, int]:
    changed = 0
    fixes = {f: 0 for f in COUNT_FIELDS}
    total = len(records)
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), " ", TimeElapsedColumn(), TimeRemainingColumn()) as prog:
        task = prog.add_task("Scanning records", total=total)
        for rec in records:
            url = rec.get("url") or rec.get("letterboxd_url") or ""
            slug_nums = set(slug_numbers_from_url(url))
            if slug_nums and cfg.apply_fixes:
                for f in COUNT_FIELDS:
                    v = rec.get(f)
                    if isinstance(v, int) and v in slug_nums:
                        rec[f] = None
                        fixes[f] += 1
                        changed += 1
            prog.advance(task)
    console.log(f"Fixes applied: {changed}")
    return {"total_changed": changed, **fixes}


def find_candidates(records: List[Dict[str, Any]], cfg: Config) -> List[int]:
    idxs: List[int] = []
    for i, rec in enumerate(records):
        if cfg.only_null_counts:
            if all(rec.get(f) is None for f in COUNT_FIELDS):
                idxs.append(i)
            continue

        url = rec.get("url") or rec.get("letterboxd_url") or ""
        if slug_numbers_from_url(url):
            # candidate if any count field is missing or None
            if any((rec.get(f) is None or not isinstance(rec.get(f), int)) for f in COUNT_FIELDS):
                idxs.append(i)
    return idxs


def rescrape_records(records: List[Dict[str, Any]], candidate_idxs: List[int], cfg: Config, console: Console) -> Dict[str, int]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        console.print("Playwright not available: cannot rescrape. Install playwright and browsers.")
        return {"rescraped": 0}

    total = len(candidate_idxs)
    if total == 0:
        return {"rescraped": 0}

    rescraped = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=cfg.headless)
        if cfg.storage_state and cfg.storage_state.exists():
            context = browser.new_context(storage_state=str(cfg.storage_state))
        else:
            context = browser.new_context()
        # block heavy resources
        try:
            context.route(re.compile(r"(\.(?:png|jpg|jpeg|svg|webp|gif|ico|woff|css|mp4|webm))"), lambda route: route.abort())
            context.route(re.compile(r"(google-analytics\.com|googletagmanager\.com|doubleclick\.net|googlesyndication\.com)"), lambda route: route.abort())
        except Exception:
            pass

        with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), " ", TimeElapsedColumn(), TimeRemainingColumn()) as prog:
            task = prog.add_task("Rescraping pages", total=total)
            attempted = 0
            for idx in candidate_idxs:
                if cfg.rescrape == "sample" and attempted >= cfg.max_rescrape:
                    break
                rec = records[idx]
                url = rec.get("url") or rec.get("letterboxd_url")
                if not url:
                    prog.advance(task)
                    continue
                attempted += 1
                console.log(f"Rescraping [{idx}] {url}")
                page = None
                try:
                    page = context.new_page()
                    try:
                        page.set_default_timeout(3000)
                    except Exception:
                        pass
                    # small random pause after opening a new page to look more human
                    _human_delay(cfg, factor=0.05)
                    # page-level blocking
                    try:
                        page.route(re.compile(r"(\.(?:png|jpg|jpeg|svg|webp|gif|ico|woff|css|mp4|webm))"), lambda route: route.abort())
                        page.route(re.compile(r"(google-analytics\.com|googletagmanager\.com|doubleclick\.net|googlesyndication\.com)"), lambda route: route.abort())
                    except Exception:
                        pass
                    page.set_default_timeout(90000)
                    # attempts with exponential backoff and respect for 429 Retry-After
                    response = None
                    success = False
                    for attempt in range(cfg.max_retries):
                        try:
                            # tiny jitter before navigation
                            _human_delay(cfg, factor=0.1)
                            response = page.goto(url, timeout=7000, wait_until="domcontentloaded")
                        except Exception as e:
                            console.log(f"goto error attempt {attempt+1} for {url}: {e}")
                            response = None
                        # handle HTTP-level 429
                        status = None
                        try:
                            if response is not None:
                                status = response.status
                        except Exception:
                            status = None
                        if status == 429:
                            # respect Retry-After if available
                            retry_after = None
                            try:
                                hdr = None
                                if response is not None:
                                    hdr = response.headers.get("retry-after")
                                if hdr:
                                    # could be seconds or http-date
                                    if hdr.isdigit():
                                        retry_after = int(hdr)
                                    else:
                                        try:
                                            retry_dt = datetime.strptime(hdr, "%a, %d %b %Y %H:%M:%S GMT")
                                            retry_after = max(0, int((retry_dt - datetime.utcnow()).total_seconds()))
                                        except Exception:
                                            retry_after = None
                            except Exception:
                                retry_after = None
                            if cfg.respect_retry_after and retry_after:
                                wait = retry_after
                            else:
                                wait = cfg.backoff_base * (2 ** attempt) + random.random()
                            console.log(f"Received 429 for {url}, sleeping {wait}s (attempt {attempt+1})")
                            time.sleep(wait)
                            continue
                        # if response is None it's likely a transient navigation error -> backoff and retry
                        if response is None:
                            wait = cfg.backoff_base * (2 ** attempt) + random.random()
                            console.log(f"Transient error navigating {url}, sleeping {wait}s (attempt {attempt+1})")
                            time.sleep(wait)
                            continue
                        # success: break
                        success = True
                        break
                    if not success:
                        console.log(f"Failed to navigate {url} after {cfg.max_retries} attempts")
                        prog.advance(task)
                        _human_delay(cfg)
                        continue

                    # Match the existing scraping pipeline behaviour (overlay/cookie dismissal)
                    try:
                        if _lb_dismiss_overlay is not None:
                            _lb_dismiss_overlay(page)
                    except Exception:
                        pass
                    try:
                        page.wait_for_selector('.fc-consent-root', state='detached', timeout=2000)
                    except Exception:
                        pass
                    # try to dismiss common overlays (cookie banners, modals)
                    overlay_selectors = [
                        '.cookie-disclaimer .dismiss',
                        'button[aria-label="Close"]',
                        '.overlay-dismiss',
                        '.dismiss',
                        '.modal__close',
                        '.overlay__close',
                        'button.close',
                        '.js-dismiss',
                        '#modal-close',
                    ]
                    for sel in overlay_selectors:
                        try:
                            locator = page.locator(sel)
                            if locator.count() and locator.first.is_visible():
                                locator.first.click()
                                break
                        except Exception:
                            pass
                    try:
                        page.wait_for_selector('.fc-consent-root', state='detached', timeout=5000)
                    except Exception:
                        pass
                    if _human_verification_detected(page):
                        console.log("Human verification detected on page.")
                        if cfg.headless:
                            console.log(
                                "Cloudflare verification cannot be solved in headless mode. "
                                "Rerun without --headless, solve it once, then reuse the session with --save-storage-state/--storage-state."
                            )
                            prog.advance(task)
                            _human_delay(cfg)
                            continue
                        if cfg.pause_on_human_check:
                            input("Solve the verification in the browser, then press Enter to continue...")
                        else:
                            console.log("Skipping due to human verification (pause disabled).")
                            prog.advance(task)
                            delay = random.uniform(cfg.min_delay, cfg.max_delay)
                            time.sleep(delay)
                            continue
                    found = _scrape_counts_with_existing_scraper(page)
                    if not found:
                        # Fallback to the previous heuristic if the shared scrapers are not importable
                        try:
                            page.wait_for_selector(
                                ".production-statistic, .filmstat, [data-original-title]",
                                timeout=5000,
                            )
                        except Exception:
                            pass
                        els = page.query_selector_all("[aria-label]")
                        for el in els:
                            al = el.get_attribute("aria-label") or ""
                            if not al:
                                continue
                            txt = al.lower()
                            val = _parse_int_from_text(al)
                            if val is None:
                                continue
                            if "watch" in txt or "member" in txt or "watched by" in txt:
                                found["nbr_watched"] = val
                            elif "appear" in txt or "appears" in txt or "appearance" in txt:
                                found["nbr_appearence"] = val
                            elif "like" in txt or "liked" in txt:
                                found["nbr_likes"] = val
                    # update record if we found values
                    for f, v in found.items():
                        records[idx][f] = v
                    if found:
                        rescraped += 1
                        console.log(f"Result [{idx}] found: {found}")
                    else:
                        try:
                            title = page.title()
                        except Exception:
                            title = "(unknown title)"
                        console.log(f"Result [{idx}] found: {found} | title: {title}")
                except Exception as e:
                    console.log(f"Error scraping [{idx}] {url}: {e}")
                finally:
                    try:
                        if page:
                            page.close()
                    except Exception:
                        pass
                prog.advance(task)
                # human-like delay to avoid transient rate-limits or overlays
                _human_delay(cfg)
        try:
            if cfg.save_storage_state is not None:
                try:
                    cfg.save_storage_state.parent.mkdir(parents=True, exist_ok=True)
                    context.storage_state(path=str(cfg.save_storage_state))
                    console.log(f"Saved storage state to: {cfg.save_storage_state}")
                except Exception as e:
                    console.log(f"Failed to save storage state: {e}")
            context.close()
            browser.close()
        except Exception:
            pass
    console.log(f"Rescraped {rescraped} pages")
    return {"rescraped": rescraped}


def interactive_save_state(url: str, state_path: str = "state.json"):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded")
        console = Console()
        console.print(
            "Fenêtre Playwright ouverte. Navigue librement (tu peux changer de page).\n"
            "Quand Cloudflare apparaît / quand tu es connecté(e) et que tout est OK, sauvegarde l'état." 
        )
        while True:
            cmd = input(
                "Commande: [Entrée]/s = sauvegarder et quitter | c = continuer sans sauvegarder | q = quitter sans sauvegarder\n> "
            ).strip().lower()
            if cmd in ("", "s", "save"):
                context.storage_state(path=state_path)
                console.print(f"Storage state saved to: {state_path}")
                break
            if cmd in ("c", "continue"):
                continue
            if cmd in ("q", "quit", "exit"):
                console.print("Quit without saving storage state")
                break
        context.close()
        browser.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True, help="Input JSON file (list of records)")
    p.add_argument("--out", dest="out_path", required=False, help="Output JSON file (defaults to overwrite input)")
    p.add_argument("--backup", action="store_true", help="Write a backup <input>.bak before overwriting")
    p.add_argument(
        "--apply-fixes",
        dest="apply_fixes",
        action="store_true",
        default=True,
        help="Apply slug-number fixes (default: enabled)",
    )
    p.add_argument(
        "--no-apply-fixes",
        dest="apply_fixes",
        action="store_false",
        help="Disable slug-number fixes",
    )
    p.add_argument(
        "--rescrape",
        choices=("none", "sample", "all"),
        default="none",
        help="Optional Playwright re-scrape for candidates (default: none)",
    )
    p.add_argument(
        "--max-rescrape",
        type=int,
        default=10,
        help="Max pages to rescrape when --rescrape=sample (default: 10)",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run browser in headless mode (default: headful)",
    )
    p.add_argument(
        "--storage-state",
        dest="storage_state",
        required=False,
        help="Path to Playwright storage state JSON to reuse authenticated session",
    )
    p.add_argument(
        "--save-storage-state",
        dest="save_storage_state",
        required=False,
        help="Path to write Playwright storage state JSON after the run (useful after solving Cloudflare/login headful)",
    )
    p.add_argument(
        "--init-storage-state",
        action="store_true",
        default=False,
        help="Open a headful browser to solve Cloudflare/login manually, then save storage state and exit",
    )
    p.add_argument(
        "--init-url",
        default="https://letterboxd.com/",
        help="URL to open when using --init-storage-state (default: https://letterboxd.com/)",
    )
    p.add_argument(
        "--min-delay",
        type=float,
        default=3.0,
        help="Minimum random delay between requests in seconds (default 3.0)",
    )
    p.add_argument(
        "--max-delay",
        type=float,
        default=6.0,
        help="Maximum random delay between requests in seconds (default 6.0)",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per page on transient errors (default 3)",
    )
    p.add_argument(
        "--backoff-base",
        type=float,
        default=1.0,
        help="Base seconds for exponential backoff (default 1.0)",
    )
    p.add_argument(
        "--no-respect-retry-after",
        dest="respect_retry_after",
        action="store_false",
        help="Do not respect Retry-After header when received",
    )
    p.add_argument(
        "--no-pause-on-human-check",
        dest="pause_on_human_check",
        action="store_false",
        help="Do not pause when a human verification page is detected",
    )
    p.add_argument(
        "--only-null-counts",
        action="store_true",
        default=False,
        help="Rescrape candidates only when nbr_watched/nbr_appearence/nbr_likes are all null",
    )
    args = p.parse_args()

    # Manual session mode: lets the user interact with the page without the script navigating away.
    if bool(getattr(args, "init_storage_state", False)):
        state_out = (
            getattr(args, "save_storage_state", None)
            or getattr(args, "storage_state", None)
            or "state.json"
        )
        interactive_save_state(str(getattr(args, "init_url", "https://letterboxd.com/")), state_path=str(state_out))
        raise SystemExit(f"Storage state saved to: {state_out}")

    in_path = Path(getattr(args, "in_path"))
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")
    out_path = Path(args.out_path) if args.out_path else in_path

    data = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("JSON root is not a list of records")

    console = Console()
    cfg = Config(
        apply_fixes=bool(args.apply_fixes),
        rescrape=str(args.rescrape),
        max_rescrape=int(args.max_rescrape),
        headless=bool(args.headless),
        backup=bool(args.backup),
        in_path=in_path,
        out_path=out_path,
        storage_state=Path(args.storage_state) if getattr(args, "storage_state", None) else None,
        save_storage_state=Path(args.save_storage_state) if getattr(args, "save_storage_state", None) else None,
        min_delay=float(args.min_delay),
        max_delay=float(args.max_delay),
        max_retries=int(args.max_retries),
        backoff_base=float(args.backoff_base),
        respect_retry_after=bool(getattr(args, "respect_retry_after", True)),
        pause_on_human_check=bool(getattr(args, "pause_on_human_check", True)),
        only_null_counts=bool(getattr(args, "only_null_counts", False)),
    )

    summary: Dict[str, int] = {}
    if cfg.apply_fixes:
        summary.update(fix_records(data, cfg, console))
    if cfg.rescrape != "none":
        summary.update(rescrape_records(data, find_candidates(data, cfg), cfg, console))

    if cfg.backup and out_path.exists():
        bak = out_path.with_suffix(out_path.suffix + ".bak")
        bak.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")

    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    console.print(f"Wrote {out_path} — summary: {summary}")


if __name__ == "__main__":
    main()
