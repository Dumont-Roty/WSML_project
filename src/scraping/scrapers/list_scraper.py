from __future__ import annotations

import json
import re
import unicodedata
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from time import perf_counter
from typing import Iterable, List, Tuple
from urllib.parse import urljoin

from playwright._impl._errors import TimeoutError
from playwright.sync_api import sync_playwright

from scraping.scrapers.batch_scraper import scrape_one
from scraping.services.dismiss_impl import dismiss_overlay

LIST_BASE_DOMAIN = "https://letterboxd.com"
LIST_BASE_PATH = "/films/popular/"
LIST_CONTAINER_SELECTOR = "#films-browser-list-container"
LIST_GRID_SELECTOR = "ul.poster-list.-p70.-grid"
LIST_FRAME_SELECTOR = f"{LIST_GRID_SELECTOR} a.frame"
LIST_PAGE_SELECTOR = "div.pagination a.next"
LIST_SCROLL_ROUNDS = 12
LIST_SCROLL_DELAY_MS = 300
EXPECTED_FRAMES_PER_PAGE = 72
LIST_PAGE_GOTO_TIMEOUT = 15000
LIST_CONTAINER_TIMEOUT = 15000
LIST_FRAME_TIMEOUT = 15000
LIST_NETWORK_IDLE_TIMEOUT = 12000


def list_page_urls(max_pages: int) -> Iterable[str]:
    """Yield the pagination URLs up to ``max_pages``."""
    for page_index in range(1, max_pages + 1):
        if page_index == 1:
            yield f"{LIST_BASE_DOMAIN}{LIST_BASE_PATH}"
        else:
            yield f"{LIST_BASE_DOMAIN}{LIST_BASE_PATH}page/{page_index}/"


def _normalize_title(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    cleaned = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return cleaned.lower()


def detect_kind(href: str, title: str | None) -> str:
    """Return ``series`` when URL or title hints at a show, else ``movie``."""
    href_lower = href.lower()
    normalized_title = _normalize_title(title or "")
    series_keywords = ("series", "serie", "tv")
    if "/series/" in href_lower or any(keyword in normalized_title for keyword in series_keywords):
        return "series"
    return "movie"


def _make_context(playwright, *, headless: bool = True):
    """Create a fresh browser/context/tmdb_page trio with resource blocking."""
    chromium = playwright.chromium
    browser = chromium.launch(headless=headless)
    context = browser.new_context()
    context.route(re.compile(r"(\.(?:png|jpg|jpeg|svg|webp|gif|ico|woff|css|mp4|webm))"), lambda route: route.abort())
    context.route(re.compile(r"(google-analytics\.com|googletagmanager\.com|doubleclick\.net|googlesyndication\.com)"), lambda route: route.abort())

    tmdb_page = context.new_page()
    tmdb_page.set_default_timeout(2000)
    tmdb_page.route(re.compile(r"(\.(?:png|jpg|jpeg|svg|webp|gif|ico|woff|css|mp4|webm))"), lambda route: route.abort())
    tmdb_page.route(re.compile(r"(doubleclick\.net|googlesyndication\.com)"), lambda route: route.abort())

    return browser, context, tmdb_page


class ProgressCounter:
    """Thread-safe progress counter for films scraped in parallel runs.

    - call `incr(n=1)` each time a film is successfully scraped
    - read `value()` to get the current count
    """

    def __init__(self, total_estimate: int = 0):
        self._lock = threading.Lock()
        self._count = 0
        self.total_estimate = total_estimate

    def incr(self, n: int = 1) -> None:
        with self._lock:
            self._count += int(n)

    def value(self) -> int:
        with self._lock:
            return int(self._count)


def collect_film_links(page) -> List[Tuple[str, str]]:
    """Collect every film link from the popular grid with its inferred kind."""
    page.wait_for_load_state("domcontentloaded", timeout=LIST_PAGE_GOTO_TIMEOUT)
    try:
        page.wait_for_selector(LIST_CONTAINER_SELECTOR, timeout=LIST_CONTAINER_TIMEOUT)
        page.wait_for_selector(LIST_FRAME_SELECTOR, timeout=LIST_FRAME_TIMEOUT)
    except TimeoutError:
        print("[warn] Grid selector not found in time; skipping page")
        return []

    def _load_full_grid() -> int:
        """Scroll to trigger lazy loading until the frame count stops growing."""
        last_count = 0
        for _ in range(LIST_SCROLL_ROUNDS):
            page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
            page.wait_for_timeout(LIST_SCROLL_DELAY_MS)
            current_count = page.locator(LIST_FRAME_SELECTOR).count()
            if current_count <= last_count:
                break
            last_count = current_count
        return last_count

    _load_full_grid()
    frames = page.locator(LIST_FRAME_SELECTOR).all()
    frame_count = len(frames)

    # If scroll didn't load the expected number, try the AJAX fragment endpoint
    if frame_count < EXPECTED_FRAMES_PER_PAGE:
        try:
            container = page.query_selector(LIST_CONTAINER_SELECTOR)
            if container:
                data_url = container.get_attribute("data-url")
                if data_url:
                    ajax_url = urljoin(LIST_BASE_DOMAIN, data_url)
                    try:
                        temp = page.context.new_page()
                        temp.set_default_timeout(5000)
                        temp.goto(ajax_url, wait_until="domcontentloaded", timeout=8000)
                        try:
                            temp.wait_for_selector(LIST_FRAME_SELECTOR, timeout=4000)
                            ajax_frames = temp.locator(LIST_FRAME_SELECTOR).all()
                            if len(ajax_frames) > frame_count:
                                frames = ajax_frames
                        except Exception:
                            # selector not present in AJAX response; ignore
                            pass
                        finally:
                            try:
                                temp.close()
                            except Exception:
                                pass
                    except Exception:
                        # network/goto failed; fall back to whatever we have
                        pass
        except Exception:
            pass
    films: List[Tuple[str, str]] = []
    for frame in frames:
        href = frame.get_attribute("href")
        if not href:
            continue
        url = urljoin(LIST_BASE_DOMAIN, href)
        title = frame.get_attribute("data-original-title")
        films.append((url, detect_kind(href, title)))
    return films


def find_next_list_page(page) -> str | None:
    """Return the absolute URL of the "next" pagination link, if present."""
    next_link = page.query_selector(LIST_PAGE_SELECTOR)
    if not next_link:
        return None
    href = next_link.get_attribute("href")
    if not href:
        return None
    return urljoin(LIST_BASE_DOMAIN, href)


def scrape_list_page(context, tmdb_page, list_page_url: str, progress: ProgressCounter | None = None) -> Tuple[List[dict], str | None]:
    """Visit a list page, collect film URLs, and return results with next URL."""
    page = context.new_page()
    page.set_default_timeout(4000)
    try:
        page.goto(list_page_url, wait_until="domcontentloaded", timeout=LIST_PAGE_GOTO_TIMEOUT)
    except TimeoutError as exc:
        print(f"[warn] List page load timed out for {list_page_url}: {exc}")
        if hasattr(page, "close"):
            page.close()
        return [], None
    try:
        page.wait_for_load_state("networkidle", timeout=LIST_NETWORK_IDLE_TIMEOUT)
    except TimeoutError:
        pass
    try:
        dismiss_overlay(page)
    except Exception:
        pass

    results: List[dict] = []
    for film_url, kind in collect_film_links(page):
        try:
            movie = scrape_one(context, tmdb_page, film_url)
        except Exception as exc:
            print(f"[warn] Failed to scrape film {film_url} on list page {list_page_url}: {exc}")
            # continue scraping other films from this page
            continue
        try:
            data = movie.model_dump()
        except Exception as exc:
            print(f"[warn] Failed to serialize movie {film_url} on list page {list_page_url}: {exc}")
            continue
        data["kind"] = kind
        data["source_page"] = list_page_url
        results.append(data)
        # update shared progress counter if provided (thread-safe)
        try:
            if progress is not None:
                progress.incr(1)
        except Exception:
            # don't let progress errors stop scraping
            pass

    next_page = find_next_list_page(page)

    if hasattr(page, "close"):
        page.close()
    return results, next_page


def _scrape_list_page_isolated(list_page_url: str, *, headless: bool = True, progress: ProgressCounter | None = None) -> List[dict]:
    """Scrape a single list page in its own browser context (for parallel runs)."""
    with sync_playwright() as playwright:
        browser, context, tmdb_page = _make_context(playwright, headless=headless)
        try:
            page_results, _ = scrape_list_page(context, tmdb_page, list_page_url, progress=progress)
            return page_results
        finally:
            try:
                tmdb_page.close()
            finally:
                try:
                    context.close()
                finally:
                    browser.close()


# Nombre de grille à scraper
def list_scrape(
    max_pages: int = 2,
    output_path: str = "results_all.json",
    *,
    start_page: int = 1,
    end_page: int | None = None,
) -> None:
    """Sequential scrape with optional `start_page`/`end_page` (1-based, inclusive).

    If `end_page` is None it defaults to `max_pages`. Pages outside [1, max_pages]
    are clipped. Results are written to `output_path`.
    """
    total_start = perf_counter()
    # Normalize start/end
    if start_page < 1:
        start_page = 1
    if end_page is None:
        end_page = max_pages
    else:
        end_page = min(end_page, max_pages)
    if start_page > end_page:
        print(f"[warn] start_page ({start_page}) > end_page ({end_page}); nothing to do")
        return

    # Build the page URL slice to process
    all_urls = list(list_page_urls(max_pages))
    urls = all_urls[start_page - 1 : end_page]

    with sync_playwright() as playwright:
        chromium = playwright.chromium
        browser = chromium.launch(headless=True)
        context = browser.new_context()
        context.route(re.compile(r"(\.(?:png|jpg|jpeg|svg|webp|gif|ico|woff|css|mp4|webm))"), lambda route: route.abort())
        context.route(re.compile(r"(google-analytics\.com|googletagmanager\.com|doubleclick\.net|googlesyndication\.com)"), lambda route: route.abort())

        tmdb_page = context.new_page()
        tmdb_page.set_default_timeout(2000)
        tmdb_page.route(re.compile(r"(\.(?:png|jpg|jpeg|svg|webp|gif|ico|woff|css|mp4|webm))"), lambda route: route.abort())
        tmdb_page.route(re.compile(r"(doubleclick\.net|googlesyndication\.com)"), lambda route: route.abort())

        results: List[dict] = []
        for url in urls:
            page_results, _ = scrape_list_page(context, tmdb_page, url)
            results.extend(page_results)

        tmdb_page.close()
        context.close()
        browser.close()

    total_duration = perf_counter() - total_start
    print(f"[timing] Total list scrape in {total_duration:.2f}s for {len(results)} films")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

# Nombre de grille à scraper en parallèle
def list_scrape_parallel(
    max_pages: int = 2,
    output_path: str = "results_all.json",
    *,
    headless: bool = True,
    workers: int = 2,
    preserve_page_order: bool = False,
    start_page: int = 1,
    end_page: int | None = None,
) -> None:
    """Scrape multiple list pages concurrently (one browser context per page).

    Concurrency is bounded by ``workers`` to avoid overloading the machine.
    Each list page is scraped exactly once; results are aggregated and written
    to ``output_path``.
    """
    total_start = perf_counter()
    # preserve order from list_page_urls, remove duplicates while keeping order
    all_urls = list(dict.fromkeys(list_page_urls(max_pages)))
    # Normalize start/end bounds (1-based inclusive)
    if start_page < 1:
        start_page = 1
    if end_page is None:
        end_page = max_pages
    else:
        end_page = min(end_page, max_pages)
    if start_page > end_page:
        print(f"[warn] start_page ({start_page}) > end_page ({end_page}); nothing to do")
        return
    urls = all_urls[start_page - 1 : end_page]
    url_to_index = {url: idx for idx, url in enumerate(urls)}
    if not urls:
        print("[warn] No list pages to scrape")
        return

    results: List[dict] = []
    worker_count = max(1, min(workers, len(urls)))
    total_pages = len(urls)
    expected_total_frames = total_pages * EXPECTED_FRAMES_PER_PAGE
    completed_pages = 0
    progress = ProgressCounter(total_estimate=expected_total_frames)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        # map future -> (index, url) so we can re-order if requested
        futures = {
            executor.submit(_scrape_list_page_isolated, url, headless=headless, progress=progress): (url_to_index[url], url)
            for url in urls
        }

        if preserve_page_order:
            results_by_index: dict[int, List[dict]] = {}
            for future in as_completed(futures):
                idx, url = futures[future]
                try:
                    page_results = future.result()
                    results_by_index[idx] = page_results
                except Exception as exc:
                    print(f"[warn] List page scrape failed for {url}: {exc}")
                finally:
                    completed_pages += 1
                    remaining_pages = total_pages - completed_pages
                    scraped_count = progress.value()
                    films_remaining = max(0, expected_total_frames - scraped_count)
                    print(
                        f"[progress] scraped: {scraped_count}/{expected_total_frames} ({completed_pages}/{total_pages} pages). "
                        f"remaining pages: {remaining_pages}, films remaining (est.): {films_remaining}"
                    )
            # extend results in page order
            for idx in sorted(results_by_index.keys()):
                results.extend(results_by_index[idx])
        else:
            for future in as_completed(futures):
                idx, url = futures[future]
                try:
                    page_results = future.result()
                    results.extend(page_results)
                except Exception as exc:
                    print(f"[warn] List page scrape failed for {url}: {exc}")
                finally:
                    completed_pages += 1
                    remaining_pages = total_pages - completed_pages
                    scraped_count = progress.value()
                    films_remaining = max(0, expected_total_frames - scraped_count)
                    print(
                        f"[progress] scraped: {scraped_count}/{expected_total_frames} ({completed_pages}/{total_pages} pages). "
                        f"remaining pages: {remaining_pages}, films remaining (est.): {films_remaining}"
                    )

    total_duration = perf_counter() - total_start
    print(f"[timing] Total parallel list scrape in {total_duration:.2f}s for {len(results)} films")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    list_scrape(max_pages=1)

__all__ = [
    "list_page_urls",
    "detect_kind",
    "collect_film_links",
    "find_next_list_page",
    "scrape_list_page",
    "list_scrape",
    "list_scrape_parallel",
]
