import json
import re
from time import perf_counter
from typing import Iterable, List

from playwright.sync_api import sync_playwright

from WSML_code.services.dismiss_impl import dismiss_overlay
from WSML_code.scrapers.page_scraper import PageScrap
from WSML_code.models import Movie


def scrape_one(context, tmdb_page, url: str) -> Movie:
    start = perf_counter()
    section_timings = {}
    page = context.new_page()
    page.set_default_timeout(3000)
    page.goto(url, wait_until="domcontentloaded", timeout=7000)
    try:
        dismiss_overlay(page)
    except Exception as e:
        print(f"Overlay dismiss error for {url}: {e}")
    page.wait_for_selector('.fc-consent-root', state='detached', timeout=2000)

    t0 = perf_counter();
    cast_block = PageScrap.scrap_cast_page(page)
    section_timings["cast_page"] = perf_counter() - t0

    t0 = perf_counter();
    crew_block = PageScrap.scrap_crew_page(page)
    section_timings["crew_page"] = perf_counter() - t0

    t0 = perf_counter();
    details_block = PageScrap.scrap_details_page(page)
    section_timings["details_page"] = perf_counter() - t0

    t0 = perf_counter();
    genres_block = PageScrap.scrap_genres_themes_page(page)
    section_timings["genres_themes_page"] = perf_counter() - t0

    t0 = perf_counter();
    tmdb_block = PageScrap.scrap_tmdb_url(page, tmdb_page)
    section_timings["tmdb_page"] = perf_counter() - t0

    data = {
        "url": url,
        **cast_block,
        **crew_block,
        **details_block,
        **genres_block,
        **tmdb_block,
    }
    movie = Movie(**data)
    page.close()
    duration = perf_counter() - start
    print(f"[timing] {url} scraped in {duration:.2f}s")
    for section, sec_time in section_timings.items():
        print(f"   [detail] {section}: {sec_time:.2f}s")

    # Affiche la note (rating) scrappée pour visibilité
    rating = data.get("rating")
    if rating is not None:
        print(f"   [rating] {rating}")

    # Affiche budget/revenue TMDB (meme si None)
    tmdb_info_b = data.get("budget")
    tmdb_info_r = data.get("revenue")
    print(f"   [tmdb] budget={tmdb_info_b} revenue={tmdb_info_r}")
    return movie


    # Legacy compatibility list: kept for callers that relied on module-level constant.
URLS_TO_SCRAP: list[str] = [
    "https://letterboxd.com/film/the-lord-of-the-rings-the-two-towers/",
    "https://letterboxd.com/film/the-godfather/",
    "https://letterboxd.com/film/parasite-2019/",
    "https://letterboxd.com/film/spirited-away/",
    "https://letterboxd.com/film/la-haine/",
    "https://letterboxd.com/film/everything-everywhere-all-at-once/",
    "https://letterboxd.com/film/2001-a-space-odyssey/",
    "https://letterboxd.com/film/portrait-of-a-lady-on-fire/",
    "https://letterboxd.com/film/spider-man-into-the-spider-verse/",
    "https://letterboxd.com/film/interstellar/",
    "https://letterboxd.com/film/whiplash-2014/",
]
def _run_with_playwright(urls: Iterable[str]) -> None:
    """Internal helper that runs the Playwright crawling for the provided `urls`.

    Separated from `main` to keep the top-level API simple and testable.
    """
    total_start = perf_counter()
    with sync_playwright() as playwright:
        chromium = playwright.chromium
        browser = chromium.launch(headless=True)
        context = browser.new_context()
        context.route(re.compile(r"(\.png|\.jpg|\.jpeg|\.svg|\.webp|\.gif|\.ico|\.woff|\.css|\.mp4|\.webm)"), lambda route: route.abort())
        context.route(re.compile(r"(google-analytics\.com|googletagmanager\.com|doubleclick\.net|googlesyndication\.com)"), lambda route: route.abort())

        # Page TMDB réutilisée pour éviter le coût d'ouverture à chaque film
        tmdb_page = context.new_page()
        tmdb_page.set_default_timeout(2000)
        tmdb_page.route(re.compile(r"(\.png|\.jpg|\.jpeg|\.svg|\.webp|\.gif|\.ico|\.woff|\.css|\.mp4|\.webm)"), lambda route: route.abort())
        tmdb_page.route(re.compile(r"(doubleclick\.net|googlesyndication\.com)" ) , lambda route: route.abort())

        results = []
        for url in urls:
            movie = scrape_one(context, tmdb_page, url)
            results.append(movie.model_dump())

        tmdb_page.close()
        context.close()
        browser.close()

        total_duration = perf_counter() - total_start
        print(f"[timing] Total run in {total_duration:.2f}s for {len(results)} films")

        with open("results_all.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)


def main(urls: Iterable[str] | None = None) -> None:
    """Run the batch scraping run for the provided `urls` iterable.

    If `urls` is None the function uses `URLS_TO_SCRAP` (legacy behaviour).
    """
    if urls is None:
        urls = URLS_TO_SCRAP
    # Delegate to the Playwright runner
    _run_with_playwright(list(urls))


if __name__ == "__main__":
    # When executed as a script, run the legacy compatibility list by default.
    main(URLS_TO_SCRAP)
