import json
import re
from time import perf_counter
from typing import List
from playwright.sync_api import sync_playwright
from page_to_scrap import PageScrap
from dismiss_overlay import dismiss_overlay
from models import Movie

# Liste des URLs à scraper (une par film)
URLS_TO_SCRAP: List[str] = [
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

def scrape_one(context, url: str) -> Movie:
    start = perf_counter()
    section_timings = {}
    page = context.new_page()
    page.goto(url, wait_until="load")
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
    tmdb_block = PageScrap.scrap_tmdb_url(page)
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
    return movie

def main() -> None:
    total_start = perf_counter()
    with sync_playwright() as playwright:
        chromium = playwright.chromium
        browser = chromium.launch(headless=False)
        context = browser.new_context()
        context.route(re.compile(r"(\.png|\.jpg|\.jpeg|\.svg|\.woff|\.css|\.mp4|\.webm)"), lambda route: route.abort())
        context.route(re.compile(r"(google-analytics\.com|googletagmanager\.com|doubleclick\.net|googlesyndication\.com)"), lambda route: route.abort())

        results = []
        for url in URLS_TO_SCRAP:
            movie = scrape_one(context, url)
            results.append(movie.model_dump())

        context.close()
        browser.close()

    total_duration = perf_counter() - total_start
    print(f"[timing] Total run in {total_duration:.2f}s for {len(results)} films")

    with open("results_all.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
