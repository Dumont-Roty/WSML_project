import json
import re
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
    page = context.new_page()
    page.goto(url, wait_until="load")
    try:
        dismiss_overlay(page)
    except Exception as e:
        print(f"Overlay dismiss error for {url}: {e}")
    page.wait_for_selector('.fc-consent-root', state='detached', timeout=2000)

    data = {
        "url": url,
        **PageScrap.scrap_cast_page(page),
        **PageScrap.scrap_crew_page(page),
        **PageScrap.scrap_details_page(page),
        **PageScrap.scrap_genres_themes_page(page),
        **PageScrap.scrap_tmdb_url(page),
    }
    movie = Movie(**data)
    page.close()
    return movie

def main() -> None:
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

    with open("results_all.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
