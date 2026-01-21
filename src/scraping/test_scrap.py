from page_to_scrap import PageScrap as s
from playwright.sync_api import sync_playwright
from dismiss_overlay import dismiss_overlay
from scrap_tmdb import TMDBScraping
from models import Movie
import json
import re

URL_TO_SCRAP = "https://letterboxd.com/film/the-lord-of-the-rings-the-two-towers/"


def run(playwright):
    chromium = playwright.chromium
    browser = chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.route(
        re.compile(r"(\.png|\.jpg|\.jpeg|\.svg|\.woff|\.css)"), 
        lambda route: route.abort()
    )
    page.route(
        re.compile(r"(google-analytics\.com|googletagmanager\.com)"), 
        lambda route: route.abort()
    )
    page.goto(URL_TO_SCRAP, wait_until="load")
    try: 
        dismiss_overlay(page)
    except Exception as e:
        print(f"Error dismissing overlay: {e}")
    page.wait_for_selector('.fc-consent-root', state='detached', timeout=2000)

    #page.goto("https://letterboxd.com", wait_until="networkidle")

    page_info: dict = {
        "url": URL_TO_SCRAP,
        **s.scrap_cast_page(page),
        **s.scrap_crew_page(page),
        **s.scrap_details_page(page),
        **s.scrap_genres_themes_page(page),
        **s.scrap_tmdb_url(page)
    }
    movie = Movie(**page_info)
    
    context.close()
    browser.close()
    return movie


with sync_playwright() as playwright:
    movie = run(playwright)
    
with open("results.json", "w", encoding="utf-8") as f:
    f.write(movie.model_dump_json(ensure_ascii=False, indent=4))