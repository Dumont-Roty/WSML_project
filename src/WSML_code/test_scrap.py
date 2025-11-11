from page_to_scrap import PageScrap as s
from playwright.sync_api import sync_playwright
from dismiss_overlay import dismiss_overlay
import json
import re

URL_TO_SCRAP = "https://letterboxd.com/film/the-lord-of-the-rings-the-two-towers/"


def run(playwright):
    chromium = playwright.chromium
    browser = chromium.launch(headless=False)
    page = browser.new_page()
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

    page_info: dict = {'url': URL_TO_SCRAP}
    page_info.update( s.scrap_cast_page(page) )
    page_info.update( s.scrap_crew_page(page) )
    page_info.update( s.scrap_details_page(page) )
    page_info.update( s.scrap_genres_themes_page(page) )

    browser.close()
    return page_info


with sync_playwright() as playwright:
    data = run(playwright)
    
output_file = 'results.json'
with open(output_file, 'w', encoding = "utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)