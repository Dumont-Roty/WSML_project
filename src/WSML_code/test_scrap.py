from element_to_scrap import scrap_title, scrap_casting, sync_playwright
import json

URL_TO_SCRAP = "https://letterboxd.com/film/the-lord-of-the-rings-the-two-towers/"


def run(playwright):
    chromium = playwright.chromium
    browser = chromium.launch(headless=False)
    page = browser.new_page()
    page.goto(URL_TO_SCRAP, wait_until="load")
    #page.goto("https://letterboxd.com", wait_until="networkidle")
     
    page_info = {
        'url': URL_TO_SCRAP,
        'title': scrap_title(page),
        'casting': scrap_casting(page)
    }
    
    browser.close()
    return page_info


with sync_playwright() as playwright:
    data = run(playwright)
    
    
output_file = 'results.json'
with open(output_file, 'w', encoding = "utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)