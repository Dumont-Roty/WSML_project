from playwright.sync_api import sync_playwright
import json

def test_json(response, results):
    try: results.append(
        {
            'url': response.url,
            'json': response.json()
        }
    )
    except:
        pass

def run(playwright):
    results = []
    chromium = playwright.chromium
    browser = chromium.launch(headless=False)
    page = browser.new_page()
    page.on("response", lambda response: test_json(response, results)) # A chaque réponse à une requete réseau, execute la fonction test_json
    #page.goto("https://letterboxd.com/film/the-lord-of-the-rings-the-two-towers/")
    page.goto("https://letterboxd.com")
    browser.close()
    return results


with sync_playwright() as playwright:
    data = run(playwright)
    with open('results.json', 'w') as f:
        json.dump(data, f)