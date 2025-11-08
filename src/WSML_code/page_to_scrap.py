from element_to_scrap import Scraping as s
from playwright.sync_api import sync_playwright
from urllib.parse import urljoin


class PageScrap:
    @staticmethod
    def scrap_cast_page(page):
        # Nous sommes déjà sur la page du film : pas besoin de naviguer
        return {
            "title": s.scrap_title(page),
            "date": s.scrap_date(page),
            "director": s.scrap_director(page),
            "casting": s.scrap_casting(page)
        }

    @staticmethod
    def scrap_crew_page(page):
        # trouver le href du lien crew et naviguer vers lui si disponible
        try:
            loc = page.locator("a[href*='/crew/']").first
            href = loc.get_attribute('href')
            if href:
                page.goto(urljoin(page.url, href), wait_until='load')
        except Exception:
            # fallback : essayer un click direct si la navigation échoue
            try:
                page.locator("a[href*='/crew/']").first.click()
            except Exception:
                pass

        return {
            "producers": s.scrap_producers(page),
            "writers": s.scrap_writers(page),
            "composer": s.scrap_composer(page)
        }

    @staticmethod
    def scrap_genres_themes_page(page):
        try:
            loc = page.locator("a[href*='/genres/']").first
            href = loc.get_attribute('href')
            if href:
                page.goto(urljoin(page.url, href), wait_until='load')
        except Exception:
            try:
                page.locator("a[href*='/genres/']").first.click()
            except Exception:
                pass

        return {
            "genres": s.scrap_genres(page),
            "themes": s.scrap_themes(page)
        }