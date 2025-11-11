from element_to_scrap import Scraping as s
from urllib.parse import urljoin


class PageScrap:
    @staticmethod
    def scrap_cast_page(page):
        # Nous sommes déjà sur la page du film : pas besoin de naviguer
        return {
            "title": s.scrap_title(page),
            "date": s.scrap_date(page),
            "director": s.scrap_director(page),
            "casting": s.scrap_casting(page),
            "duration": s.scrap_duree(page),
            "nbr_watched": s.nbr_watched(page),
            "nbr_appearence": s.scrap_appearence(page),
            "nbr_likes": s.scrap_like(page),
            "rating": s.scrap_rate(page),
            "fans favoris": s.scrap_nbr_fan(page)
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
    def scrap_details_page(page):
        try:
            loc = page.locator("a[href*='/details/']").first
            href = loc.get_attribute('href')
            if href:
                page.goto(urljoin(page.url, href), wait_until='load')
        except Exception:
            # fallback : essayer un click direct si la navigation échoue
            try:
                page.locator("a[href*='/details/']").first.click()
            except Exception:
                pass

        return {
            "studio": s.scrap_studios(page),
            "languages": s.scrap_languages(page)
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