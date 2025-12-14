from element_to_scrap import Scraping as s
from scrap_tmdb import TMDBScraping as t
from urllib.parse import urljoin
import re


class PageScrap:
    @staticmethod
    def scrap_cast_page(page):
        # Nous sommes déjà sur la page du film : pas besoin de naviguer
        return {
            "title": s.scrap_title(page),
            "year": s.scrap_year(page),
            "directors": s.scrap_directors(page),
            "casting": s.scrap_casting(page),
            "duration": s.scrap_duree(page),
            "nbr_watched": s.nbr_watched(page),
            "nbr_appearence": s.scrap_appearence(page),
            "nbr_likes": s.scrap_like(page),
            "rating": s.scrap_rate(page),
            "fans_favoris": s.scrap_nbr_fan(page)
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
        
    @staticmethod
    def scrap_tmdb_url(page):
        # Récupère le lien TMDB, ouvre une nouvelle page dans le même contexte, bloque CSS/images, scrape budget/revenue
        try:
            page.wait_for_selector("a[href*='themoviedb.org/movie']", timeout=2000)
            href = page.locator("a[href*='themoviedb.org/movie']").first.get_attribute('href')
        except Exception:
            href = None

        if not href:
            return {"budget": None, "revenue": None}

        tmdb_page = page.context.new_page()
        tmdb_page.set_default_timeout(2500)
        tmdb_page.route(re.compile(r"(\.png|\.jpg|\.jpeg|\.svg|\.woff|\.css|\.mp4|\.webm)"), lambda r: r.abort())
        tmdb_page.route(re.compile(r"(doubleclick\.net|googlesyndication\.com)"), lambda r: r.abort())
        try:
            tmdb_page.goto(href, wait_until='domcontentloaded', timeout=3000)
            t._dismiss_tmdb_cookies(tmdb_page)

            budget = t.scrap_budget(tmdb_page)
            revenue = t.scrap_revenue(tmdb_page)

            if budget is None or revenue is None:
                try:
                    t._dismiss_tmdb_cookies(tmdb_page)
                except Exception:
                    pass
                if budget is None:
                    budget = t.scrap_budget(tmdb_page)
                if revenue is None:
                    revenue = t.scrap_revenue(tmdb_page)

            return {
                "budget": budget,
                "revenue": revenue
            }
        finally:
            tmdb_page.close()