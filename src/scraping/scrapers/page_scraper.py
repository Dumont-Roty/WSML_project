from __future__ import annotations
from urllib.parse import urljoin
import re
import os
from time import perf_counter
from typing import List, Tuple


from scraping.scrapers.element_scraper import Scraping as s
from scraping.services.tmdb_impl import TMDBScraping as t


def safe(call, default=None):
    """Appelle call(), retourne default si exception."""
    try:
        return call()
    except Exception:
        return default


def _tmdb_max_seconds() -> float:
    """Return the max time budget (seconds) for a TMDB scrape.

    Can be overridden via env var `WSML_TMDB_MAX_SECONDS`. Minimum enforced to 3s.
    """
    try:
        return max(3.0, float(os.getenv("WSML_TMDB_MAX_SECONDS", "20")))
    except Exception:
        return 20.0


class PageScrap:
    @staticmethod
    def scrap_cast_page(page):
        return {
            "title": safe(lambda: s.scrap_title(page), None),
            "year": safe(lambda: s.scrap_year(page), None),
            "directors": safe(lambda: s.scrap_directors(page), []),
            "casting": safe(lambda: s.scrap_casting(page), []),
            "duration": safe(lambda: s.scrap_duree(page), None),
            "nbr_watched": safe(lambda: s.nbr_watched(page), None),
            "nbr_appearence": safe(lambda: s.scrap_appearence(page), None),
            "nbr_likes": safe(lambda: s.scrap_like(page), None),
            "rating": safe(lambda: s.scrap_rate(page), None),
            "fans_favoris": safe(lambda: s.scrap_nbr_fan(page), None),
        }

    @staticmethod
    def scrap_crew_page(page):
        try:
            loc = page.locator("a[href*='/crew/']").first
            href = loc.get_attribute('href')
            if href:
                page.goto(urljoin(page.url, href), wait_until='load')
        except Exception:
            try:
                page.locator("a[href*='/crew/']").first.click()
            except Exception:
                pass
        return {
            "producers": safe(lambda: s.scrap_producers(page), []),
            "writers": safe(lambda: s.scrap_writers(page), []),
            "composer": safe(lambda: s.scrap_composer(page), []),
        }

    @staticmethod
    def scrap_details_page(page):
        try:
            loc = page.locator("a[href*='/details/']").first
            href = loc.get_attribute('href')
            if href:
                page.goto(urljoin(page.url, href), wait_until='load')
        except Exception:
            try:
                page.locator("a[href*='/details/']").first.click()
            except Exception:
                pass
        return {
            "studio": safe(lambda: s.scrap_studios(page), []),
            "languages": safe(lambda: s.scrap_languages(page), []),
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
            "genres": safe(lambda: s.scrap_genres(page), []),
            "themes": safe(lambda: s.scrap_themes(page), []),
        }

    @staticmethod
    def scrap_tmdb_url(page, tmdb_page=None):
        """Fetch budget/revenue from the TMDB link if present.

        If a reusable `tmdb_page` is provided, it is used directly; otherwise a
        temporary page is created (and closed). Returns a dict with keys
        `budget` and `revenue`, defaulting to None when missing.
        """
        try:
            page.wait_for_selector("a[href*='themoviedb.org/movie']", timeout=1500)
            href = page.locator("a[href*='themoviedb.org/movie']").first.get_attribute('href')
        except Exception:
            href = None

        if not href:
            return {"budget": None, "revenue": None}

        def _scrape_with(page_obj, deadline=None):
            try:
                page_obj.goto(href, wait_until='domcontentloaded')
            except Exception:
                try:
                    page_obj.goto(href)
                except Exception:
                    return {"budget": None, "revenue": None}

            try:
                t._dismiss_tmdb_cookies(page_obj)
            except Exception:
                pass

            try:
                page_obj.wait_for_timeout(600)
            except Exception:
                pass

            return {
                "budget": t.scrap_budget(page_obj),
                "revenue": t.scrap_revenue(page_obj),
            }

        if tmdb_page is not None:
            return _scrape_with(tmdb_page)

        new_tmdb_page = page.context.new_page()
        try:
            new_tmdb_page.set_default_timeout(5000)
            new_tmdb_page.route(re.compile(r"(\.png|\.jpg|\.jpeg|\.svg|\.webp|\.gif|\.ico|\.woff|\.css|\.mp4|\.webm)"), lambda r: r.abort())
            new_tmdb_page.route(re.compile(r"(doubleclick\.net|googlesyndication\.com)"), lambda r: r.abort())
            result = _scrape_with(new_tmdb_page, deadline=perf_counter() + _tmdb_max_seconds())
        finally:
            new_tmdb_page.close()

        return result

__all__ = ["PageScrap"]
