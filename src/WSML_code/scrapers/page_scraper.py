from __future__ import annotations
from urllib.parse import urljoin
import re
from typing import List, Tuple

from WSML_code.scrapers.element_scraper import Scraping as s
from WSML_code.services.tmdb_impl import TMDBScraping as t


class PageScrap:
    @staticmethod
    def scrap_cast_page(page):
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
            "fans_favoris": s.scrap_nbr_fan(page),
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
            "producers": s.scrap_producers(page),
            "writers": s.scrap_writers(page),
            "composer": s.scrap_composer(page),
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
            "studio": s.scrap_studios(page),
            "languages": s.scrap_languages(page),
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
            "themes": s.scrap_themes(page),
        }

    @staticmethod
    def scrap_tmdb_url(page, tmdb_page=None):
        try:
            page.wait_for_selector("a[href*='themoviedb.org/movie']", timeout=1500)
            href = page.locator("a[href*='themoviedb.org/movie']").first.get_attribute('href')
        except Exception:
            href = None

        if not href:
            return {"budget": None, "revenue": None}

        new_tmdb_page = None
        if tmdb_page is None:
            new_tmdb_page = page.context.new_page()
            tmdb_page = new_tmdb_page
            # Keep unit-test expectation for the newly created TMDB page.
            tmdb_page.set_default_timeout(5000)
            tmdb_page.route(re.compile(r"(\.png|\.jpg|\.jpeg|\.svg|\.webp|\.gif|\.ico|\.woff|\.css|\.mp4|\.webm)"), lambda r: r.abort())
            tmdb_page.route(re.compile(r"(doubleclick\.net|googlesyndication\.com)"), lambda r: r.abort())

        try:
            budget = None
            revenue = None

            # Robust retry loop: TMDB can be slow or render values after initial load.
            for timeout in (5000, 8000, 12000):
                try:
                    tmdb_page.goto(href, wait_until='domcontentloaded', timeout=timeout)
                except Exception:
                    # Fallback to full load if domcontentloaded is flaky.
                    tmdb_page.goto(href, wait_until='load', timeout=timeout)

                try:
                    t._dismiss_tmdb_cookies(tmdb_page)
                except Exception:
                    pass

                if budget is None:
                    budget = t.scrap_budget(tmdb_page)
                if revenue is None:
                    revenue = t.scrap_revenue(tmdb_page)

                if budget is not None and revenue is not None:
                    break

                # One extra re-fetch attempt for missing fields.
                if budget is None or revenue is None:
                    try:
                        tmdb_page.goto(href, wait_until='load', timeout=timeout)
                        try:
                            t._dismiss_tmdb_cookies(tmdb_page)
                        except Exception:
                            pass
                    except Exception:
                        pass
                    if budget is None:
                        budget = t.scrap_budget(tmdb_page)
                    if revenue is None:
                        revenue = t.scrap_revenue(tmdb_page)

                if budget is not None and revenue is not None:
                    break

            return {"budget": budget, "revenue": revenue}
        except Exception:
            return {"budget": None, "revenue": None}
        finally:
            if new_tmdb_page:
                tmdb_page.close()

__all__ = ["PageScrap"]
