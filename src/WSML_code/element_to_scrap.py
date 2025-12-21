"""Compatibility wrapper for the old `WSML_code.element_to_scrap` module.

This file exposes `Scraping` from the new location
`WSML_code.scrapers.element_scraper` while providing a small fallback
stub during migration.
"""
from typing import Any

try:
    from WSML_code.scrapers.element_scraper import Scraping  # type: ignore
except Exception:  # pragma: no cover - fallback during migration
    class Scraping:  # type: ignore
        @staticmethod
        def scrap_title(page: Any):
            raise RuntimeError("`WSML_code.scrapers.element_scraper.Scraping` introuvable")

        @staticmethod
        def scrap_directors(page: Any):
            raise RuntimeError("`WSML_code.scrapers.element_scraper.Scraping` introuvable")

        @staticmethod
        def scrap_duree(page: Any):
            raise RuntimeError("`WSML_code.scrapers.element_scraper.Scraping` introuvable")

        @staticmethod
        def nbr_watched(page: Any):
            raise RuntimeError("`WSML_code.scrapers.element_scraper.Scraping` introuvable")

        @staticmethod
        def scrap_appearence(page: Any):
            raise RuntimeError("`WSML_code.scrapers.element_scraper.Scraping` introuvable")

        @staticmethod
        def scrap_like(page: Any):
            raise RuntimeError("`WSML_code.scrapers.element_scraper.Scraping` introuvable")

        @staticmethod
        def scrap_rate(page: Any):
            raise RuntimeError("`WSML_code.scrapers.element_scraper.Scraping` introuvable")

        @staticmethod
        def scrap_nbr_fan(page: Any):
            raise RuntimeError("`WSML_code.scrapers.element_scraper.Scraping` introuvable")

        @staticmethod
        def scrap_casting(page: Any):
            raise RuntimeError("`WSML_code.scrapers.element_scraper.Scraping` introuvable")

        @staticmethod
        def scrap_producers(page: Any):
            raise RuntimeError("`WSML_code.scrapers.element_scraper.Scraping` introuvable")

        @staticmethod
        def scrap_writers(page: Any):
            raise RuntimeError("`WSML_code.scrapers.element_scraper.Scraping` introuvable")

        @staticmethod
        def scrap_composer(page: Any):
            raise RuntimeError("`WSML_code.scrapers.element_scraper.Scraping` introuvable")

        @staticmethod
        def scrap_year(page: Any):
            raise RuntimeError("`WSML_code.scrapers.element_scraper.Scraping` introuvable")

        @staticmethod
        def scrap_studios(page: Any):
            raise RuntimeError("`WSML_code.scrapers.element_scraper.Scraping` introuvable")

        @staticmethod
        def scrap_languages(page: Any):
            raise RuntimeError("`WSML_code.scrapers.element_scraper.Scraping` introuvable")

        @staticmethod
        def scrap_genres(page: Any):
            raise RuntimeError("`WSML_code.scrapers.element_scraper.Scraping` introuvable")

        @staticmethod
        def scrap_themes(page: Any):
            raise RuntimeError("`WSML_code.scrapers.element_scraper.Scraping` introuvable")

__all__ = ["Scraping"]
