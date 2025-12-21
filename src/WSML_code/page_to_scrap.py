"""Compatibility wrapper for `WSML_code.page_to_scrap`.

The migrated implementation lives in `WSML_code.scrapers.page_scraper`.

This wrapper intentionally keeps module-level names `s` and `t` so unit tests
can monkeypatch them (see `tests/test_page_to_scrap.py`).
"""

from __future__ import annotations

import importlib

try:
    from WSML_code.scrapers.element_scraper import Scraping as s  # type: ignore
    from WSML_code.services.tmdb_impl import TMDBScraping as t  # type: ignore
except Exception:  # pragma: no cover - fallback during migration
    from element_to_scrap import Scraping as s  # type: ignore
    from scrap_tmdb import TMDBScraping as t  # type: ignore

_impl = importlib.import_module("WSML_code.scrapers.page_scraper")


def _sync_impl_deps() -> None:
    # The implementation module references `s` and `t` as globals.
    # Keep them in sync so tests can monkeypatch `WSML_code.page_to_scrap.s/t`.
    setattr(_impl, "s", s)
    setattr(_impl, "t", t)


class PageScrap:
    @staticmethod
    def scrap_cast_page(page):
        _sync_impl_deps()
        return _impl.PageScrap.scrap_cast_page(page)

    @staticmethod
    def scrap_crew_page(page):
        _sync_impl_deps()
        return _impl.PageScrap.scrap_crew_page(page)

    @staticmethod
    def scrap_details_page(page):
        _sync_impl_deps()
        return _impl.PageScrap.scrap_details_page(page)

    @staticmethod
    def scrap_genres_themes_page(page):
        _sync_impl_deps()
        return _impl.PageScrap.scrap_genres_themes_page(page)

    @staticmethod
    def scrap_tmdb_url(page, tmdb_page=None):
        _sync_impl_deps()
        return _impl.PageScrap.scrap_tmdb_url(page, tmdb_page=tmdb_page)


__all__ = ["PageScrap", "s", "t"]