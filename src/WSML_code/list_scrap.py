"""Compatibility wrapper for the original `WSML_code.list_scrap` module.

This module re-exports the implementation from
`WSML_code.scrapers.list_scraper`. During the migration phase the
wrapper provides a minimal fallback stub if the new module cannot be
imported.
"""

from typing import Any

try:
    from WSML_code.scrapers.list_scraper import (
        list_page_urls,
        list_scrape,
        list_scrape_parallel,
        collect_film_links,
        scrape_list_page,
        _scrape_list_page_isolated,
    )
except Exception:  # pragma: no cover - fallback during migration
    def _missing(*_args, **_kwargs):
        raise RuntimeError("`WSML_code.scrapers.list_scraper` unavailable; migration in progress")

    list_page_urls = _missing
    list_scrape = _missing
    list_scrape_parallel = _missing
    collect_film_links = _missing
    scrape_list_page = _missing
    _scrape_list_page_isolated = _missing

__all__ = [
    "list_page_urls",
    "list_scrape",
    "list_scrape_parallel",
    "collect_film_links",
    "scrape_list_page",
    "_scrape_list_page_isolated",
]
