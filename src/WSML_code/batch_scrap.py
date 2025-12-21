"""Compatibility wrapper for `WSML_code.batch_scrap`.

The migrated implementation lives in `WSML_code.scrapers.batch_scraper`.

This wrapper keeps module-level symbols patchable for unit tests (notably
`tests/test_batch_scrap.py`).
"""

from __future__ import annotations

import importlib
import json

try:
    from playwright.sync_api import sync_playwright  # re-export for tests
except Exception:  # pragma: no cover
    sync_playwright = None  # type: ignore

try:
    from WSML_code.dismiss_overlay import dismiss_overlay  # type: ignore
except Exception:  # pragma: no cover
    def dismiss_overlay(*_args, **_kwargs):
        raise RuntimeError("dismiss_overlay unavailable")

try:
    from WSML_code.page_to_scrap import PageScrap  # type: ignore
except Exception:  # pragma: no cover
    PageScrap = None  # type: ignore


try:
    from WSML_code.scrapers.batch_scraper import scrape_one as _impl_scrape_one, main as _impl_main  # type: ignore
except Exception:  # pragma: no cover
    def _missing(*_args, **_kwargs):
        raise RuntimeError("`WSML_code.scrapers.batch_scraper` unavailable; migration in progress")

    _impl_scrape_one = _missing  # type: ignore
    _impl_main = _missing  # type: ignore


# legacy list kept at wrapper level for compatibility; prefer passing explicit urls
URLS_TO_SCRAP: list[str] = [
    "https://letterboxd.com/film/the-lord-of-the-rings-the-two-towers/",
    "https://letterboxd.com/film/the-godfather/",
    "https://letterboxd.com/film/parasite-2019/",
    "https://letterboxd.com/film/spirited-away/",
    "https://letterboxd.com/film/la-haine/",
    "https://letterboxd.com/film/everything-everywhere-all-at-once/",
    "https://letterboxd.com/film/2001-a-space-odyssey/",
    "https://letterboxd.com/film/portrait-of-a-lady-on-fire/",
    "https://letterboxd.com/film/spider-man-into-the-spider-verse/",
    "https://letterboxd.com/film/interstellar/",
    "https://letterboxd.com/film/whiplash-2014/",
]


def _propagate_hooks():
    impl = importlib.import_module("WSML_code.scrapers.batch_scraper")
    setattr(impl, "sync_playwright", sync_playwright)
    setattr(impl, "json", json)
    setattr(impl, "dismiss_overlay", dismiss_overlay)
    setattr(impl, "PageScrap", PageScrap)
    # allow tests to monkeypatch `WSML_code.batch_scrap.scrape_one`
    setattr(impl, "scrape_one", scrape_one)
    return impl


def _scrape_one_wrapper(context, tmdb_page, url: str):
    # Make sure patched wrapper globals are seen by the implementation.
    impl = _propagate_hooks()
    # Avoid recursion when implementation points back to this wrapper.
    if getattr(impl, "scrape_one", None) is _WRAPPER_SCRAPE_ONE:
        setattr(impl, "scrape_one", _impl_scrape_one)
    return _impl_scrape_one(context, tmdb_page, url)


_WRAPPER_SCRAPE_ONE = _scrape_one_wrapper
scrape_one = _scrape_one_wrapper


def main():
    """Compatibility entrypoint: call migrated implementation with legacy URL list."""
    impl = _propagate_hooks()
    if getattr(impl, "scrape_one", None) is _WRAPPER_SCRAPE_ONE:
        setattr(impl, "scrape_one", _impl_scrape_one)
    return _impl_main(URLS_TO_SCRAP)


__all__ = [
    "scrape_one",
    "main",
    "URLS_TO_SCRAP",
    "sync_playwright",
    "dismiss_overlay",
    "PageScrap",
    "json",
]
