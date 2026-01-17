"""Compat shim exposing `TMDBScraping` from the new package layout."""
try:
    from scraping.services.tmdb_impl import TMDBScraping as TMDBScraping  # type: ignore
except Exception:  # pragma: no cover
    TMDBScraping = None

__all__ = ["TMDBScraping"]
