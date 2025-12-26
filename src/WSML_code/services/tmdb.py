"""Wrapper léger pour le module TMDB existant (`WSML_code.scrap_tmdb`).
Expose le module sous `WSML_code.services.tmdb`.
"""
try:
    from WSML_code.services.tmdb_impl import TMDBScraping as TMDBScraping  # type: ignore
except Exception:  # pragma: no cover
    TMDBScraping = None

__all__ = ["TMDBScraping"]
