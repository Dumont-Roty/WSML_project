"""Wrapper léger pour le module TMDB existant (`WSML_code.scrap_tmdb`).
Expose le module sous `WSML_code.services.tmdb`.
"""
try:
    from WSML_code import scrap_tmdb as scrap_tmdb  # type: ignore
except Exception:  # pragma: no cover
    scrap_tmdb = None

__all__ = ["scrap_tmdb"]
