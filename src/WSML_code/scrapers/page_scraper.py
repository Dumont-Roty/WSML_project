"""Wrapper pour `WSML_code.page_to_scrap`.
Expose `PageScrap` ou les fonctions d'orchestration par-film sous
`WSML_code.scrapers.page_scraper`.
"""
try:
    from WSML_code.page_to_scrap import PageScrap  # type: ignore
except Exception:  # pragma: no cover
    class PageScrap:  # type: ignore
        @staticmethod
        def scrap_tmdb_url(*_args, **_kwargs):
            return {"budget": None, "revenue": None}

__all__ = ["PageScrap"]
