"""Wrapper pour `WSML_code.list_scrap`.
Expose fonctions de scraping de listes (pagination, collecte de liens).
"""
try:
    from WSML_code.list_scrap import (
        list_scrape,
        list_scrape_parallel,
        collect_film_links,
    )  # type: ignore
except Exception:  # pragma: no cover
    def list_scrape(*_args, **_kwargs):
        raise RuntimeError("`WSML_code.list_scrap.list_scrape` introuvable")

    def list_scrape_parallel(*_args, **_kwargs):
        raise RuntimeError("`WSML_code.list_scrap.list_scrape_parallel` introuvable")

    def collect_film_links(*_args, **_kwargs):
        return []

__all__ = ["list_scrape", "list_scrape_parallel", "collect_film_links"]
