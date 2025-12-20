"""Wrapper pour `WSML_code.element_to_scrap`.
Expose la classe `Scraping` sous le nouveau chemin `WSML_code.scrapers.element_scraper`.
Le wrapper utilise un import sécurisé pour ne pas casser l'import si le module source change.
"""
try:
    from WSML_code.element_to_scrap import Scraping  # type: ignore
except Exception:  # pragma: no cover - defensive
    # Fournir un stub minimal pour éviter d'échouer à l'import
    class Scraping:  # type: ignore
        @staticmethod
        def scrap_title(page):
            raise RuntimeError("`WSML_code.element_to_scrap.Scraping` introuvable")

__all__ = ["Scraping"]
