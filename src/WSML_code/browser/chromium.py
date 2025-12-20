"""Wrapper d'accès centralisé (léger) pour la configuration Chromium/Playwright.
Ce fichier ne recrée pas la logique existante mais tente d'importer une
fonction utilitaire si elle existe dans `WSML_code.list_scrap` (ou autre).
"""
try:
    # Certains anciens modules peuvent exposer une fabrique de contexte
    from WSML_code.list_scrap import _make_context  # type: ignore
except Exception:  # pragma: no cover
    _make_context = None

__all__ = ["_make_context"]
