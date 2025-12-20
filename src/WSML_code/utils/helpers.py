"""Helpers ré-exportant des fonctions utilitaires existantes si présentes.
Ce wrapper importe en toute sécurité depuis `WSML_code.list_scrap` et fournit
un fallback minimal si nécessaire.
"""
try:
    from WSML_code.list_scrap import detect_kind, _normalize_title  # type: ignore
except Exception:  # pragma: no cover
    import unicodedata

    def _normalize_title(value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value)
        cleaned = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        return cleaned.lower()

    def detect_kind(href: str, title: str | None) -> str:
        href_lower = (href or "").lower()
        normalized_title = _normalize_title(title or "")
        series_keywords = ("series", "serie", "tv")
        if "/series/" in href_lower or any(k in normalized_title for k in series_keywords):
            return "series"
        return "movie"

__all__ = ["detect_kind", "_normalize_title"]
