"""Wrapper pour `scraping.dismiss_overlay`.
Expose `dismiss_overlay` comme `scraping.services.dismiss.dismiss_overlay`.
"""
try:
    from scraping.services.dismiss_impl import dismiss_overlay  # type: ignore
except Exception:  # pragma: no cover
    def dismiss_overlay(*_args, **_kwargs):
        raise RuntimeError("`scraping.services.dismiss_impl.dismiss_overlay` introuvable")

__all__ = ["dismiss_overlay"]
