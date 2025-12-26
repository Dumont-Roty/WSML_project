"""Wrapper pour `WSML_code.dismiss_overlay`.
Expose `dismiss_overlay` comme `WSML_code.services.dismiss.dismiss_overlay`.
"""
try:
    from WSML_code.services.dismiss_impl import dismiss_overlay  # type: ignore
except Exception:  # pragma: no cover
    def dismiss_overlay(*_args, **_kwargs):
        raise RuntimeError("`WSML_code.services.dismiss_impl.dismiss_overlay` introuvable")

__all__ = ["dismiss_overlay"]
