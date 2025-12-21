try:
	# importer depuis le nouveau service si disponible
	from WSML_code.services.dismiss_impl import dismiss_overlay  # type: ignore
except Exception:  # pragma: no cover - fallback
	def dismiss_overlay(page):
		raise RuntimeError("dismiss_overlay unavailable: migration in progress")

