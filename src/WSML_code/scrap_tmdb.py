try:
    from WSML_code.services.tmdb_impl import TMDBScraping as TMDBScraping  # type: ignore
except Exception:  # pragma: no cover - fallback
    class TMDBScraping:  # type: ignore
        @staticmethod
        def _dismiss_tmdb_cookies(*_args, **_kwargs):
            return None

        @staticmethod
        def scrap_budget(*_args, **_kwargs):
            return None

        @staticmethod
        def scrap_revenue(*_args, **_kwargs):
            return None