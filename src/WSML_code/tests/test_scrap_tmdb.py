from WSML_code.scrap_tmdb import TMDBScraping


class DummyPage:
    def __init__(self, text=""):
        self._text = text

    def wait_for_selector(self, *_args, **_kwargs):
        return None

    def locator(self, *_args, **_kwargs):
        class Loc:
            def __init__(self, text):
                self._text = text

            def inner_text(self):
                return self._text

        return Loc(self._text)

    def inner_text(self, *_args, **_kwargs):
        return self._text


def test_budget_parser_handles_empty_string():
    assert TMDBScraping.scrap_budget(DummyPage()) is None


def test_budget_parser_parses_value():
    page = DummyPage("Budget $12,345,678.00")
    assert TMDBScraping.scrap_budget(page) == 12345678


def test_revenue_parser_handles_empty_string():
    assert TMDBScraping.scrap_revenue(DummyPage()) is None


def test_revenue_parser_parses_value():
    page = DummyPage("Revenue $90,000,000")
    assert TMDBScraping.scrap_revenue(page) == 90000000