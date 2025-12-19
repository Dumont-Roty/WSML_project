from WSML_code.scrap_tmdb import TMDBScraping


class TMDBPageStub:
    def __init__(self, selectors=None, body_text=""):
        self.selectors = selectors or {}
        self.body_text = body_text

    class Loc:
        def __init__(self, text):
            self.text = text

        @property
        def first(self):
            return self

        def inner_text(self):
            return self.text

    def wait_for_selector(self, selector, **_kwargs):
        if selector not in self.selectors:
            raise Exception("missing")
        value = self.selectors.get(selector)
        if value is None:
            raise Exception("empty")
        return None

    def locator(self, selector, **_kwargs):
        return TMDBPageStub.Loc(self.selectors.get(selector, ""))

    def inner_text(self, *_args, **_kwargs):
        return self.body_text


class CookiePageStub:
    def __init__(self, fail_selectors):
        self.fail_selectors = fail_selectors
        self.clicked = []

    class Locator:
        def __init__(self, parent, selector):
            self.parent = parent
            self.selector = selector

        @property
        def first(self):
            return self

        def click(self, **_kwargs):
            if self.selector in self.parent.fail_selectors:
                raise Exception("boom")
            self.parent.clicked.append(self.selector)

    def locator(self, selector, **_kwargs):
        return CookiePageStub.Locator(self, selector)


def test_dismiss_tmdb_cookies_stops_on_first_success():
    page = CookiePageStub(fail_selectors={
        "button:has-text('Tout refuser')",
        "button:has-text('Reject all')",
    })

    TMDBScraping._dismiss_tmdb_cookies(page)

    assert len(page.clicked) == 1


def test_scrap_budget_falls_back_to_body():
    page = TMDBPageStub(body_text="Budget $5,000")

    assert TMDBScraping.scrap_budget(page) == 5000


def test_scrap_budget_returns_none_on_invalid_body(monkeypatch):
    page = TMDBPageStub(body_text="Budget ABC")

    assert TMDBScraping.scrap_budget(page) is None


def test_scrap_revenue_falls_back_to_body():
    page = TMDBPageStub(body_text="Revenue $7,500")

    assert TMDBScraping.scrap_revenue(page) == 7500


def test_scrap_revenue_returns_none_when_missing():
    page = TMDBPageStub()

    assert TMDBScraping.scrap_revenue(page) is None