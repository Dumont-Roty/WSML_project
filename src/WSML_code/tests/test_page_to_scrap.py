from WSML_code.page_to_scrap import PageScrap
import WSML_code.page_to_scrap as page_module


class DummyLocator:
    def __init__(self, href):
        self._href = href

    @property
    def first(self):
        return self

    def get_attribute(self, _name):
        return self._href


class DummyPage:
    def __init__(self, href=None):
        self._href = href
        self.url = "https://letterboxd.com/film/foobar/"
        self.context = DummyContext()

    def wait_for_selector(self, *_args, **_kwargs):
        return None

    def locator(self, *_args, **_kwargs):
        return DummyLocator(self._href)


class DummyContext:
    def __init__(self):
        self.pages = []

    def new_page(self):
        page = DummyTMDBPage()
        self.pages.append(page)
        return page


class DummyTMDBPage:
    def __init__(self):
        self.visited = []
        self.routes = []
        self.closed = False
        self.timeout = None

    def set_default_timeout(self, value):
        self.timeout = value

    def route(self, pattern, handler):
        self.routes.append(pattern)

    def goto(self, href, **_kwargs):
        self.visited.append(href)

    def close(self):
        self.closed = True


def _patch_tmdb_helpers(monkeypatch):
    monkeypatch.setattr(
        page_module.t,
        "_dismiss_tmdb_cookies",
        lambda *_: None,
    )
    monkeypatch.setattr(page_module.t, "scrap_budget", lambda *_: 11)
    monkeypatch.setattr(page_module.t, "scrap_revenue", lambda *_: 22)


def test_scrap_tmdb_url_returns_none_without_link():
    page = DummyPage()

    assert PageScrap.scrap_tmdb_url(page) == {"budget": None, "revenue": None}


def test_scrap_tmdb_url_uses_provided_tmdb_page(monkeypatch):
    page = DummyPage(href="https://themoviedb.org/movie/1")
    tmdb_page = DummyTMDBPage()
    _patch_tmdb_helpers(monkeypatch)

    result = PageScrap.scrap_tmdb_url(page, tmdb_page=tmdb_page)

    assert result == {"budget": 11, "revenue": 22}
    assert tmdb_page.visited == ["https://themoviedb.org/movie/1"]


def test_scrap_tmdb_url_creates_tmdb_page_when_missing(monkeypatch):
    page = DummyPage(href="https://themoviedb.org/movie/2")
    _patch_tmdb_helpers(monkeypatch)

    result = PageScrap.scrap_tmdb_url(page)

    assert result == {"budget": 11, "revenue": 22}
    assert len(page.context.pages) == 1
    tmdb_page = page.context.pages[0]
    assert tmdb_page.closed
    assert tmdb_page.routes and len(tmdb_page.routes) == 2
    assert tmdb_page.visited == ["https://themoviedb.org/movie/2"]
    assert tmdb_page.timeout == 2000
