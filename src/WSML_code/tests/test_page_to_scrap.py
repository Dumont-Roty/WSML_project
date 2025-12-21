from urllib.parse import urljoin

from WSML_code.scrapers.page_scraper import PageScrap
import WSML_code.scrapers.page_scraper as page_module


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
    assert tmdb_page.timeout == 5000


class LinkLocator:
    def __init__(self, href=None):
        self._href = href
        self.clicked = False

    @property
    def first(self):
        return self

    def get_attribute(self, _name):
        return self._href

    def click(self, **_kwargs):
        self.clicked = True


class LinkPage:
    def __init__(self, href_map=None, raise_on_goto=False):
        self.url = "https://letterboxd.com/film/foobar/"
        self.href_map = href_map or {}
        self.goto_calls = []
        self.raise_on_goto = raise_on_goto

    def locator(self, selector, **_kwargs):
        return self.href_map.get(selector, LinkLocator())

    def goto(self, url, **_kwargs):
        self.goto_calls.append(url)
        if self.raise_on_goto:
            raise RuntimeError("nav failed")

    def click(self, selector, **_kwargs):
        raise NotImplementedError


def _patch_scraping(monkeypatch, **values):
    for name, value in values.items():
        monkeypatch.setattr(page_module.s, name, lambda *_args, _value=value: _value)


def test_scrap_cast_page_aggregates_values(monkeypatch):
    _patch_scraping(
        monkeypatch,
        scrap_title="Titre",
        scrap_year="2025",
        scrap_directors=["Dir"],
        scrap_casting=["Acteur"],
        scrap_duree=125,
        nbr_watched=1000,
        scrap_appearence=50,
        scrap_like=200,
        scrap_rate=7.5,
        scrap_nbr_fan=500,
    )
    page = object()

    result = PageScrap.scrap_cast_page(page)

    assert result["title"] == "Titre"
    assert result["year"] == "2025"
    assert result["directors"] == ["Dir"]
    assert result["casting"] == ["Acteur"]
    assert result["duration"] == 125
    assert result["nbr_watched"] == 1000
    assert result["nbr_appearence"] == 50
    assert result["nbr_likes"] == 200
    assert result["rating"] == 7.5
    assert result["fans_favoris"] == 500


def test_scrap_crew_page_navigates_on_href(monkeypatch):
    href = "crew/"
    page = LinkPage(href_map={"a[href*='/crew/']": LinkLocator(href)})
    _patch_scraping(
        monkeypatch,
        scrap_producers=["Prod"],
        scrap_writers=["Writer"],
        scrap_composer=["Comp"],
    )

    result = PageScrap.scrap_crew_page(page)

    assert urljoin(page.url, href) in page.goto_calls
    assert result["producers"] == ["Prod"]
    assert result["writers"] == ["Writer"]
    assert result["composer"] == ["Comp"]


def test_scrap_crew_page_fallback_click(monkeypatch):
    locator = LinkLocator(href="crew/")
    page = LinkPage(href_map={"a[href*='/crew/']": locator}, raise_on_goto=True)
    _patch_scraping(monkeypatch, scrap_producers=[], scrap_writers=[], scrap_composer=[])

    PageScrap.scrap_crew_page(page)

    assert locator.clicked


def test_scrap_details_page_handles_navigation(monkeypatch):
    href = "details/"
    page = LinkPage(href_map={"a[href*='/details/']": LinkLocator(href)})
    _patch_scraping(monkeypatch, scrap_studios=["Studio"], scrap_languages=["Fr"])

    result = PageScrap.scrap_details_page(page)

    assert urljoin(page.url, href) in page.goto_calls
    assert result["studio"] == ["Studio"]
    assert result["languages"] == ["Fr"]


def test_scrap_genres_themes_page(monkeypatch):
    href = "genres/"
    page = LinkPage(href_map={"a[href*='/genres/']": LinkLocator(href)})
    _patch_scraping(monkeypatch, scrap_genres=["Action"], scrap_themes=["Drame"])

    result = PageScrap.scrap_genres_themes_page(page)

    assert urljoin(page.url, href) in page.goto_calls
    assert result["genres"] == ["Action"]
    assert result["themes"] == ["Drame"]
