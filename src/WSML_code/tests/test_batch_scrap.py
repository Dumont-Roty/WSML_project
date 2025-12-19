from typing import Dict

import builtins
import WSML_code.batch_scrap as batch_scrap_module
from WSML_code.batch_scrap import scrape_one
from models import Movie


class DummyPage:
    def __init__(self):
        self.closed = False
        self.timeout = None
        self.goto_calls = []

    def set_default_timeout(self, *_args, **_kwargs):
        self.timeout = _kwargs.get("timeout", 0) or (_args[0] if _args else None)

    def goto(self, url, **_kwargs):
        self.goto_calls.append((url, _kwargs))

    def wait_for_selector(self, *_args, **_kwargs):
        return None

    def close(self):
        self.closed = True


class DummyContext:
    def __init__(self):
        self.pages = []

    def new_page(self):
        page = DummyPage()
        self.pages.append(page)
        return page


class DummyTMDBPage:
    def __init__(self):
        self.closed = False

    def goto(self, *_args, **_kwargs):
        return None

    def close(self):
        self.closed = True


SECTION_VALUE: Dict[str, Dict[str, object]] = {
    "cast": {"title": "Film"},
    "crew": {"producers": ["Prod"]},
    "details": {"studio": ["Studio"], "year": 2024},
    "genres": {"genres": ["Action"]},
    "tmdb": {"budget": 10, "revenue": 20},
}


def register_sections(monkeypatch):
    monkeypatch.setattr("WSML_code.batch_scrap.dismiss_overlay", lambda *_: True)
    monkeypatch.setattr("WSML_code.batch_scrap.PageScrap.scrap_cast_page", lambda *_: SECTION_VALUE["cast"])
    monkeypatch.setattr("WSML_code.batch_scrap.PageScrap.scrap_crew_page", lambda *_: SECTION_VALUE["crew"])
    monkeypatch.setattr("WSML_code.batch_scrap.PageScrap.scrap_details_page", lambda *_: SECTION_VALUE["details"])
    monkeypatch.setattr("WSML_code.batch_scrap.PageScrap.scrap_genres_themes_page", lambda *_: SECTION_VALUE["genres"])
    monkeypatch.setattr("WSML_code.batch_scrap.PageScrap.scrap_tmdb_url", lambda *_: SECTION_VALUE["tmdb"])


def test_scrape_one_builds_movie(monkeypatch):
    register_sections(monkeypatch)
    context = DummyContext()
    tmdb_page = DummyTMDBPage()

    movie = scrape_one(context, tmdb_page, "https://letterboxd.com/film/example/")

    assert isinstance(movie, Movie)
    assert movie.url == "https://letterboxd.com/film/example/"
    assert movie.title == "Film"
    assert movie.producers == ["Prod"]
    assert movie.studio == ["Studio"]
    assert movie.genres == ["Action"]
    assert movie.budget == 10
    assert movie.revenue == 20
    assert context.pages[0].closed is True


def test_scrape_one_handles_overlay_failures(monkeypatch):
    register_sections(monkeypatch)
    def fail(page):
        raise RuntimeError("boom")

    monkeypatch.setattr("WSML_code.batch_scrap.dismiss_overlay", fail)
    context = DummyContext()
    tmdb_page = DummyTMDBPage()

    movie = scrape_one(context, tmdb_page, "https://letterboxd.com/film/the-lord-of-the-rings-the-two-towers/")

    assert movie.title == "Film"
    assert context.pages[0].closed


def test_scrape_one_calls_tmdb_without_reusable_page(monkeypatch):
    register_sections(monkeypatch)
    seen: Dict[str, object] = {}

    def tmdb_stub(page, tmdb_page):
        seen["tmdb_page"] = tmdb_page
        return SECTION_VALUE["tmdb"]

    monkeypatch.setattr("WSML_code.batch_scrap.PageScrap.scrap_tmdb_url", tmdb_stub)

    context = DummyContext()

    movie = scrape_one(context, None, "https://letterboxd.com/film/the-lord-of-the-rings-the-two-towers/")

    assert seen["tmdb_page"] is None
    assert movie.budget == 10
    assert context.pages[0].closed


class DummyFile:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class DummyScrapeResult:
    def __init__(self, url: str):
        self.url = url

    def model_dump(self):
        return {"url": self.url}


class DummyPage:
    def __init__(self):
        self.closed = False
        self.routes = []
        self.timeout = None

    def set_default_timeout(self, value):
        self.timeout = value

    def route(self, *args, **kwargs):
        self.routes.append((args, kwargs))

    def goto(self, *args, **kwargs):
        return None

    def wait_for_selector(self, *args, **kwargs):
        return None

    def close(self):
        self.closed = True


class DummyPageContext:
    def __init__(self):
        self.routes = []
        self.pages = []
        self.closed = False

    def route(self, *args, **kwargs):
        self.routes.append((args, kwargs))

    def new_page(self):
        page = DummyPage()
        self.pages.append(page)
        return page

    def close(self):
        self.closed = True


class DummyBrowser:
    def __init__(self):
        self.contexts = []
        self.closed = False

    def new_context(self):
        context = DummyPageContext()
        self.contexts.append(context)
        return context

    def close(self):
        self.closed = True


class DummyChromium:
    def __init__(self, browser: DummyBrowser):
        self._browser = browser

    def launch(self, *_, **__):
        return self._browser


class DummyPlaywright:
    def __init__(self, browser: DummyBrowser):
        self.chromium = DummyChromium(browser)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_main_runs_with_mocked_playwright(monkeypatch):
    browser = DummyBrowser()
    playwright = DummyPlaywright(browser)
    results: list[str] = []

    def fake_sync_playwright():
        return playwright

    def fake_scrape_one(context, tmdb_page, url):
        page = context.pages[-1]
        page.close()
        results.append(url)
        return DummyScrapeResult(url)

    dumped: dict[str, object] = {}

    def fake_dump(data, *_args, **_kwargs):
        dumped["data"] = data

    monkeypatch.setattr("WSML_code.batch_scrap.sync_playwright", fake_sync_playwright)
    monkeypatch.setattr("WSML_code.batch_scrap.scrape_one", fake_scrape_one)
    monkeypatch.setattr(builtins, "open", lambda *args, **kwargs: DummyFile())
    monkeypatch.setattr("WSML_code.batch_scrap.json.dump", fake_dump)

    batch_scrap_module.main()

    assert len(results) == len(batch_scrap_module.URLS_TO_SCRAP)
    assert dumped["data"] == [{"url": url} for url in batch_scrap_module.URLS_TO_SCRAP]

    main_context = browser.contexts[0]
    assert all(page.closed for page in main_context.pages)
    tmdb_page = main_context.pages[0]
    assert tmdb_page.timeout == 2000
    assert len(main_context.routes) == 2
    assert len(tmdb_page.routes) == 2
    assert browser.closed