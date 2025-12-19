from pathlib import Path

content = """from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from WSML_code import list_scrap


class DummyFrame:
    def __init__(self, href: str, title: str | None = None) -> None:
        self._href = href
        self._title = title

    def get_attribute(self, name: str) -> str | None:
        if name == "href":
            return self._href
        if name == "data-original-title":
            return self._title
        return None


class DummyLocator:
    def __init__(self, frames: list[DummyFrame]) -> None:
        self._frames = frames

    def all(self) -> list[DummyFrame]:
        return self._frames


class DummyListPage:
    def __init__(self, frames: list[DummyFrame] | None = None) -> None:
        self.frames = frames or []
        self.closed = False
        self.goto_calls: list[str] = []

    def wait_for_selector(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def locator(self, *_args: Any, **_kwargs: Any) -> DummyLocator:
        return DummyLocator(self.frames)

    def set_default_timeout(self, value: int) -> None:
        self.timeout = value

    def goto(self, url: str, **_kwargs: Any) -> None:
        self.goto_calls.append(url)

    def close(self) -> None:
        self.closed = True


class DummyContext:
    def __init__(self) -> None:
        self.pages: list[DummyListPage] = []

    def new_page(self) -> DummyListPage:
        page = DummyListPage()
        self.pages.append(page)
        return page


class DummyMovie:
    def __init__(self, dataset: dict) -> None:
        self.dataset = dataset

    def model_dump(self) -> dict:
        return dict(self.dataset)


class DummyTMDBPage(DummyListPage):
    def __init__(self) -> None:
        super().__init__()
        self.routes: list[str] = []

    def route(self, pattern: Any, handler: Any) -> None:
        self.routes.append(str(pattern))


class DummyPageForScrape(DummyListPage):
    def __init__(self) -> None:
        super().__init__()
        self.routing: list[str] = []

    def route(self, pattern: Any, handler: Any) -> None:
        self.routing.append(str(pattern))


class DummyContextManager:
    def __init__(self) -> None:
        self.routes: list[str] = []
        self.pages: list[Any] = []
        self._tmdb_created = False

    def route(self, pattern: Any, handler: Any) -> None:
        self.routes.append(str(pattern))

    def new_page(self) -> Any:
        if not self._tmdb_created:
            self._tmdb_created = True
            page = DummyTMDBPage()
        else:
            page = DummyPageForScrape()
        self.pages.append(page)
        return page

    def close(self) -> None:
        self.closed = True


class DummyBrowser:
    def __init__(self) -> None:
        self.context = DummyContextManager()
        self.closed = False

    def new_context(self) -> DummyContextManager:
        return self.context

    def close(self) -> None:
        self.closed = True


class DummyChromium:
    def __init__(self) -> None:
        self._browser: DummyBrowser | None = None

    def launch(self, headless: bool = True) -> DummyBrowser:
        self.headless = headless
        self._browser = DummyBrowser()
        return self._browser


class DummyPlaywright:
    def __init__(self) -> None:
        self.chromium = DummyChromium()

    def __enter__(self) -> "DummyPlaywright":
        return self

    def __exit__(self, *_: Any) -> None:
        return None


@pytest.mark.parametrize(
    "max_pages,expected",
    [
        (1, ["https://letterboxd.com/films/popular/"])
    ],
)
def test_list_page_urls_generates_first_page(max_pages: int, expected: list[str]) -> None:
    assert list(list_scrap.list_page_urls(max_pages)) == expected


def test_detect_kind_flags_series_when_needed() -> None:
    assert list_scrap.detect_kind("/series/1", "Une série") == "series"
    assert list_scrap.detect_kind("/film/2", "Mini-série") == "series"
    assert list_scrap.detect_kind("/film/3", "Film") == "movie"


def test_collect_film_links_builds_urls_and_kind() -> None:
    frames = [
        DummyFrame("/film/foo", "Le Film"),
        DummyFrame("/series/bar", "La Série"),
    ]
    page = DummyListPage(frames)

    result = list_scrap.collect_film_links(page)

    assert result == [
        ("https://letterboxd.com/film/foo", "movie"),
        ("https://letterboxd.com/series/bar", "series"),
    ]


def test_scrape_list_page_runs_scrape_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(list_scrap, "collect_film_links", lambda *_: [("https://letterboxd.com/film/test/", "movie")])
    movie = DummyMovie({"title": "Test"})
    monkeypatch.setattr(list_scrap, "scrape_one", lambda *args: movie)
    overlay_calls: list[DummyListPage] = []
    monkeypatch.setattr(list_scrap, "dismiss_overlay", lambda page: overlay_calls.append(page))

    context = DummyContext()
    tmdb_page = object()
    result = list_scrap.scrape_list_page(context, tmdb_page, "https://letterboxd.com/films/popular/")

    assert result == [{"title": "Test", "kind": "movie", "source_page": "https://letterboxd.com/films/popular/"}]
    assert context.pages[0].closed
    assert overlay_calls == [context.pages[0]]


def test_list_scrape_writes_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called_urls: list[str] = []

    def fake_list_urls(max_pages: int) -> list[str]:
        return [f"page-{i}" for i in range(1, max_pages + 1)]

    def fake_scrape_list_page(_, __, url: str) -> list[dict]:
        called_urls.append(url)
        return [{"title": url}]

    monkeypatch.setattr(list_scrap, "sync_playwright", lambda: DummyPlaywright())
    monkeypatch.setattr(list_scrap, "list_page_urls", fake_list_urls)
    monkeypatch.setattr(list_scrap, "scrape_list_page", fake_scrape_list_page)

    output_path = tmp_path / "list_results.json"
    list_scrap.list_scrape(max_pages=2, output_path=str(output_path))

    assert called_urls == ["page-1", "page-2"]
    dumped = json.loads(output_path.read_text())
    assert dumped == [{"title": "page-1"}, {"title": "page-2"}]
"""
Path(r"c:/Users/pierr/Documents/Université de Tours/Master MEcEn/MECEN - S9/WSML_project/src/WSML_code/tests/test_list_scrap.py").write_text(content, encoding="utf-8")
