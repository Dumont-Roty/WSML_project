import math
import os
import time

import pytest

from scraping.services import tmdb_impl as tmdb


def test_effective_timeouts_clips_deadline(monkeypatch):
    start = time.perf_counter()
    deadline = start + 0.5  # 500ms
    out = tmdb._effective_timeouts(deadline, (1000, 2000))
    assert all(t <= 1000 for t in out)


def test_effective_timeouts_returns_floor_when_no_time():
    deadline = time.perf_counter() - 0.1
    out = tmdb._effective_timeouts(deadline, (1000,))
    assert out == [500]


def test_remaining_timeout_respects_min_and_deadline():
    start = time.perf_counter()
    deadline = start + 0.1
    val = tmdb._remaining_timeout(deadline, 2000)
    assert 300 <= val <= 2000


def test_parse_money_variants_and_invalid():
    assert tmdb._parse_money("$1,234.50") == 1234
    assert tmdb._parse_money("N/A") is None
    assert tmdb._parse_money("10000000000000000") is None  # too long


def test_parse_money_debug_long_value(monkeypatch, capsys):
    monkeypatch.setenv("WSML_DEBUG_TMDB", "1")
    assert tmdb._parse_money("9999999999999999") is None
    out = capsys.readouterr().out
    assert "discard" in out
    monkeypatch.delenv("WSML_DEBUG_TMDB")


class _FakeLocator:
    def __init__(self, text: str):
        self._text = text
        self.first = self

    def wait_for(self, *a, **k):
        return None

    def inner_text(self):
        return self._text

    def click(self, *a, **k):
        return None


class _FakePage:
    def __init__(self, text: str):
        self._text = text
        self.locators = {}

    def locator(self, sel):
        return self.locators.get(sel, _FakeLocator(self._text))

    def inner_text(self, sel):
        return self._text

    def wait_for_selector(self, sel, timeout=None):
        return None


def test_extract_from_facts_reads_label():
    page = _FakePage("Budget: $1,000")
    page.locators["section.facts"] = _FakeLocator("Budget: $1,000")
    out = tmdb._extract_from_facts(page, "Budget", deadline=None)
    assert out == "$1,000"


def test_scrap_budget_and_revenue_selectors(monkeypatch):
    budget_page = _FakePage("Budget: $2,000")
    revenue_page = _FakePage("Revenue: $3,500")
    # Ensure wait_for_selector uses our locator text
    budget_page.locators = {"p:has(strong:has-text('Budget'))": _FakeLocator("Budget: $2,000")}
    revenue_page.locators = {"p:has(strong:has-text('Revenue'))": _FakeLocator("Revenue: $3,500")}

    assert tmdb.TMDBScraping.scrap_budget(budget_page, deadline=None) == 2000
    assert tmdb.TMDBScraping.scrap_revenue(revenue_page, deadline=None) == 3500


def test_scrap_budget_debug_print(monkeypatch, capsys):
    monkeypatch.setenv("WSML_DEBUG_TMDB", "1")
    page = _FakePage("Budget: 1 000")
    page.locators = {"p:has(strong:has-text('Budget'))": _FakeLocator("Budget: 1 000")}
    assert tmdb.TMDBScraping.scrap_budget(page, deadline=None) == 1000
    assert "budget via selector" in capsys.readouterr().out
    monkeypatch.delenv("WSML_DEBUG_TMDB")


def test_scrap_budget_fallback_body(monkeypatch):
    page = _FakePage("body Budget 9 000")
    page.locators = {"p:has(strong:has-text('Budget'))": _FakeLocator("-")}
    assert tmdb.TMDBScraping.scrap_budget(page, deadline=None) == 9000


def test_scrap_revenue_body_exception_returns_none(monkeypatch):
    class PageFail(_FakePage):
        def inner_text(self, sel):
            raise RuntimeError("boom")

    page = PageFail("text")
    assert tmdb.TMDBScraping.scrap_revenue(page, deadline=None) is None


def test_dismiss_tmdb_cookies_tries_selectors(monkeypatch):
    clicks = []

    class Locator:
        def __init__(self, name):
            self.name = name
            self.first = self

        def click(self, timeout=None):
            clicks.append(self.name)
            if self.name != "button[aria-label*='Reject']":
                raise Exception("fail")

    class Page:
        def locator(self, sel):
            return Locator(sel)

    tmdb.TMDBScraping._dismiss_tmdb_cookies(Page())
    assert "button[aria-label*='Reject']" in clicks
