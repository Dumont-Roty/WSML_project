from WSML_code.element_to_scrap import Scraping
from typing import Any, cast

class DummyElement:
    def __init__(self, text):
        self._text = text

    def text_content(self):
        return self._text


class DummyLocator:
    def __init__(self, elements):
        self._elements = elements

    def all(self):
        return self._elements


class DummyPage:
    def __init__(self, composers=None, raise_wait=False):
        self._composers = composers or []
        self._raise_wait = raise_wait

    def click(self, *_args, **_kwargs):
        return None

    def wait_for_selector(self, *_args, **_kwargs):
        if self._raise_wait:
            raise Exception("timeout")
        return None

    def locator(self, selector, **_kwargs):
        if "composer" in selector:
            elements = [DummyElement(text) for text in self._composers]
            return DummyLocator(elements)
        return DummyLocator([])


def test_scrap_composer_returns_empty_if_selector_missing():
    page = DummyPage(raise_wait=True)
    assert Scraping.scrap_composer(cast(Any, page)) == []


def test_scrap_composer_deduplicates_names():
    page = DummyPage(composers=["John", "John ", " Alice"])
    assert Scraping.scrap_composer(cast(Any, page)) == ["John", "Alice"]
