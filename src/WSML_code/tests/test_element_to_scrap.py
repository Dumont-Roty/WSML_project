from typing import Any, Protocol, cast

from WSML_code.element_to_scrap import Scraping



class ElementStub:
    def __init__(self, value: str | None):
        self._value = value

    def text_content(self):
        return self._value


class LocatorStub:
    def __init__(self, *, text=None, inner_html=None, elements=None, texts=None):
        self._text = text
        self._inner_html = inner_html
        self._elements = elements or []
        self._texts = texts or []

    @property
    def first(self):
        return self

    def text_content(self):
        return self._text

    def inner_html(self):
        return self._inner_html

    def all(self):
        return [ElementStub(text) for text in self._elements]

    def all_text_contents(self):
        return list(self._texts)


class PageProtocol(Protocol):
    def wait_for_selector(self, selector: str, **_kwargs: Any) -> None:
        ...

    def locator(self, selector: str, **_kwargs: Any) -> LocatorStub:
        ...

    def click(self, selector: str, **_kwargs: Any) -> None:
        ...


class PageStub:
    def __init__(self, locators=None, wait_failures=None):
        self.locators = locators or {}
        self.wait_failures = wait_failures or set()
        self.clicked = []

    def wait_for_selector(self, selector, **_kwargs):
        if selector in self.wait_failures:
            raise Exception("timeout")
        return None

    def locator(self, selector, **_kwargs):
        return self.locators.get(selector, LocatorStub())

    def click(self, selector, **_kwargs):
        self.clicked.append(selector)


def test_scrap_title_strips_whitespace():
    page = PageStub(locators={"h1.headline-1": LocatorStub(text="  Film titre  ")})

    assert Scraping.scrap_title(page) == "Film titre"


def test_scrap_title_returns_empty_when_missing_text():
    page = PageStub(locators={"h1.headline-1": LocatorStub(text=None)})

    assert Scraping.scrap_title(page) == ""


def test_scrap_duree_parses_minutes():
    page = PageStub(locators={".text-link.text-footer": LocatorStub(text="90 min")})

    assert Scraping.scrap_duree(page) == 90


def test_scrap_duree_returns_zero_when_unparsable():
    page = PageStub(locators={".text-link.text-footer": LocatorStub(text="Durée inconnue")})

    assert Scraping.scrap_duree(page) == 0


def test_scrap_directors_deduplicates_results_and_defaults():
    page = PageStub(
        locators={
            "a[href^='/director/']": LocatorStub(elements=[" Alice ", "Alice", None, "Bob"])
        }
    )

    assert Scraping.scrap_directors(page) == ["Alice", "Bob"]
    assert Scraping.scrap_directors(PageStub()) == ["Directeur non trouvé"]


def test_scrap_numeric_stats_handle_commas():
    page = PageStub(
        locators={
            ".production-statistic.-watches": LocatorStub(inner_html="1,234"),
            ".production-statistic.-lists": LocatorStub(inner_html="2,345"),
            ".production-statistic.-likes": LocatorStub(inner_html="3,456"),
        }
    )

    assert Scraping.nbr_watched(page) == 1234
    assert Scraping.scrap_appearence(page) == 2345
    assert Scraping.scrap_like(page) == 3456


def test_scrap_rate_and_fan_parsing():
    page = PageStub(
        locators={
            ".tooltip.display-rating.-highlight": LocatorStub(inner_html="Note 8.3/10"),
            "a.all-link.more-link": LocatorStub(text="12 K fans"),
        }
    )

    assert Scraping.scrap_rate(page) == 8.3
    assert Scraping.scrap_nbr_fan(page) == 12_000


def test_scrap_casting_defaults_when_missing():
    page = PageStub(wait_failures={"a[href^='/actor/']"})

    assert Scraping.scrap_casting(page) == ["Casting non trouvé"]


def test_scrap_producers_and_writers_deduplicate_entries():
    page = PageStub(
        locators={
            "a[href^='/producer/']": LocatorStub(elements=["Prod A", "Prod A"]),
            "a[href^='/writer/'], a[href^='/original-writer/']": LocatorStub(
                elements=["Writer 1", None, "Writer 2"]
            ),
        }
    )

    assert Scraping.scrap_producers(page) == ["Prod A"]
    assert Scraping.scrap_writers(page) == ["Writer 1", "Writer 2"]


def test_scrap_year_and_informations_list():
    page = PageStub(
        locators={
            "a[href^='/films/year/']": LocatorStub(text=" 2021 "),
            "a[href^='/studio/']": LocatorStub(elements=["Studio A"]),
            "a[href^='/films/language/']": LocatorStub(elements=["English"]),
        }
    )

    assert Scraping.scrap_year(page) == "2021"
    assert Scraping.scrap_studios(page) == ["Studio A"]
    assert Scraping.scrap_languages(page) == ["English"]


def test_scrap_genres_and_themes_default_to_not_found():
    page = PageStub()

    assert Scraping.scrap_genres(page) == ["Genres non trouvés"]
    assert Scraping.scrap_themes(page) == ["Thèmes non trouvés"]


def test_scrap_genres_returns_list_when_available():
    page = PageStub(locators={"a[href^='/films/genre/']": LocatorStub(elements=[" Action ", "Drame"])})

    assert Scraping.scrap_genres(page) == ["Action", "Drame"]


def test_scrap_themes_returns_values():
    page = PageStub(locators={"a[href^='/films/theme/'], a[href^='/films/mini-theme/']": LocatorStub(elements=[" Thriller ", "Aventure"])})

    assert Scraping.scrap_themes(page) == ["Thriller", "Aventure"]


class ComposerDummyPage:
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
            elements = [ElementStub(text) for text in self._composers]
            return LocatorStub(elements=[text for text in self._composers])
        return LocatorStub()


def test_scrap_composer_returns_empty_if_selector_missing():
    page: PageProtocol = ComposerDummyPage(raise_wait=True)

    assert Scraping.scrap_composer(page) == []


def test_scrap_composer_deduplicates_names():
    page: PageProtocol = ComposerDummyPage(composers=["John", "John ", " Alice"])

    assert Scraping.scrap_composer(page) == ["John", "Alice"]
