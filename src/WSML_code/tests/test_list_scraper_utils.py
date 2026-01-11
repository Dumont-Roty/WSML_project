from WSML_code.scrapers import list_scraper as ls


def test_list_page_urls_first_and_next():
    urls = list(ls.list_page_urls(3))
    assert urls[0].endswith("/films/popular/")
    assert urls[1].endswith("/films/popular/page/2/")
    assert urls[2].endswith("/films/popular/page/3/")


def test_progress_counter_increments_thread_safe():
    pc = ls.ProgressCounter(total_estimate=10)
    pc.incr()
    pc.incr(2)
    assert pc.value() == 3
    assert pc.total_estimate == 10


def test_find_next_list_page_returns_absolute_url():
    class NextLink:
        def __init__(self, href: str):
            self._href = href

        def get_attribute(self, name: str):
            return self._href if name == "href" else None

    class Page:
        def __init__(self, href: str | None):
            self._next = NextLink(href) if href else None

        def query_selector(self, selector: str):
            return self._next

    page = Page("/films/popular/page/2/")
    out = ls.find_next_list_page(page)
    assert out.endswith("/films/popular/page/2/")

    page_none = Page(None)
    assert ls.find_next_list_page(page_none) is None


def test_normalize_title_and_detect_kind():
    # Accents removed and lowercased
    assert ls._normalize_title("Sérîe") == "serie"
    # detect series by href and keywords
    assert ls.detect_kind("/series/abc", "Movie") == "series"
    assert ls.detect_kind("/films/abc", "TV Show") == "series"
    assert ls.detect_kind("/films/abc", "Regular Film") == "movie"


class _FakeLocator:
    def __init__(self, frames, counts=None):
        self._frames = frames
        self._counts = counts or []
        self._i = 0

    def count(self):
        if self._i < len(self._counts):
            val = self._counts[self._i]
            self._i += 1
            return val
        return len(self._frames)

    def all(self):
        return list(self._frames)


class _FakeFrame:
    def __init__(self, href, title):
        self._href = href
        self._title = title

    def get_attribute(self, name):
        if name == "href":
            return self._href
        if name == "data-original-title":
            return self._title
        return None


def test_collect_film_links_scroll_stops_and_collects(monkeypatch):
    frames = [_FakeFrame("/films/a", "Film A"), _FakeFrame("/series/b", "Series B")]

    class Page:
        def __init__(self):
            self.context = None
            self._locator = _FakeLocator(frames, counts=[1, 2, 2])

        def wait_for_load_state(self, *a, **k):
            return None

        def wait_for_selector(self, *a, **k):
            return None

        def evaluate(self, *a, **k):
            return None

        def wait_for_timeout(self, *a, **k):
            return None

        def locator(self, selector):
            return self._locator

    page = Page()
    out = ls.collect_film_links(page)
    assert len(out) == 2
    assert out[0][1] == "movie"
    assert out[1][1] == "series"


def test_collect_film_links_ajax_fallback(monkeypatch):
    initial_frames = [_FakeFrame("/films/a", "Film A")]
    ajax_frames = [_FakeFrame("/films/c", "Film C"), _FakeFrame("/series/d", "Series D")]

    class TempPage:
        def __init__(self):
            self._locator = _FakeLocator(ajax_frames)

        def set_default_timeout(self, *a, **k):
            return None

        def route(self, *a, **k):
            return None

        def goto(self, *a, **k):
            return None

        def wait_for_selector(self, *a, **k):
            return None

        def locator(self, selector):
            return self._locator

        def close(self):
            return None

    class Container:
        def get_attribute(self, name):
            return "/ajax/fragment" if name == "data-url" else None

    class Context:
        def __init__(self):
            self.created = False

        def new_page(self):
            self.created = True
            return TempPage()

    class Page:
        def __init__(self):
            self.context = Context()
            self._locator = _FakeLocator(initial_frames, counts=[1])

        def wait_for_load_state(self, *a, **k):
            return None

        def wait_for_selector(self, *a, **k):
            return None

        def evaluate(self, *a, **k):
            return None

        def wait_for_timeout(self, *a, **k):
            return None

        def locator(self, selector):
            return self._locator

        def query_selector(self, selector):
            return Container()

    page = Page()
    out = ls.collect_film_links(page)
    # AJAX frames should replace the initial single frame
    assert len(out) == 2
    kinds = {k for _, k in out}
    assert kinds == {"movie", "series"}


def test_scrape_list_page_success(monkeypatch):
    called = {"dismiss": 0, "scrape": 0, "next": 0}

    class FakePage:
        def __init__(self):
            self.closed = False

        def set_default_timeout(self, *a, **k):
            return None

        def goto(self, *a, **k):
            return None

        def wait_for_load_state(self, *a, **k):
            return None

        def close(self):
            self.closed = True

    class FakeContext:
        def __init__(self):
            self.page = FakePage()

        def new_page(self):
            return self.page

    class FakeMovie:
        def model_dump(self):
            return {"title": "ok"}

    def fake_collect(page):
        return [("http://film", "movie")]

    def fake_dismiss(page):
        called["dismiss"] += 1

    def fake_scrape_one(context, tmdb_page, film_url):
        called["scrape"] += 1
        return FakeMovie()

    def fake_next(page):
        called["next"] += 1
        return "http://next"

    monkeypatch.setattr(ls, "collect_film_links", fake_collect)
    monkeypatch.setattr(ls, "dismiss_overlay", fake_dismiss)
    monkeypatch.setattr(ls, "scrape_one", fake_scrape_one)
    monkeypatch.setattr(ls, "find_next_list_page", fake_next)

    progress = ls.ProgressCounter()
    results, nxt = ls.scrape_list_page(FakeContext(), object(), "http://list", progress=progress)

    assert results[0]["title"] == "ok"
    assert results[0]["kind"] == "movie"
    assert nxt == "http://next"
    assert called["dismiss"] == 1 and called["scrape"] == 1 and called["next"] == 1
    assert progress.value() == 1


def test_scrape_list_page_timeout(monkeypatch):
    class FakePage:
        def __init__(self):
            self.closed = False

        def set_default_timeout(self, *a, **k):
            return None

        def goto(self, *a, **k):
            raise ls.TimeoutError("timeout")

        def wait_for_load_state(self, *a, **k):
            return None

        def close(self):
            self.closed = True

    class FakeContext:
        def __init__(self):
            self.page = FakePage()

        def new_page(self):
            return self.page

    results, nxt = ls.scrape_list_page(FakeContext(), object(), "http://list")
    assert results == []
    assert nxt is None
