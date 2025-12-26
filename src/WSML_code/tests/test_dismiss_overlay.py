from WSML_code.services.dismiss_impl import dismiss_overlay


class LocatorStub:
    def __init__(self, count=1, throws=False):
        self._count = count
        self._throws = throws
        self.clicked = False

    @property
    def first(self):
        return self

    def count(self):
        return self._count

    def evaluate(self, _script):
        if self._throws:
            raise Exception("evaluate fail")
        self.clicked = True

    def click(self, **_kwargs):
        if self._throws:
            raise Exception("click fail")
        self.clicked = True


class PageStub:
    def __init__(self, behavior, fallback_success=True):
        self.behavior = behavior
        self.fallback_success = fallback_success
        self.evaluated = False
        self.removed = False
        self.last_locator = None

    def locator(self, selector):
        data = self.behavior.get(selector)
        if data is None:
            stub = LocatorStub(count=0)
        else:
            stub = LocatorStub(**data)
        self.last_locator = stub
        return stub

    def evaluate(self, _script):
        self.evaluated = True
        if not self.fallback_success:
            raise Exception("fallback fail")
        self.removed = True


def test_dismiss_overlay_clicks_available_button():
    page = PageStub({"button#onetrust-accept-btn-handler": {}})

    assert dismiss_overlay(page)
    assert page.last_locator is not None and page.last_locator.clicked


def test_dismiss_overlay_handles_fc_button_via_eval():
    page = PageStub({"p.fc-button-label": {}})

    assert dismiss_overlay(page)
    assert page.last_locator is not None and page.last_locator.clicked


def test_dismiss_overlay_fallback_cleanup():
    page = PageStub({}, fallback_success=True)

    assert dismiss_overlay(page)
    assert page.evaluated
    assert page.removed


def test_dismiss_overlay_returns_false_when_all_fail():
    page = PageStub({}, fallback_success=False)

    assert not dismiss_overlay(page)


class BehaviorLocator:
    def __init__(self, count=1, click_raises=False, eval_raises=False):
        self._count = count
        self.clicked = False
        self.eval_called = False
        self.click_raises = click_raises
        self.eval_raises = eval_raises

    @property
    def first(self):
        return self

    def count(self):
        return self._count

    def evaluate(self, script):
        self.eval_called = True
        if self.eval_raises:
            raise Exception("eval fail")

    def click(self, **_kwargs):
        if self.click_raises:
            raise Exception("click fail")
        self.clicked = True


class BehaviorPage:
    def __init__(self, selectors, js_raise=False):
        self.selectors = selectors
        self.js_raise = js_raise
        self.eval_removed = False
        self.clicked = []

    def locator(self, selector):
        return self.selectors.get(selector, BehaviorLocator(count=0))

    def evaluate(self, script):
        self.eval_removed = True
        if self.js_raise:
            raise Exception("evaluate fail")


def test_dismiss_overlay_clicks_selector_in_order():
    page = BehaviorPage({
        "button#onetrust-accept-btn-handler": BehaviorLocator(),
        "button[aria-label='Accept']": BehaviorLocator(count=0),
    })

    assert dismiss_overlay(page)
    assert page.locator("button#onetrust-accept-btn-handler").clicked


def test_dismiss_overlay_handles_fc_button():
    page = BehaviorPage({
        "p.fc-button-label": BehaviorLocator(eval_raises=False)
    })

    assert dismiss_overlay(page)
    assert page.locator("p.fc-button-label").eval_called


def test_dismiss_overlay_fallback_succeeds_after_all_selectors():
    page = BehaviorPage({}, js_raise=False)

    assert dismiss_overlay(page)
    assert page.eval_removed


def test_dismiss_overlay_fallback_failure():
    page = BehaviorPage({}, js_raise=True)

    assert not dismiss_overlay(page)
