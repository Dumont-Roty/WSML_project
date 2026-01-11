from WSML_code.utils import helpers


def test_normalize_title_removes_accents_and_lowercases():
    assert helpers._normalize_title("Éléphant") == "elephant"


def test_detect_kind_uses_title_and_href():
    assert helpers.detect_kind("/films/123", "My Movie") == "movie"
    assert helpers.detect_kind("/series/456", "Whatever") == "series"
    assert helpers.detect_kind("/films/789", "TV Show") == "series"
