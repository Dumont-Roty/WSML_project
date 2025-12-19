from WSML_code.models import Movie


def test_movie_defaults_lists_and_optional_none():
    movie = Movie(url="https://foo", title="Bar", year=2023)

    assert movie.directors == []
    assert movie.producers == []
    assert movie.themes == []
    assert movie.budget is None
    assert movie.fans_favoris is None


def test_movie_accepts_optional_values():
    movie = Movie(
        url="https://foo",
        title="Bar",
        year=2023,
        duration=120,
        rating=8.2,
        fans_favoris=42,
        languages=["Français"],
        genres=["Action"],
    )

    assert movie.duration == 120
    assert movie.rating == 8.2
    assert movie.languages == ["Français"]
    assert movie.genres == ["Action"]
