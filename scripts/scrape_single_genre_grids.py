r"""Scrape Letterboxd list pages, but keep only movies with a *single* genre.

Goal (requested): behave like `list_scraper`, except:
- only keep movies whose film page exposes a `/genre/` (singular) link (and not `/genres/`)
- no warning when a movie has multiple genres; we just skip it
- output the same fields as `scraping.models.Movie` (via Pydantic serialization)

Typical run (PowerShell):
  $env:PYTHONPATH='src'; .\.venv\Scripts\python .\scripts\scrape_single_genre_grids.py --max-pages 40 --start 1 --end 40 --output single_genre_movies.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from playwright.sync_api import sync_playwright

from scraping.models import Movie
from scraping.scrapers.list_scraper import _make_context, collect_film_links, list_page_urls
from scraping.scrapers.page_scraper import PageScrap
from scraping.services.dismiss_impl import dismiss_overlay


FILM_SINGLE_GENRE_LINK_SELECTOR = "a[href^='/film/'][href$='/genre/']"
FILM_MULTI_GENRES_LINK_SELECTOR = "a[href^='/film/'][href$='/genres/']"


def _normalize_bounds(*, max_pages: int, start_page: int, end_page: int | None) -> tuple[int, int]:
    if start_page < 1:
        start_page = 1
    if end_page is None:
        end_page = max_pages
    else:
        end_page = min(end_page, max_pages)
    return start_page, end_page


def _has_single_genre_link(page) -> bool:
    """Return True when the film page indicates a single genre ("Genre"), not multiple ("Genres")."""
    try:
        has_single = page.locator(FILM_SINGLE_GENRE_LINK_SELECTOR).count() > 0
    except Exception:
        has_single = False
    try:
        has_multi = page.locator(FILM_MULTI_GENRES_LINK_SELECTOR).count() > 0
    except Exception:
        has_multi = False
    return bool(has_single and not has_multi)


def scrape_single_genre_movies(
    *,
    max_pages: int,
    output_path: Path,
    start_page: int = 1,
    end_page: int | None = None,
    headless: bool = True,
) -> None:
    start_page, end_page = _normalize_bounds(max_pages=max_pages, start_page=start_page, end_page=end_page)
    if start_page > end_page:
        print(f"[warn] start_page ({start_page}) > end_page ({end_page}); nothing to do")
        return

    all_urls = list(list_page_urls(max_pages))
    slice_urls = all_urls[start_page - 1 : end_page]

    results: List[dict] = []
    total_films_seen = 0
    total_films_kept = 0
    total_films_skipped_multi = 0
    total_films_skipped_unknown = 0
    total_genres_scraped = 0

    with sync_playwright() as playwright:
        browser, context, tmdb_page = _make_context(playwright, headless=headless)
        try:
            for list_index, list_url in enumerate(slice_urls, start=start_page):
                print(f"[page] {list_index}/{end_page}: {list_url}")

                list_page = context.new_page()
                list_page.set_default_timeout(15000)
                try:
                    list_page.goto(list_url, wait_until="domcontentloaded", timeout=15000)
                    try:
                        dismiss_overlay(list_page)
                    except Exception:
                        pass
                    film_links = collect_film_links(list_page)
                except Exception as exc:
                    print(f"[warn] failed to open/scrape list page {list_url}: {exc}")
                    try:
                        list_page.close()
                    except Exception:
                        pass
                    continue
                finally:
                    try:
                        list_page.close()
                    except Exception:
                        pass

                page_kept = 0
                page_skipped_multi = 0
                page_skipped_unknown = 0


                for film_url, _kind in film_links:
                    total_films_seen += 1

                    film_page = context.new_page()
                    film_page.set_default_timeout(8000)
                    try:
                        film_page.goto(film_url, wait_until="domcontentloaded", timeout=12000)
                        try:
                            dismiss_overlay(film_page)
                        except Exception:
                            pass

                        # Scrape genres and filter: only keep films with exactly 1 genre
                        genres_block = PageScrap.scrap_genres_themes_page(film_page)
                        try:
                            scraped_genres = [g for g in (genres_block.get("genres") or []) if g]
                        except Exception:
                            scraped_genres = []

                        if len(scraped_genres) != 1:
                            total_films_skipped_multi += 1
                            page_skipped_multi += 1
                            continue

                        # Revenir sur la page principale du film pour le reste des infos
                        film_page.goto(film_url, wait_until="domcontentloaded", timeout=12000)
                        try:
                            dismiss_overlay(film_page)
                        except Exception:
                            pass

                        cast_block = PageScrap.scrap_cast_page(film_page)
                        crew_block = PageScrap.scrap_crew_page(film_page)
                        details_block = PageScrap.scrap_details_page(film_page)
                        tmdb_block = PageScrap.scrap_tmdb_url(film_page, tmdb_page)

                        data = {
                            "url": film_url,
                            **cast_block,
                            **crew_block,
                            **details_block,
                            **genres_block,
                            **tmdb_block,
                        }

                        movie = Movie(**data)
                        dumped = movie.model_dump()
                        results.append(dumped)

                        total_films_kept += 1
                        page_kept += 1
                        try:
                            total_genres_scraped += len(dumped.get("genres") or [])
                        except Exception:
                            pass

                    except Exception as exc:
                        print(f"[warn] failed to scrape film {film_url}: {exc}")
                    finally:
                        try:
                            film_page.close()
                        except Exception:
                            pass

                print(
                    f"  [stats] kept={page_kept} | skipped_multi={page_skipped_multi} | skipped_unknown={page_skipped_unknown}"
                )

        finally:
            try:
                tmdb_page.close()
            except Exception:
                pass
            try:
                context.close()
            except Exception:
                pass
            try:
                browser.close()
            except Exception:
                pass

    print(
        "[done] "
        f"films_seen={total_films_seen} | kept_single_genre={total_films_kept} | "
        f"skipped_multi={total_films_skipped_multi} | skipped_unknown={total_films_skipped_unknown} | "
        f"genres_scraped_total={total_genres_scraped}"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape full Movie data but keep only single-genre films")
    parser.add_argument("--max-pages", type=int, default=40)

    # Keep compatibility with older CLI usage.
    parser.add_argument("--start", type=int, default=None, help="Alias for --start-page")
    parser.add_argument("--end", type=int, default=None, help="Alias for --end-page")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=None)

    parser.add_argument("--output", type=str, default="single_genre_movies.json")
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)

    args = parser.parse_args()
    start_page = args.start_page
    end_page = args.end_page
    if args.start is not None:
        start_page = args.start
    if args.end is not None:
        end_page = args.end

    scrape_single_genre_movies(
        max_pages=int(args.max_pages),
        start_page=int(start_page),
        end_page=None if end_page is None else int(end_page),
        headless=bool(args.headless),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
