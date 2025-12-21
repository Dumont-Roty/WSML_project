from playwright.sync_api import Page, TimeoutError
from typing import Optional
import re
import json
import time

class Scraping:
    @staticmethod
    def scrap_title(page: Page) -> str:
        page.wait_for_selector("h1.headline-1", timeout=5000)
        title = page.locator("h1.headline-1").text_content()
        if title is not None:
            return title.strip()
        else:
            return ""

    @staticmethod
    def scrap_directors(page: Page) -> list[str]:
        page.wait_for_selector("a[href^='/director/']", timeout=5000)
        director = page.locator("a[href^='/director/']").all()
        res_dir = []
        for dir in director:
            director_name = dir.text_content()
            if director_name is not None and director_name.strip() not in res_dir:
                res_dir.append(director_name.strip())
        return res_dir if res_dir else ["Directeur non trouvé"]

    @staticmethod
    def scrap_duree(page: Page) -> int:
        page.wait_for_selector(".text-link.text-footer", timeout=5000)
        duree_text = page.locator(".text-link.text-footer").text_content()
        if duree_text is not None:
            match = re.search(r"\d+\s", duree_text)
            if match:
                return int(match.group(0).strip())
        return 0

    @staticmethod
    def nbr_watched(page: Page) -> int:
        page.wait_for_selector(".production-statistic.-watches", timeout=5000)
        watched_int = page.locator(".production-statistic.-watches").inner_html()
        if watched_int is not None:
            match = re.search(r"\b([1-9],\d{3},\d{3}|[1-9]\d{0,2},\d{3}|[1-9]\d{0,2})\b", watched_int)
            if match:
                return int(match.group(0).replace(',', ''))
        return 0

    @staticmethod
    def scrap_appearence(page: Page) -> int:
        page.wait_for_selector(".production-statistic.-lists", timeout=5000)
        appearence_int = page.locator(".production-statistic.-lists").inner_html()
        if appearence_int is not None:
            match = re.search(r"\b([1-9],\d{3},\d{3}|[1-9]\d{0,2},\d{3}|[1-9]\d{0,2})\b", appearence_int)
            if match:
                return int(match.group(0).replace(',', ''))
        return 0

    @staticmethod
    def scrap_like(page: Page) -> int:
        page.wait_for_selector(".production-statistic.-likes", timeout=5000)
        like_int = page.locator(".production-statistic.-likes").inner_html()
        if like_int is not None:
            match = re.search(r"\b([1-9],\d{3},\d{3}|[1-9]\d{0,2},\d{3}|[1-9]\d{0,2})\b", like_int)
            if match:
                return int(match.group(0).replace(',', ''))
        return 0

    @staticmethod
    def scrap_rate(page: Page) -> Optional[float]:
        for attempt in range(3):
            try:
                scripts = page.locator("script[type='application/ld+json']").all_text_contents()
                for s in scripts:
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                    except Exception:
                        continue
                    def extract(o):
                        if isinstance(o, dict):
                            ar = o.get('aggregateRating')
                            if isinstance(ar, dict) and ar.get('ratingValue') is not None:
                                return ar.get('ratingValue')
                            if o.get('ratingValue') is not None:
                                return o.get('ratingValue')
                        return None

                    val = None
                    if isinstance(obj, list):
                        for item in obj:
                            val = extract(item)
                            if val is not None:
                                break
                    else:
                        val = extract(obj)

                    if val is not None:
                        try:
                            return float(val)
                        except Exception:
                            pass
            except Exception:
                pass

            try:
                rate_text = page.locator(".tooltip.display-rating.-highlight").inner_html()
                if rate_text:
                    m = re.search(r"\d+(\.\d+)?", rate_text)
                    if m:
                        return float(m.group(0))
            except Exception:
                pass

            try:
                meta = page.query_selector("meta[itemprop='ratingValue']")
                if meta:
                    try:
                        v = meta.get_attribute('content')
                    except Exception:
                        v = None
                    if v:
                        return float(v)
            except Exception:
                pass

            try:
                loc = page.locator("[data-rating], .rating, .average-rating")
                try:
                    attr = loc.inner_html()
                except Exception:
                    try:
                        attr = loc.text_content()
                    except Exception:
                        attr = None
                if attr:
                    m = re.search(r"\d+(\.\d+)?", attr)
                    if m:
                        return float(m.group(0))
            except Exception:
                pass

            try:
                time.sleep(0.25 + attempt * 0.25)
            except Exception:
                pass
            try:
                if attempt == 1:
                    page.reload(wait_until='domcontentloaded', timeout=5000)
            except Exception:
                pass

        return None

    @staticmethod
    def scrap_nbr_fan(page: Page) -> int:
        page.wait_for_selector("a.all-link.more-link", timeout=5000)
        text = (page.locator("a.all-link.more-link").text_content() or "").strip()
        if not text:
            return 0
        m = re.search(r"(?i)\b(\d+)\s*([k])?\b", text)
        if not m:
            return 0
        num = m.group(1)
        try:
            value = int(num)
        except ValueError:
            return 0
        if (m.group(2) or '').upper() == 'K':
            value *= 1_000

        return int(value)

    @staticmethod
    def scrap_casting(page: Page) -> list[str]:
        try:
            page.wait_for_selector("a[href^='/actor/']", timeout=5000, state='attached')
        except Exception:
            return ["Casting non trouvé"]
        try:
            texts = page.locator("a[href^='/actor/']").all_text_contents()
        except Exception:
            return ["Casting non trouvé"]
        res_cast = [t.strip() for t in texts if t and t.strip()][:10]
        return res_cast if res_cast else ["Casting non trouvé"]

    @staticmethod
    def scrap_producers(page: Page) -> list[str]:
        page.click("a[href^='/film/'][href$='/crew/']")
        page.wait_for_selector("a[href^='/film/'][href$='/crew/']", timeout=5000)
        producers = page.locator("a[href^='/producer/']").all()
        res_producers = []
        for producer in producers:
            producer_name = producer.text_content()
            if producer_name is not None and producer_name.strip() not in res_producers:
                res_producers.append(producer_name.strip())
        return res_producers if res_producers else ["Producteurs non trouvés"]

    @staticmethod
    def scrap_writers(page: Page) -> list[str]:
        page.click("a[href^='/film/'][href$='/crew/']")
        page.wait_for_selector("a[href^='/writer/'], a[href^='/original-writer/']", timeout=5000)
        writers = page.locator("a[href^='/writer/'], a[href^='/original-writer/']").all()
        res_writers = []
        for writer in writers:
            writer_name = writer.text_content()
            if writer_name is not None and writer_name.strip() not in res_writers:
                res_writers.append(writer_name.strip())
        return res_writers if res_writers else ["Ecrivains non trouvés"]

    @staticmethod
    def scrap_composer(page: Page) -> list[str]:
        try:
            page.click("a[href^='/film/'][href$='/crew/']")
        except Exception:
            pass
        try:
            page.wait_for_selector("a[href^='/composer/']", timeout=1500)
            composers = page.locator("a[href^='/composer/']").all()
        except Exception:
            return []
        res_composers = []
        for composer in composers:
            composer_name = composer.text_content()
            if composer_name is not None and composer_name.strip() not in res_composers:
                res_composers.append(composer_name.strip())
        return res_composers

    @staticmethod
    def scrap_year(page: Page) -> str:
        page.click("a[href^='/film/'][href$='/crew/']")
        page.wait_for_selector("a[href^='/films/year/']", timeout=5000)
        date = page.locator("a[href^='/films/year/']").text_content()
        if date is not None:
            return date.strip()
        else:
            return "Date non trouvée"

    @staticmethod
    def scrap_studios(page: Page) -> list[str]:
        page.click("a[href^='/film/'][href$='/details/']")
        page.wait_for_selector("a[href^='/studio/']", timeout=5000)
        studios = page.locator("a[href^='/studio/']").all()
        res_studios = []
        for studio in studios:
            studio_name = studio.text_content()
            if studio_name is not None and studio_name.strip() not in res_studios:
                res_studios.append(studio_name.strip())
        return res_studios if res_studios else ["Studios non trouvés"]

    @staticmethod
    def scrap_languages(page: Page) -> list[str]:
        page.click("a[href^='/film/'][href$='/details/']")
        page.wait_for_selector("a[href^='/films/language/']", timeout=5000)
        languages = page.locator("a[href^='/films/language/']").all()
        res_languages = []
        for language in languages:
            language_name = language.text_content()
            if language_name is not None and language_name.strip() not in res_languages:
                res_languages.append(language_name.strip())
        return res_languages if res_languages else ["Langues non trouvées"]

    @staticmethod
    def scrap_genres(page: Page) -> list[str]:
        page.click("a[href^='/film/'][href$='/genres/']")
        page.wait_for_selector("a[href^='/films/genre/']", timeout=5000)
        genres_elements = page.locator("a[href^='/films/genre/']").all()
        res_genres = []
        for element in genres_elements:
            genre_name = element.text_content()
            if genre_name is not None:
                res_genres.append(genre_name.strip())
        return res_genres if res_genres else ["Genres non trouvés"]

    @staticmethod
    def scrap_themes(page: Page) -> list[str]:
        page.click("a[href^='/film/'][href$='/genres/']")
        page.wait_for_selector("a[href^='/films/theme/'], a[href^='/films/mini-theme/']", timeout=5000)
        themes_elements = page.locator("a[href^='/films/theme/'], a[href^='/films/mini-theme/']").all()
        res_themes = []
        for element in themes_elements:
            theme_name = element.text_content()
            if theme_name is not None:
                res_themes.append(theme_name.strip())
        return res_themes if res_themes else ["Thèmes non trouvés"]

__all__ = ["Scraping"]
