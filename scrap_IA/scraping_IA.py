import json
import re
from pathlib import Path
import argparse
from typing import Optional, Tuple

from bs4 import BeautifulSoup as BS
from playwright.sync_api import sync_playwright


DEFAULT_URL = "https://letterboxd.com/film/the-lord-of-the-rings-the-two-towers/"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:143.0) Gecko/20100101 Firefox/143.0"
)


def fetch_rendered_html(
    url: str,
    user_agent: str = USER_AGENT,
    wait_selector: str = "section.ratings-histogram-chart, a.display-rating",
    timeout_ms: int = 15000,
) -> str:
    """Render the page with Playwright (Chromium) and return the DOM HTML."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=user_agent, locale="en-US", color_scheme="light")
        page.goto(url, wait_until="networkidle")
        try:
            page.wait_for_selector(wait_selector, timeout=timeout_ms)
        except Exception:
            # Continue anyway; some pages may already have the needed nodes
            pass
        rendered = page.content()
        browser.close()
        return rendered


def parse_float(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"\d+(?:[.,]\d+)?", text)
    return float(m.group(0).replace(",", ".")) if m else None


# accepte “ratings | votes | notes | évaluations” + espaces insécables
WORD_VOTES = r"(?:ratings|votes|notes|évaluations)"


def parse_int_with_sep(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(
        rf"([\d]{{1,3}}(?:[,\.\s\u00A0]\d{{3}})*)\s+{WORD_VOTES}", text, flags=re.I
    )
    if not m:
        return None
    return int(re.sub(r"[\s\u00A0,._]", "", m.group(1)))


def extract_title(soup: BS) -> Optional[str]:
    el = soup.select_one("h1.headline-1.primaryname") or soup.find("h1")
    return el.get_text(strip=True) if el else None


def extract_rating_block(soup: BS) -> Tuple[Optional[float], Optional[float], Optional[int], str]:
    """Essaye plusieurs méthodes pour récupérer (displayed, weighted, voters).

    Retourne (displayed_value, weighted_value, voters, found_via)
    """
    # 1) CSS selectors tolérants
    candidates = [
        "aside.sidebar a.display-rating",
        "a.display-rating",
        "aside.sidebar a[class*='display-rating']",
        "a[class*='display-rating']",
        "aside.sidebar a[href$='/ratings/']",
        "a[href$='/ratings/']",
        "aside.sidebar a[href*='/ratings/']",
    ]
    a = None
    found_via = ""
    for sel in candidates:
        a = soup.select_one(sel)
        if a:
            found_via = f"css:{sel}"
            break

    # 2) href regex
    if not a:
        a = soup.find("a", href=re.compile(r"/film/[^/]+/ratings/?$"))
        if a:
            found_via = "href-regex"

    displayed_value = weighted_value = None
    voters = None
    if a:
        displayed_value = parse_float(a.get_text(strip=True))
        # a.get(...) can return an AttributeValueList (or list/tuple) in BeautifulSoup;
        # normalize to a string before using replace/strip.
        raw_attr = a.get("data-original-title") or a.get("title") or ""
        if isinstance(raw_attr, (list, tuple)):
            raw_attr = " ".join(raw_attr)
        attr_text = str(raw_attr).replace("\xa0", " ").strip()
        if attr_text:
            weighted_value = parse_float(attr_text)
            voters = parse_int_with_sep(attr_text)
        return displayed_value, weighted_value, voters, found_via

    # 3) JSON-LD fallback (aggregateRating)
    for sc in soup.find_all("script", type="application/ld+json"):
        t = sc.string or sc.text or ""
        if "aggregateRating" not in t:
            continue
        try:
            data = json.loads(t)
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        for obj in items:
            if isinstance(obj, dict) and "aggregateRating" in obj:
                agg = obj["aggregateRating"]
                weighted_value = parse_float(str(agg.get("ratingValue")))
                rc = str(agg.get("ratingCount") or "").replace("\xa0", " ")
                voters = parse_int_with_sep(f"{rc} ratings") or (int(rc) if rc.isdigit() else None)
                return None, weighted_value, voters, "jsonld"

    return None, None, None, ""


def parse_with_bs(html: str) -> BS:
    try:
        return BS(html, "lxml")  # si lxml dispo, plus robuste
    except Exception:
        return BS(html, "html.parser")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Extract ratings from a Letterboxd film page using Playwright")
    parser.add_argument("url", nargs="?", default=DEFAULT_URL, help="Film URL on letterboxd.com")
    parser.add_argument("--timeout", type=int, default=15, help="Playwright wait timeout in seconds (default 15)")
    parser.add_argument("--wait-selector", default="section.ratings-histogram-chart, a.display-rating", help="CSS selector to wait for when rendering")
    parser.add_argument("--user-agent", default=USER_AGENT, help="Custom User-Agent string for Playwright")
    args = parser.parse_args(argv)

    url = args.url

    print("Rendering page with Playwright…")
    try:
        rendered_html = fetch_rendered_html(
            url,
            user_agent=args.user_agent,
            wait_selector=args.wait_selector,
            timeout_ms=args.timeout * 1000,
        )
        # Sauvegarde pour l'inspection
        output_path = Path("rendered_letterboxd.html")
        output_path.write_text(rendered_html, encoding="utf-8", errors="replace")
        print(f"HTML rendered and saved to {output_path}")

        soup = parse_with_bs(rendered_html)

        title = extract_title(soup)
        print("Title:", title or "Titre non trouvé")

        displayed, weighted, voters, via = extract_rating_block(soup)
        print("Found via:", via or "(none)")
        print("Displayed:", displayed, "| Weighted:", weighted, "| Voters:", voters)

        if not (displayed or weighted or voters):
            print("Aucun rating trouvé. Vérifiez le fichier rendered_letterboxd.html.")

    except ImportError:
        print(
            "Playwright n'est pas installé. Installez-le puis exécutez : \n"
            "  python -m pip install playwright beautifulsoup4 lxml\n"
            "  python -m playwright install\n"
            "Ensuite, relancez ce script."
        )
    except Exception as e:
        print(f"Une erreur est survenue avec Playwright: {e}")


if __name__ == "__main__":
    main()


