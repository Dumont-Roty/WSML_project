from playwright.sync_api import Page
from typing import Optional
import re
import os


def _debug_enabled() -> bool:
    return bool(os.getenv("WSML_DEBUG_TMDB"))


def _parse_money(value: str | None) -> Optional[int]:
    if not value:
        return None
    # Keep only digits and separators, then normalize.
    cleaned = re.sub(r"[^0-9,\.\s]", "", value).strip()
    if not cleaned:
        return None

    # If we have spaces/commas/dots as thousands separators, remove them.
    # We assume TMDB displays currency as an integer amount.
    digits_only = re.sub(r"[^0-9]", "", cleaned)
    if not digits_only:
        return None
    try:
        return int(digits_only)
    except ValueError:
        return None

class TMDBScraping:
    @staticmethod
    def _dismiss_tmdb_cookies(page):
        selectors = [
            "button:has-text('Tout refuser')",
            "button:has-text('Reject all')",
            "button:has-text('Refuser')",
            "button[aria-label*='Reject']",
            "#onetrust-reject-all-handler",
            "#onetrust-accept-btn-handler",
            "#accept-recommended-btn-handler",
        ]
        for sel in selectors:
            try:
                page.locator(sel).first.click(timeout=3000)
                break
            except Exception:
                continue

    @staticmethod
    def scrap_budget(page: Page) -> Optional[int]:
        debug = _debug_enabled()
        selectors = [
            "p:has(strong:has-text('Budget'))",
            "li:has(strong:has-text('Budget'))",
            "*:has-text('Budget')",
        ]

        budget_text: str | None = None
        for timeout in (4000, 8000, 12000):
            for sel in selectors:
                try:
                    page.wait_for_selector(sel, timeout=timeout)
                    budget_text = page.locator(sel).first.inner_text()
                    if budget_text and re.search(r"\d", budget_text):
                        if debug:
                            print(f"[tmdb] budget via selector {sel!r}: {budget_text!r}")
                        break
                except Exception:
                    continue
            if budget_text:
                break

        if not budget_text:
            try:
                body_text = page.inner_text("body")
                # Capture the number part to avoid parsing the whole label.
                m = re.search(r"(?i)\bBudget\b\s*[:\-]?\s*[\$€£]?\s*([\d\s.,]+)", body_text)
                if m:
                    budget_text = m.group(1)
                    if debug:
                        print(f"[tmdb] budget via body regex: {budget_text!r}")
            except Exception:
                return None

        return _parse_money(budget_text)

    @staticmethod
    def scrap_revenue(page: Page) -> Optional[int]:
        debug = _debug_enabled()
        selectors = [
            "p:has(strong:has-text('Revenue'))",
            "p:has(strong:has-text('Box Office'))",
            "li:has(strong:has-text('Revenue'))",
            "*:has-text('Revenue')",
        ]

        revenue_text: str | None = None
        for timeout in (4000, 8000, 12000):
            for sel in selectors:
                try:
                    page.wait_for_selector(sel, timeout=timeout)
                    revenue_text = page.locator(sel).first.inner_text()
                    if revenue_text and re.search(r"\d", revenue_text):
                        if debug:
                            print(f"[tmdb] revenue via selector {sel!r}: {revenue_text!r}")
                        break
                except Exception:
                    continue
            if revenue_text:
                break

        if not revenue_text:
            try:
                body_text = page.inner_text("body")
                m = re.search(
                    r"(?i)\b(?:Revenue|Recettes|Recette|Box Office|Gross)\b\s*[:\-]?\s*[\$€£]?\s*([\d\s.,]+)",
                    body_text,
                )
                if m:
                    revenue_text = m.group(1)
                    if debug:
                        print(f"[tmdb] revenue via body regex: {revenue_text!r}")
            except Exception:
                return None

        return _parse_money(revenue_text)

__all__ = ["TMDBScraping"]
