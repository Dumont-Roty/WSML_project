from playwright.sync_api import Page
from typing import Optional
import re

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
        selectors = [
            "p:has(strong:has-text('Budget'))"
        ]

        budget_text = None
        for sel in selectors:
            try:
                page.wait_for_selector(sel, timeout=3000)
                budget_text = page.locator(sel).first.inner_text()
                if budget_text:
                    break
            except Exception:
                continue

        if not budget_text:
            try:
                body_text = page.inner_text("body")
                m = re.search(r"Budget\s*\$?[\d,.]+", body_text, re.IGNORECASE)
                if m:
                    budget_text = m.group(0)
            except Exception:
                return None
            if not budget_text:
                return None

        budget_number = re.sub(r"[^\d.]", "", budget_text)
        try:
            return int(float(budget_number)) if budget_number else 0
        except ValueError:
            return None

    @staticmethod
    def scrap_revenue(page: Page) -> Optional[int]:
        selectors = [
            "p:has(strong:has-text('Revenue'))"
        ]

        revenue_text = None
        for sel in selectors:
            try:
                page.wait_for_selector(sel, timeout=3000)
                revenue_text = page.locator(sel).first.inner_text()
                if revenue_text:
                    break
            except Exception:
                continue

        if not revenue_text:
            try:
                body_text = page.inner_text("body")
                m = re.search(r"(Revenue|Recettes|Recette|Box Office|Gross)\s*\$?[\d,.]+", body_text, re.IGNORECASE)
                if m:
                    revenue_text = m.group(0)
            except Exception:
                return None
            if not revenue_text:
                return None

        revenue_number = re.sub(r"[^\d.]", "", revenue_text)
        try:
            return int(float(revenue_number)) if revenue_number else 0
        except ValueError:
            return None

__all__ = ["TMDBScraping"]
