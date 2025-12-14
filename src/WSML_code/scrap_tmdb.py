from playwright.sync_api import Page
from typing import Optional
import re

class TMDBScraping:
    @staticmethod
    def _dismiss_tmdb_cookies(page):
        """Tente de fermer la bannière cookies TMDB (OneTrust)."""
        selectors = [
            "button:has-text('Tout refuser')",
            "button:has-text('Reject all')",
            "button:has-text('Refuser')",
            "button[aria-label*='Reject']",
            "#onetrust-reject-all-handler",
            "#onetrust-accept-btn-handler",  # accept fallback
            "#accept-recommended-btn-handler",  # accept recommended (observé)
        ]
        for sel in selectors:
            try:
                page.locator(sel).first.click(timeout=1500)
                break
            except Exception:
                continue

    @staticmethod
    def scrap_budget(page: Page) -> Optional[int]:
        """
        Scrap the budget of the movie from the TMDB page.
        Args:
            page (Page): Playwright Page object representing the TMDB movie page.
        Returns:
            int: The budget of the movie as an integer.
        """
        # TMDB affiche généralement Budget dans un bloc Facts sous forme <p><strong>Budget</strong> $79,000,000.00</p>
        selectors = [
            "p:has(strong:has-text('Budget'))"
        ]

        budget_text = None
        for sel in selectors:
            try:
                page.wait_for_selector(sel, timeout=1200)
                budget_text = page.locator(sel).first.inner_text()
                if budget_text:
                    break
            except Exception:
                continue

        if not budget_text:
            # fallback: chercher dans tout le body
            try:
                body_text = page.inner_text("body")
                m = re.search(r"Budget\s*\$?[\d,.]+", body_text, re.IGNORECASE)
                if m:
                    budget_text = m.group(0)
            except Exception:
                return None
            if not budget_text:
                return None

        # conserver le point pour gérer les formats avec décimales (ex: $70,000,000.00)
        budget_number = re.sub(r"[^\d.]", "", budget_text)
        try:
            return int(float(budget_number)) if budget_number else 0
        except ValueError:
            return None
        
    @staticmethod
    def scrap_revenue(page: Page) -> Optional[int]:
        """
        Scrap the budget of the movie from the TMDB page.
        Args:
            page (Page): Playwright Page object representing the TMDB movie page.
        Returns:
            int: The budget of the movie as an integer.
        """
        selectors = [
            "p:has(strong:has-text('Revenue'))"
        ]

        revenue_text = None
        for sel in selectors:
            try:
                page.wait_for_selector(sel, timeout=1200)
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