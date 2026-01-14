from playwright.sync_api import Page
from typing import Optional
import re
import os
from time import perf_counter


def _debug_enabled() -> bool:
    return bool(os.getenv("WSML_DEBUG_TMDB"))


def _effective_timeouts(deadline: float | None, defaults: tuple[int, ...]) -> list[int]:
    """Clip default timeouts to the remaining deadline (ms)."""
    if deadline is None:
        return list(defaults)
    result: list[int] = []
    for default_timeout in defaults:
        remaining_ms = int((deadline - perf_counter()) * 1000)
        if remaining_ms <= 500:
            break
        # Avoid too-small values that cause immediate timeout but clip to remaining.
        result.append(max(500, min(default_timeout, remaining_ms)))
    return result or [500]


def _remaining_timeout(deadline: float | None, base_ms: int) -> int:
    if deadline is None:
        return base_ms
    remaining_ms = int((deadline - perf_counter()) * 1000)
    return max(300, min(base_ms, remaining_ms))


def _extract_from_facts(page: Page, label: str, deadline: float | None) -> Optional[str]:
    """Try to read a label/value from the TMDB facts column."""
    timeout = _remaining_timeout(deadline, 1500)
    try:
        facts = page.locator("section.facts")
        facts.wait_for(timeout=timeout)
        text = facts.inner_text()
    except Exception:
        return None
    m = re.search(fr"(?i){re.escape(label)}\s*[:\-]?\s*([\$€£]?\s*[\d\s.,]+)", text)
    return m.group(1) if m else None


def _parse_money(value: str | None) -> Optional[int]:
    if not value:
        return None
    # Explicit markers that mean "no value" on TMDB pages
    if value.strip() in ("-", "—", "–", "N/A", "n/a", "None"):
        return None
    # Keep only digits and separators, then normalize.
    cleaned = re.sub(r"[^0-9,\.\s]", "", value).strip()
    if not cleaned:
        return None

    candidates = re.findall(r"[0-9][0-9\s,\.]*[0-9]", cleaned)
    if not candidates:
        return None

    def _to_int(text: str) -> Optional[int]:
        # Normalize spaces
        txt = text.replace("\xa0", " ").strip()
        # Heuristic for decimal part: if ends with .xx or ,xx, drop decimals
        if re.search(r"[\.,]\d{1,2}$", txt):
            txt = re.sub(r"[\.,](\d{1,2})$", "", txt)
        # Remove thousand separators (commas, dots, spaces)
        digits_only = re.sub(r"[\s,\.]", "", txt)
        if not digits_only:
            return None
        # If excessively long, likely concatenation (keep <= 12 digits ~ up to hundreds of billions)
        if len(digits_only) > 12:
            if _debug_enabled():
                print(f"[tmdb] discard concatenated/oversized candidate: {digits_only}")
            return None
        try:
            return int(digits_only)
        except ValueError:
            return None

    parsed_vals: list[int] = []
    for cand in candidates:
        val = _to_int(cand)
        if val is None:
            continue
        # Discard obviously incorrect huge values (greater than 200 billion)
        if 0 <= val <= 200_000_000_000:
            parsed_vals.append(val)

    if not parsed_vals:
        return None

    return parsed_vals[0]

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
    def scrap_budget(page: Page, deadline: float | None = None) -> Optional[int]:
        debug = _debug_enabled()
        selectors = [
            "p:has(strong:has-text('Budget'))",
            "li:has(strong:has-text('Budget'))",
            "*:has-text('Budget')",
        ]

        budget_text: str | None = None
        for timeout in _effective_timeouts(deadline, (4000, 8000, 12000)):
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
            # Fallback: look inside the facts column
            budget_text = _extract_from_facts(page, "Budget", deadline)
            if budget_text and debug:
                print(f"[tmdb] budget via facts: {budget_text!r}")

        if not budget_text:
            try:
                body_text = page.inner_text("body")
                m = re.search(r"(?i)\bBudget\b\s*[:\-]?\s*[\$€£]?\s*([\d\s.,]+)", body_text)
                if m:
                    budget_text = m.group(1)
                    if debug:
                        print(f"[tmdb] budget via body regex: {budget_text!r}")
            except Exception:
                return None

        return _parse_money(budget_text)

    @staticmethod
    def scrap_revenue(page: Page, deadline: float | None = None) -> Optional[int]:
        debug = _debug_enabled()
        selectors = [
            "p:has(strong:has-text('Revenue'))",
            "p:has(strong:has-text('Box Office'))",
            "p:has(strong:has-text('Recettes'))",
            "p:has(strong:has-text('Recette'))",
            "li:has(strong:has-text('Revenue'))",
            "*:has-text('Revenue')",
            "*:has-text('Recettes')",
            "*:has-text('Recette')",
        ]

        revenue_text: str | None = None
        for timeout in _effective_timeouts(deadline, (4000, 8000, 12000)):
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
            # Fallback: look inside the facts column
            revenue_text = _extract_from_facts(page, "Revenue", deadline)
            if revenue_text and debug:
                print(f"[tmdb] revenue via facts: {revenue_text!r}")

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
