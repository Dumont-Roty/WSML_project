def dismiss_overlay(page):
    selectors = [
        "button#onetrust-accept-btn-handler",
        "button[aria-label='Accept']",
        "button[title*='Accept']",
        "p.fc-button-label",
        "text=Consent",
    ]

    for sel in selectors:
        try:
            loc = page.locator(sel)
            if loc.count() == 0:
                continue
            el = loc.first
            if sel == "p.fc-button-label":
                try:
                    el.evaluate("el => { const b = el.closest('button'); if (b) b.click(); else el.click(); }")
                    return True
                except Exception:
                    pass

            try:
                el.click(timeout=2000)
                return True
            except Exception:
                pass
        except Exception:
            continue

    try:
        page.evaluate("""() => {
            ['.fc-consent-root', '#onetrust-banner-sdk', '.cookie-consent', '.consent-banner'].forEach(s => {
                const el = document.querySelector(s);
                if (el) el.remove();
            });
        }""")
        return True
    except Exception:
        return False

__all__ = ["dismiss_overlay"]
