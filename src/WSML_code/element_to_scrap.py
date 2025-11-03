from playwright.sync_api import Page, sync_playwright

def scrap_title(page: Page) -> str:
    page.wait_for_selector("h1.headline-1", timeout=5000)
    title = page.locator("h1.headline-1").text_content()
    if title is not None:
        return title.strip()
    else: 
        return "Titre non trouvé"

def scrap_casting(page: Page) -> list[str]:
    page.wait_for_selector("a[href^='/actor/']", timeout=5000)
    casting_elements = page.locator("a[href^='/actor/']").all()
    results = []
    for element in casting_elements:
        actor_name = element.text_content()
        if actor_name is not None:
            results.append(actor_name.strip())
    return results if results else ["Casting non trouvé"]