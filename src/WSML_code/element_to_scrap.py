from playwright.sync_api import Page, sync_playwright

class Scraping:
    @staticmethod
    def scrap_title(page: Page) -> str:
        """La fonction scrap_title nous permet de récupérer le titre du film depuis la page letterboxd en se basant sur le selecteur CSS

        Args:
            page (Page): _description_

        Returns:
            str: _description_
        """
        page.wait_for_selector("h1.headline-1", timeout=5000)
        title = page.locator("h1.headline-1").text_content()
        if title is not None:
            return title.strip()
        else: 
            return ""

    @staticmethod
    def scrap_director(page: Page) -> list[str]:
        """
        La fonction scrap_director nous permet de récupérer le réalisateur du film depuis la page letterboxd en se basant sur le selecteur CSS

        Args:
            page (Page): La page à partir de laquelle extraire les informations

        Returns:
            list[str]: Une liste des noms des réalisateurs
        """
        page.wait_for_selector("a[href^='/director/']", timeout=5000)
        director = page.locator("a[href^='/director/']").all()
        res_dir = []
        for dir in director:
            director_name = dir.text_content()
            if director_name is not None and director_name.strip() not in res_dir:
                res_dir.append(director_name.strip())
        return res_dir if res_dir else ["Directeur non trouvé"]

#page CAST
    @staticmethod
    def scrap_casting(page: Page) -> list[str]:
        """
        La fonction scrap_casting nous permet de récupérer le casting du film depuis la page letterboxd en se basant sur le selecteur CSS
        Args:
            page (Page): La page à partir de laquelle extraire les informations
        Returns:
            list[str]: Une liste des noms des acteurs
        """
        # attendre que les éléments soient présents dans le DOM (attachés)
        try:
            page.wait_for_selector("a[href^='/actor/']", timeout=5000, state='attached')
        except Exception:
            return ["Casting non trouvé"]

        # récupérer directement les textes; ils peuvent être dans le DOM mais masqués
        try:
            texts = page.locator("a[href^='/actor/']").all_text_contents()
        except Exception:
            return ["Casting non trouvé"]

        res_cast = [t.strip() for t in texts if t and t.strip()][:10]
        return res_cast if res_cast else ["Casting non trouvé"]

#page CREW
    @staticmethod
    def scrap_producers(page: Page) -> list[str]:
        """
        La fonction scrap_producers nous permet de récupérer les producteurs du film depuis la page letterboxd en se basant sur le selecteur CSS

        Args:
            page (Page): La page à partir de laquelle extraire les informations

        Returns:
            list[str]: Une liste des noms des membres de l'équipe
        """
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
        """
        La fonction scrap_writers nous permet de récupérer les scénaristes du film depuis la page letterboxd en se basant sur le selecteur CSS

        Args:
            page (Page): La page à partir de laquelle extraire les informations

        Returns:
            list[str]: Une liste des noms des scénaristes
        """
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
        """
        La fonction scrap_composer nous permet de récupérer le compositeur du film depuis la page letterboxd en se basant sur le selecteur CSS

        Args:
            page (Page): La page à partir de laquelle extraire les informations

        Returns:
            list[str]: Une liste des noms des compositeurs
        """
        page.click("a[href^='/film/'][href$='/crew/']")
        page.wait_for_selector("a[href^='/composer/']", timeout=5000)
        composers = page.locator("a[href^='/composer/']").all()
        res_composers = []
        for composer in composers:
            composer_name = composer.text_content()
            if composer_name is not None and composer_name.strip() not in res_composers:
                res_composers.append(composer_name.strip())
        return res_composers if res_composers else ["Compositeurs non trouvés"]

    @staticmethod
    def scrap_date(page: Page) -> str:
        """
        La fonction scrap_date nous permet de récupérer la date de sortie du film depuis la page letterboxd en se basant sur le selecteur CSS

        Args:
            page (Page): La page à partir de laquelle extraire les informations

        Returns:
            str: La date de sortie du film
        """
        page.click("a[href^='/film/'][href$='/crew/']")
        page.wait_for_selector("a[href^='/films/year/']", timeout=5000)
        date = page.locator("a[href^='/films/year/']").text_content()
        if date is not None:
            return date.strip()
        else: 
            return "Date non trouvée"
        
# page GENRES
    @staticmethod
    def scrap_genres(page: Page) -> list[str]:
        """
        La fonction scrap_genres nous permet de récupérer les genres du film depuis la page letterboxd en se basant sur le selecteur CSS

        Args:
            page (Page): La page à partir de laquelle extraire les informations

        Returns:
            list[str]: Une liste des genres du film
        """
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
        """
        La fonction scrap_themes nous permet de récupérer les thèmes du film depuis la page letterboxd en se basant sur le selecteur CSS

        Args:
            page (Page): La page à partir de laquelle extraire les informations

        Returns:
            list[str]: Une liste des thèmes du film
        """
        page.click("a[href^='/film/'][href$='/genres/']")
        page.wait_for_selector("a[href^='/films/theme/'], a[href^='/films/mini-theme/']", timeout=5000)
        themes_elements = page.locator("a[href^='/films/theme/'], a[href^='/films/mini-theme/']").all()
        res_themes = []
        for element in themes_elements:
            theme_name = element.text_content()
            if theme_name is not None:
                res_themes.append(theme_name.strip())
        return res_themes if res_themes else ["Thèmes non trouvés"]
    
