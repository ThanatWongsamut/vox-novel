from typing import Dict, List, Optional, Type
from vox_novel.scrapers.base import BaseScraper
from vox_novel.scrapers.webnovel import WebnovelScraper


class ScraperRegistry:
    """Registry allowing dynamic discovery and pluggable scrapers."""

    def __init__(self):
        self._scrapers: List[Type[BaseScraper]] = []

    def register(self, scraper_cls: Type[BaseScraper]) -> None:
        if scraper_cls not in self._scrapers:
            self._scrapers.append(scraper_cls)

    def get_scraper_for_url(self, url: str) -> BaseScraper:
        for scraper_cls in self._scrapers:
            if scraper_cls.can_handle(url):
                return scraper_cls()
        raise ValueError(f"No suitable scraper found for URL: {url}")


# Default global registry pre-populated with built-in scrapers
registry = ScraperRegistry()
registry.register(WebnovelScraper)
