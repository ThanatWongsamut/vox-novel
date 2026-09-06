from abc import ABC, abstractmethod
from typing import List, Optional
from vox_novel.models.domain import Chapter, ChapterSummary, Novel


class BaseScraper(ABC):
    """Abstract Base Class for novel scrapers."""

    @classmethod
    @abstractmethod
    def can_handle(cls, url: str) -> bool:
        """Return True if this scraper can handle the given URL."""
        pass

    @abstractmethod
    async def get_novel_info(self, url: str) -> Novel:
        """Scrape novel metadata and chapter index."""
        pass

    @abstractmethod
    async def get_chapter(self, url: str) -> Chapter:
        """Scrape chapter content including title and paragraphs."""
        pass
