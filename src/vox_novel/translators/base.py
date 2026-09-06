from abc import ABC, abstractmethod
from typing import List, Optional
from vox_novel.models.domain import Chapter, Paragraph


class BaseTranslator(ABC):
    """Abstract Base Class for translation backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the translator backend."""
        pass

    @abstractmethod
    async def translate_text(self, text: str, target_lang: str, source_lang: str = "en") -> str:
        """Translate a single block of text."""
        pass

    @abstractmethod
    async def translate_paragraphs(
        self,
        paragraphs: List[Paragraph],
        target_lang: str,
        source_lang: str = "en",
        progress_callback: Optional[callable] = None,
    ) -> List[Paragraph]:
        """Translate a list of paragraphs in place or return updated paragraphs."""
        pass

    async def translate_chapter(
        self,
        chapter: Chapter,
        target_lang: str,
        source_lang: Optional[str] = None,
    ) -> Chapter:
        """Translate an entire chapter including title and paragraphs."""
        src = source_lang or chapter.source_language
        translated_title = await self.translate_text(chapter.title, target_lang=target_lang, source_lang=src)
        translated_paras = await self.translate_paragraphs(chapter.paragraphs, target_lang=target_lang, source_lang=src)
        
        chapter.translated_title = translated_title
        chapter.paragraphs = translated_paras
        chapter.target_language = target_lang
        return chapter
