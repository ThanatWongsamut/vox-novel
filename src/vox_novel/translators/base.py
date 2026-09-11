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

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Run a one-off instruction against the backing model.

        Translators are this project's gateway to an LLM, so callers that need a
        short completion for something other than novel text (deriving a TTS voice
        control prompt, say) can reuse the configured backend instead of wiring up
        their own client. Backends without a chat endpoint leave this unsupported;
        callers are expected to have a non-LLM fallback.
        """
        raise NotImplementedError(f"{self.name} does not support one-off completions")

    async def structured(
        self,
        system_prompt: str,
        user_prompt: str,
        output_model: type,
        temperature: float = 0.0,
    ):
        """Run an instruction and validate the reply against a Pydantic model.

        Speaker attribution needs the reply to be machine-readable, not prose.
        Backends without schema enforcement leave this unsupported; callers are
        expected to handle that rather than parse free text and hope.
        """
        raise NotImplementedError(f"{self.name} does not support structured output")

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
