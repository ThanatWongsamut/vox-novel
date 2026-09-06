from typing import List
from vox_novel.models.domain import Paragraph
from vox_novel.translators.base import BaseTranslator


class DummyTranslator(BaseTranslator):
    """A mock/dummy translator for testing pipeline without API keys."""

    def __init__(self, prefix: str = "[Translated] "):
        self.prefix = prefix

    @property
    def name(self) -> str:
        return "dummy"

    async def translate_text(self, text: str, target_lang: str, source_lang: str = "en") -> str:
        return f"{self.prefix}({target_lang}) {text}"

    async def translate_paragraphs(
        self,
        paragraphs: List[Paragraph],
        target_lang: str,
        source_lang: str = "en",
    ) -> List[Paragraph]:
        for p in paragraphs:
            p.translated_text = f"{self.prefix}({target_lang}) {p.text}"
        return paragraphs
