import os
from typing import List, Optional
from google import genai
from google.genai import types

from vox_novel.models.domain import Paragraph
from vox_novel.translators.base import BaseTranslator


class GeminiTranslator(BaseTranslator):
    """Translation engine powered by Google Gemini via google-genai SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None

    @property
    def name(self) -> str:
        return f"gemini ({self.model_name})"

    def _ensure_client(self):
        if not self.client:
            self.api_key = os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY environment variable is not set.")
            self.client = genai.Client(api_key=self.api_key)

    async def translate_text(self, text: str, target_lang: str, source_lang: str = "en") -> str:
        if not text.strip():
            return text
        self._ensure_client()

        prompt = (
            f"You are a professional literary novel translator. "
            f"Translate the following novel text from {source_lang} to {target_lang}. "
            f"Maintain emotional tone, storytelling style, and literary nuance. "
            f"Do not add commentary, notes, or quotes around the result. Output only the translated text.\n\n"
            f"Text:\n{text}"
        )

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return (response.text or "").strip()

    async def translate_paragraphs(
        self,
        paragraphs: List[Paragraph],
        target_lang: str,
        source_lang: str = "en",
        batch_size: int = 15,
    ) -> List[Paragraph]:
        self._ensure_client()
        if not paragraphs:
            return paragraphs

        for i in range(0, len(paragraphs), batch_size):
            chunk = paragraphs[i : i + batch_size]
            prompt = (
                f"You are a professional literary novel translator. "
                f"Translate the following numbered paragraphs of a web novel from {source_lang} to {target_lang}.\n"
                f"Requirements:\n"
                f"1. Preserve exact numbering format [INDEX] at the beginning of each paragraph.\n"
                f"2. Maintain literary nuance, dialogue emotions, sound effects, and character tone.\n"
                f"3. Return ONLY the translated paragraphs in order with their [INDEX] tags.\n\n"
                f"Original Paragraphs:\n"
            )
            for p in chunk:
                prompt += f"[{p.index}] {p.text}\n"

            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )

            raw_out = response.text or ""
            # Parse responses by [INDEX]
            import re
            pattern = re.compile(r"\[(\d+)\]\s*(.*?)(?=\n\[\d+\]|\Z)", re.S)
            found_map = {}
            for match in pattern.finditer(raw_out):
                p_idx = int(match.group(1))
                p_text = match.group(2).strip()
                found_map[p_idx] = p_text

            for p in chunk:
                if p.index in found_map:
                    p.translated_text = found_map[p.index]
                else:
                    # Fallback if specific tag missed: keep original or translate individually
                    p.translated_text = p.text

        return paragraphs
