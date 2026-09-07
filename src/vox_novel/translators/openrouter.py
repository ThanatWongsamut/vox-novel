import asyncio
import json
import os
import re
from typing import Dict, List, Optional, Tuple
import httpx
from dotenv import load_dotenv

from vox_novel.models.domain import Paragraph
from vox_novel.models.series_knowledge import SeriesKnowledge
from vox_novel.translators.base import BaseTranslator
from vox_novel.translators.prompts import (
    DEFAULT_THAI_SYSTEM_PROMPT,
    EDITOR_SYSTEM_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
)

load_dotenv()


class OpenRouterTranslator(BaseTranslator):
    """Translation engine using OpenRouter API supporting models like google/gemma-4-31b-it:free."""

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model_name = model or os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
        
        if fallback_models is not None:
            self.fallback_models = fallback_models
        else:
            env_fallbacks = os.getenv("OPENROUTER_FALLBACK_MODELS")
            if env_fallbacks:
                self.fallback_models = [m.strip() for m in env_fallbacks.split(",") if m.strip()]
            elif "minimax" in self.model_name:
                self.fallback_models = ["minimax/minimax-m2.7:free"]
            else:
                self.fallback_models = []

        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout

    @property
    def name(self) -> str:
        return f"openrouter ({self.model_name})"

    def _get_headers(self) -> dict:
        key = self.api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Please set it in your environment or in a .env file."
            )
        return {
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": "https://github.com/vox-novel",
            "X-Title": "VoxNovel",
            "Content-Type": "application/json",
        }

    def _build_system_prompt(self, target_lang: str, knowledge: Optional[SeriesKnowledge] = None) -> str:
        glossary_text = knowledge.format_glossary_prompt() if knowledge else ""
        if target_lang == "th":
            return DEFAULT_THAI_SYSTEM_PROMPT.format(glossary_section=glossary_text)
        
        return (
            f"You are a professional literary novel translator.\n"
            f"Translate the provided novel text from English to {target_lang} with high literary nuance, "
            f"dialogue emotions, and gaming terminology immersion.\n\n"
            f"{glossary_text}\n\n"
            f"Translate without creating a Markdown (.md) document wrapper."
        )

    async def _call_chat_completion(
        self,
        messages: list,
        temperature: float = 0.3,
        response_format: Optional[dict] = None,
        status_callback: Optional[callable] = None,
        max_retries: int = 4,
        max_tokens: Optional[int] = None,
    ) -> str:
        headers = self._get_headers()
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        # Without this OpenRouter reserves the model's whole context window, which a
        # low-balance account cannot afford even for a one-line answer.
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        # Enable OpenRouter native model fallbacks
        models_chain = [self.model_name]
        for fb in self.fallback_models:
            if fb not in models_chain:
                models_chain.append(fb)
        if len(models_chain) > 1:
            payload["models"] = models_chain

        if response_format:
            payload["response_format"] = response_format

        last_error = ""
        async with httpx.AsyncClient(headers=headers, timeout=self.timeout) as client:
            for attempt in range(max_retries):
                try:
                    resp = await client.post(f"{self.base_url}/chat/completions", json=payload)
                except httpx.RequestError as exc:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2.0 * (attempt + 1))
                        continue
                    raise RuntimeError(f"OpenRouter connection error: {exc}")

                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]

                last_error = resp.text

                # Handle HTTP 429 Rate Limit (e.g. upstream shared pool exhaustion)
                if resp.status_code == 429:
                    wait_seconds = 8 * (attempt + 1)
                    try:
                        err_json = resp.json()
                        metadata = err_json.get("error", {}).get("metadata", {})
                        if "retry_after_seconds" in metadata:
                            wait_seconds = min(int(metadata["retry_after_seconds"]), 65)
                        elif "Retry-After" in resp.headers:
                            wait_seconds = min(int(resp.headers["Retry-After"]), 65)
                    except Exception:
                        pass

                    if attempt < max_retries - 1:
                        msg = f"Rate limit reached upstream (429). Waiting {wait_seconds}s before retrying (attempt {attempt + 1}/{max_retries})..."
                        if status_callback:
                            if asyncio.iscoroutinefunction(status_callback):
                                await status_callback(msg)
                            else:
                                status_callback(msg)
                        await asyncio.sleep(wait_seconds)
                        continue

                # Handle temporary 5xx gateway/server errors
                if resp.status_code in (502, 503, 504) and attempt < max_retries - 1:
                    await asyncio.sleep(3.0 * (attempt + 1))
                    continue

                raise RuntimeError(
                    f"OpenRouter API error (Status {resp.status_code}): {resp.text}"
                )

        raise RuntimeError(
            f"OpenRouter API failed after {max_retries} attempts: {last_error}"
        )

    async def detect_new_entities_for_review(
        self,
        chapter_text: str,
        existing_knowledge: SeriesKnowledge,
        target_lang: str = "th",
        status_callback: Optional[callable] = None,
    ) -> Tuple[List[dict], List[dict]]:
        """Pre-scan chapter text BEFORE translation to detect newly introduced characters or terms."""
        known_terms = list(existing_knowledge.terms.keys())
        known_chars = list(existing_knowledge.characters.keys())

        prompt = (
            f"You are a web novel translation assistant.\n"
            f"Analyze the following chapter text (source English) and find any NEW character names, special game terms, "
            f"skills, monsters, or item names that do NOT appear in the already known list.\n\n"
            f"CRITICAL ALIAS & CHARACTER NORMALIZATION RULES:\n"
            f"1. Characters with different hyphenation or spacing (e.g. 'Ahn Yoonseung' vs 'Ahn Yoon-seung' vs 'Ahn Yoon Seung') ARE THE SAME PERSON!\n"
            f"   - Group them under ONE single primary 'name_en'!\n"
            f"   - Put the alternate spacing/hyphenation into the 'aliases' array.\n"
            f"   - Provide ONE unified Thai name for 'suggested_target' (do NOT create two separate entries).\n"
            f"2. If the same entity or boss is mentioned under synonymous titles (e.g. 'Baron of Flowers', 'Flower Garden Baron', 'Hwawon Baron'):\n"
            f"   - Group them under the primary descriptive name and put variations in 'aliases'.\n\n"
            f"Already Known Terms: {known_terms[:80]}\n"
            f"Already Known Characters: {known_chars[:50]}\n\n"
            f"Chapter Text:\n{chapter_text[:8000]}\n\n"
            f"Return ONLY a JSON object formatted exactly as:\n"
            f'{{\n'
            f'  "new_terms": [\n'
            f'    {{"source": "Primary English term", "suggested_target": "{target_lang} translation", "category": "gaming|skill|item|monster|location|general", "aliases": ["alternate name 1", "alternate name 2"], "notes": "brief context"}}\n'
            f'  ],\n'
            f'  "new_characters": [\n'
            f'    {{"name_en": "Primary English Name", "suggested_target": "{target_lang} transliteration", "gender": "male|female|unknown", "role": "e.g. Hunter / Boss", "aliases": ["alternate title or romanization"], "notes": "context"}}\n'
            f'  ]\n'
            f'}}'
        )

        messages = [
            {"role": "system", "content": "You are a professional literary entity extractor and transliterator. Strictly merge spelling variations and hyphens into aliases. Output only valid JSON."},
            {"role": "user", "content": prompt},
        ]

        try:
            raw_out = await self._call_chat_completion(messages, temperature=0.1, status_callback=status_callback)
            clean_json = re.sub(r"^```json\s*", "", raw_out.strip())
            clean_json = re.sub(r"\s*```$", "", clean_json.strip())
            data = json.loads(clean_json)

            new_terms = []
            for t in data.get("new_terms", []):
                src = t.get("source", "").strip()
                if src and src.lower() not in existing_knowledge.terms:
                    new_terms.append(t)

            new_chars = []
            for c in data.get("new_characters", []):
                name = c.get("name_en", "").strip()
                if name:
                    # Check if already known via smart normalized matching or alias
                    existing_char = existing_knowledge.find_character(name)
                    if not existing_char:
                        new_chars.append(c)
                    elif name not in existing_char.aliases and name.lower() != existing_char.name_en.lower():
                        # Append as alias to existing character
                        existing_char.aliases.append(name)

            return new_terms, new_chars
        except Exception:
            return [], []

    async def translate_text(
        self,
        text: str,
        target_lang: str,
        source_lang: str = "en",
        knowledge: Optional[SeriesKnowledge] = None,
    ) -> str:
        if not text.strip():
            return text

        sys_prompt = self._build_system_prompt(target_lang, knowledge)
        user_prompt = f"Translate the following novel title/heading into {target_lang}:\n{text}"
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        res = await self._call_chat_completion(messages)
        cleaned = re.sub(r"^\s*\[\d+\]\s*", "", res.strip()).strip()
        return cleaned

    # A one-off completion is a short answer, not a chapter.
    COMPLETION_MAX_TOKENS = 256

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        return (
            await self._call_chat_completion(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.COMPLETION_MAX_TOKENS,
            )
        ).strip()

    @staticmethod
    def _is_punctuation_or_pause(text: str) -> bool:
        clean = text.strip()
        if not clean:
            return True
        return bool(re.fullmatch(r"^[.\s\-—_~*!?,;:()\[\]{}'\"]+$", clean))

    @staticmethod
    def _clean_translated_text(text: str, original: str) -> str:
        # Strip any accidental residual index tag like "[1]" or "[1] ."
        cleaned = re.sub(r"^\s*\[\d+\]\s*", "", text).strip()
        # If the LLM outputted assistant boilerplate for short punctuation lines
        if re.search(r"(forgot to include|please paste|don't have any text|only sent a period|what would you like me to translate)", cleaned, re.IGNORECASE):
            return original.strip() or "."
        return cleaned

    async def translate_paragraphs(
        self,
        paragraphs: List[Paragraph],
        target_lang: str,
        source_lang: str = "en",
        knowledge: Optional[SeriesKnowledge] = None,
        progress_callback: Optional[callable] = None,
        batch_size: int = 15,
    ) -> List[Paragraph]:
        if not paragraphs:
            return paragraphs

        sys_prompt = self._build_system_prompt(target_lang, knowledge)

        # Pre-pass: automatically handle lonely pause lines (e.g. ".", "...", "---")
        to_translate = []
        for p in paragraphs:
            if self._is_punctuation_or_pause(p.text):
                p.translated_text = p.text.strip()
            else:
                to_translate.append(p)

        total_batches = (len(to_translate) + batch_size - 1) // batch_size if to_translate else 1

        for batch_idx, i in enumerate(range(0, len(to_translate), batch_size), start=1):
            chunk = to_translate[i : i + batch_size]

            if progress_callback:
                pct = int((batch_idx - 1) / total_batches * 65) + 15  # Scale between 15% and 80%
                msg = f"Translating draft batch {batch_idx}/{total_batches} (paragraphs {i+1} to {min(i+batch_size, len(to_translate))})..."
                if asyncio.iscoroutinefunction(progress_callback):
                    await progress_callback(pct, msg)
                else:
                    progress_callback(pct, msg)

            prompt = (
                f"Translate the following numbered paragraphs into {target_lang}.\n"
                f"Strict instructions:\n"
                f"1. You MUST keep the exact [INDEX] format at the beginning of each paragraph.\n"
                f"2. Follow the series glossary and rules strictly.\n"
                f"3. Return ONLY the translated paragraphs in order with their [INDEX] tags.\n\n"
                f"Paragraphs to translate:\n"
            )
            for p in chunk:
                prompt += f"[{p.index}] {p.text}\n"

            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ]

            async def _rate_limit_cb(cooldown_msg: str):
                if progress_callback:
                    if asyncio.iscoroutinefunction(progress_callback):
                        await progress_callback(pct, cooldown_msg)
                    else:
                        progress_callback(pct, cooldown_msg)

            raw_out = await self._call_chat_completion(messages, status_callback=_rate_limit_cb)

            pattern = re.compile(r"\[(\d+)\]\s*(.*?)(?=(?:\n\s*\[\d+\])|\Z)", re.S)
            found_map = {}
            for match in pattern.finditer(raw_out):
                p_idx = int(match.group(1))
                p_text = match.group(2).strip()
                found_map[p_idx] = p_text

            for p in chunk:
                if p.index in found_map and found_map[p.index]:
                    p.translated_text = self._clean_translated_text(found_map[p.index], p.text)
                else:
                    single_prompt = f"Translate this single paragraph into {target_lang}:\n{p.text}"
                    try:
                        raw_single = await self._call_chat_completion([
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": single_prompt},
                        ], status_callback=_rate_limit_cb)
                        p.translated_text = self._clean_translated_text(raw_single, p.text)
                    except Exception:
                        p.translated_text = p.text

            # Gentle pacing between batch calls to prevent hitting upstream rate limits
            if batch_idx < total_batches:
                await asyncio.sleep(1.2)

        # Ensure all paragraphs have clean text without leading tags
        for p in paragraphs:
            if p.translated_text:
                p.translated_text = self._clean_translated_text(p.translated_text, p.text)

        if progress_callback:
            msg = "Draft translation complete. Ready for polishing/saving..."
            if asyncio.iscoroutinefunction(progress_callback):
                await progress_callback(80, msg)
            else:
                progress_callback(80, msg)

        return paragraphs

    async def polish_paragraphs_agent(
        self,
        paragraphs: List[Paragraph],
        target_lang: str = "th",
        knowledge: Optional[SeriesKnowledge] = None,
        progress_callback: Optional[callable] = None,
        batch_size: int = 25,
    ) -> List[Paragraph]:
        """Agentic Editor Pass: Critiques and refines the draft for natural Thai prose and consistency."""
        if not paragraphs or target_lang != "th":
            return paragraphs

        glossary_text = knowledge.format_glossary_prompt() if knowledge else ""
        editor_sys_prompt = EDITOR_SYSTEM_PROMPT.format(glossary_section=glossary_text)

        # Filter out lonely punctuation lines
        to_polish = [p for p in paragraphs if not self._is_punctuation_or_pause(p.text) and p.translated_text]
        total_batches = (len(to_polish) + batch_size - 1) // batch_size if to_polish else 1

        for batch_idx, i in enumerate(range(0, len(to_polish), batch_size), start=1):
            chunk = to_polish[i : i + batch_size]

            if progress_callback:
                pct = int((batch_idx - 1) / total_batches * 15) + 80  # 80% to 95%
                msg = f"Agentic Editor: Polishing & checking consistency (batch {batch_idx}/{total_batches})..."
                if asyncio.iscoroutinefunction(progress_callback):
                    await progress_callback(pct, msg)
                else:
                    progress_callback(pct, msg)

            prompt = (
                "Critique and polish the following draft translation paragraphs.\n"
                "Refine any stiff phrasing, remove awkward passive voice, unify pronouns, and ensure full glossary adherence.\n"
                "Return ONLY the polished paragraphs preserving their exact [INDEX] tags:\n\n"
            )
            for p in chunk:
                prompt += f"[{p.index}]\nDraft: {p.translated_text}\nOriginal: {p.text}\n\n"

            messages = [
                {"role": "system", "content": editor_sys_prompt},
                {"role": "user", "content": prompt},
            ]

            async def _rate_limit_cb(cooldown_msg: str):
                if progress_callback:
                    if asyncio.iscoroutinefunction(progress_callback):
                        await progress_callback(pct, cooldown_msg)
                    else:
                        progress_callback(pct, cooldown_msg)

            try:
                raw_out = await self._call_chat_completion(messages, temperature=0.2, status_callback=_rate_limit_cb)
                pattern = re.compile(r"\[(\d+)\]\s*(.*?)(?=(?:\n\s*\[\d+\])|\Z)", re.S)
                for match in pattern.finditer(raw_out):
                    p_idx = int(match.group(1))
                    polished_text = match.group(2).strip()
                    # Clean any accidental labels like "Polished:" or "Draft:"
                    polished_text = re.sub(r"^(?:Polished|Translation|Draft)\s*:\s*", "", polished_text, flags=re.IGNORECASE).strip()
                    for p in chunk:
                        if p.index == p_idx and len(polished_text) > 3:
                            p.translated_text = self._clean_translated_text(polished_text, p.text)
            except Exception:
                # If editor pass fails for any reason, keep the solid draft
                pass

            if batch_idx < total_batches:
                await asyncio.sleep(1.2)

        if progress_callback:
            msg = "Agentic polishing complete! Finalizing and saving..."
            if asyncio.iscoroutinefunction(progress_callback):
                await progress_callback(95, msg)
            else:
                progress_callback(95, msg)

        return paragraphs

    async def extract_novel_knowledge(
        self,
        sample_text: str,
        translated_text: str,
        existing_knowledge: SeriesKnowledge,
    ) -> SeriesKnowledge:
        prompt = (
            f"Analyze this novel chapter to identify NEW terms or character names not yet in this existing glossary:\n"
            f"Current Known Terms: {list(existing_knowledge.terms.keys())}\n"
            f"Current Known Characters: {list(existing_knowledge.characters.keys())}\n\n"
            f"Original Chapter Excerpt:\n{sample_text[:3000]}\n\n"
            f"Translated Excerpt:\n{translated_text[:3000]}"
        )

        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            raw_json = await self._call_chat_completion(messages, temperature=0.1)
            clean_json = re.sub(r"^```json\s*", "", raw_json.strip())
            clean_json = re.sub(r"\s*```$", "", clean_json.strip())
            data = json.loads(clean_json)

            for term in data.get("new_terms", []):
                src = term.get("source", "").strip()
                tgt = term.get("target", "").strip()
                cat = term.get("category", "general")
                notes = term.get("notes")
                if src and tgt and src.lower() not in existing_knowledge.terms:
                    existing_knowledge.add_term(src, tgt, category=cat, notes=notes)

            for char in data.get("new_characters", []):
                name_en = char.get("name_en", "").strip()
                name_tgt = char.get("name_target", "").strip()
                role = char.get("role")
                notes = char.get("notes")
                if name_en and name_tgt and name_en.lower() not in existing_knowledge.characters:
                    existing_knowledge.add_character(name_en, name_tgt, role=role, notes=notes)

        except Exception:
            pass

        return existing_knowledge
