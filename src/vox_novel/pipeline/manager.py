from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple
from vox_novel.models.domain import Chapter, Novel
from vox_novel.models.series_knowledge import SeriesKnowledge
from vox_novel.scrapers.registry import registry as default_scraper_registry
from vox_novel.storage.file import StorageManager
from vox_novel.storage.knowledge import KnowledgeManager
from vox_novel.translators.openrouter import OpenRouterTranslator
from vox_novel.translators.registry import translator_registry
from vox_novel.tts.registry import tts_registry


class NovelPipeline:
    """Orchestrates scraping, entity discovery/review, translating, knowledge retention, and storage."""

    def __init__(
        self,
        scraper_registry=None,
        storage_manager: Optional[StorageManager] = None,
        knowledge_manager: Optional[KnowledgeManager] = None,
    ):
        self.scraper_registry = scraper_registry or default_scraper_registry
        self.storage = storage_manager or StorageManager()
        self.knowledge = knowledge_manager or KnowledgeManager()

    async def scrape_novel(self, url: str) -> Novel:
        scraper = self.scraper_registry.get_scraper_for_url(url)
        novel = await scraper.get_novel_info(url)
        self.storage.save_novel_metadata(novel)
        return novel

    async def get_chapter_raw(self, url: str) -> Chapter:
        scraper = self.scraper_registry.get_scraper_for_url(url)
        return await scraper.get_chapter(url)

    async def pre_scan_chapter_entities(
        self,
        chapter: Chapter,
        target_lang: str = "th",
        translator_name: str = "openrouter",
        model: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[List[dict], List[dict], SeriesKnowledge]:
        """Detect newly introduced terms and characters for user review before translating."""
        series_knowledge = self.knowledge.load_or_init(
            series_id=chapter.book_id, target_lang=target_lang
        )

        if chapter.source_language == target_lang:
            return [], [], series_knowledge

        chosen_model = model or os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
        translator = translator_registry.get_translator(translator_name, model=chosen_model)
        if isinstance(translator, OpenRouterTranslator):
            new_terms, new_chars = await translator.detect_new_entities_for_review(
                chapter_text=chapter.full_text,
                existing_knowledge=series_knowledge,
                target_lang=target_lang,
                status_callback=progress_callback,
            )
            return new_terms, new_chars, series_knowledge

        return [], [], series_knowledge

    async def scrape_and_translate_chapter(
        self,
        url: str,
        target_lang: Optional[str] = "th",
        translator_name: str = "openrouter",
        translator_kwargs: Optional[dict] = None,
        auto_learn: bool = True,
        chapter_obj: Optional[Chapter] = None,
        progress_callback: Optional[callable] = None,
        agentic_mode: bool = True,
    ) -> Chapter:
        chapter = chapter_obj or await self.get_chapter_raw(url)

        # If chapter source language already matches target language (e.g. Readtoon is already Thai):
        if target_lang and chapter.source_language == target_lang:
            chapter.target_language = target_lang
            if not chapter.translated_title:
                chapter.translated_title = chapter.title
            for p in chapter.paragraphs:
                if not p.translated_text:
                    p.translated_text = p.text
            if progress_callback:
                import inspect
                cb_res = progress_callback(100, "Content already in Thai. Ingested directly!")
                if inspect.isawaitable(cb_res):
                    await cb_res
            self.storage.save_chapter(chapter)
            return chapter

        if target_lang:
            kwargs = translator_kwargs or {}

            translator = translator_registry.get_translator(translator_name, **kwargs)

            series_knowledge = self.knowledge.load_or_init(
                series_id=chapter.book_id, target_lang=target_lang
            )

            if isinstance(translator, OpenRouterTranslator):
                chapter.translated_title = await translator.translate_text(
                    chapter.title,
                    target_lang=target_lang,
                    knowledge=series_knowledge,
                )
                chapter.paragraphs = await translator.translate_paragraphs(
                    chapter.paragraphs,
                    target_lang=target_lang,
                    knowledge=series_knowledge,
                    progress_callback=progress_callback,
                )
                chapter.target_language = target_lang

                # Agentic Critic/Editor Step: Polishes draft for natural Thai flow & strict glossary alignment
                if agentic_mode:
                    # The editor pass sees both the draft and the English original, so
                    # it can repair meaning as well as phrasing -- it is where a
                    # stronger model earns its cost. It also batches 25 paragraphs to
                    # the draft's 15, so running the better model only here is cheaper
                    # than running it only on the draft.
                    polish_translator = self._polish_translator(translator, translator_name, kwargs)
                    chapter.paragraphs = await polish_translator.polish_paragraphs_agent(
                        chapter.paragraphs,
                        target_lang=target_lang,
                        knowledge=series_knowledge,
                        progress_callback=progress_callback,
                    )

                if auto_learn:
                    sample_orig = chapter.full_text[:4000]
                    sample_trans = chapter.translated_full_text[:4000]
                    updated_knowledge = await translator.extract_novel_knowledge(
                        sample_orig, sample_trans, series_knowledge
                    )
                    self.knowledge.save(updated_knowledge)
            else:
                chapter = await translator.translate_chapter(chapter, target_lang=target_lang)

        self.storage.save_chapter(chapter)
        return chapter

    async def synthesize_chapter_audio(
        self,
        series_id: str,
        chapter_id: str,
        engine_name: str = "voxcpm2",
        voice_description: Optional[str] = None,
        reference_audio: Optional[Path] = None,
        progress_callback: Optional[Callable[[int, str], Any]] = None,
        engine_kwargs: Optional[dict] = None,
    ) -> Path:
        """Synthesize chapter text into master audio using the selected TTS engine."""
        chapter = self.storage.get_chapter(series_id, chapter_id)
        if not chapter:
            raise ValueError(f"Chapter '{chapter_id}' not found in series '{series_id}'")

        knowledge = self.knowledge.load_or_init(
            series_id, target_lang=chapter.target_language or "th"
        )
        ref_audio = reference_audio or (
            Path(knowledge.narrator_voice_ref_audio)
            if knowledge.narrator_voice_ref_audio and Path(knowledge.narrator_voice_ref_audio).exists()
            else None
        )

        kwargs = engine_kwargs or {}
        tts_engine = tts_registry.get_tts(engine_name, **kwargs)

        output_dir = self.storage.base_dir / series_id / "chapters"
        audio_path = await tts_engine.synthesize_chapter(
            chapter=chapter,
            output_dir=output_dir,
            use_translated=True,
            # Pass the per-chapter override as-is. synthesize_chapter falls back to
            # knowledge on its own; collapsing the two here would make every call
            # indistinguishable from an override and defeat the prompt cache.
            voice_description=voice_description,
            reference_audio=ref_audio,
            knowledge=knowledge,
            progress_callback=progress_callback,
            translator=self.voice_prompt_translator(),
        )

        # synthesize_chapter caches the control prompts it derived; persist only
        # those fields. A full save would overwrite any glossary or voice edit made
        # while this multi-minute job was running.
        self._persist_derived_control_prompts(
            series_id, knowledge, overridden=bool(voice_description)
        )
        chapter.audio_path = str(audio_path)
        self.storage.save_chapter(chapter)
        return audio_path

    def _persist_derived_control_prompts(self, series_id: str, knowledge, overridden: bool) -> None:
        """Backfill control prompts for a series no endpoint has derived one for.

        The Voice Studio endpoints derive at save time, so synthesis only needs to
        fill a gap -- never to overwrite. Writing only into empty fields removes the
        need to track which description each prompt came from: an edit made during
        this multi-minute job is simply left alone, and a per-chapter override is
        never mistaken for the series voice.
        """
        if overridden:
            return
        fresh = self.knowledge.load_or_init(series_id, target_lang=knowledge.target_language)

        def _same_description(stored: Optional[str], used: Optional[str]) -> bool:
            # A description change clears the cached prompt rather than replacing
            # it, so an empty field can mean either "never derived" or "just
            # edited". Comparing the description this job derived from tells them
            # apart -- and unlike a whole-file timestamp, an unrelated glossary
            # write during the job does not suppress the backfill.
            return (stored or "").strip() == (used or "").strip()

        dirty = False
        if (
            knowledge.narrator_voice_control_prompt
            and not fresh.narrator_voice_control_prompt
            and _same_description(
                fresh.narrator_voice_description, knowledge.narrator_voice_description
            )
        ):
            fresh.narrator_voice_control_prompt = knowledge.narrator_voice_control_prompt
            dirty = True
        for char in knowledge.characters.values():
            target = fresh.find_character(char.name_en)
            if (
                target is not None
                and char.voice_control_prompt
                and not target.voice_control_prompt
                and _same_description(target.voice_description, char.voice_description)
            ):
                target.voice_control_prompt = char.voice_control_prompt
                dirty = True
        if dirty:
            self.knowledge.save(fresh)

    @staticmethod
    def _polish_translator(draft_translator, translator_name: str, kwargs: dict):
        """Return the translator to use for the editor pass.

        OPENROUTER_POLISH_MODEL lets the editor pass run on a different (usually
        stronger) model than the draft. Unset, the draft translator is reused, so
        behaviour is unchanged.
        """
        import os

        polish_model = os.getenv("OPENROUTER_POLISH_MODEL", "").strip()
        if not polish_model or polish_model == getattr(draft_translator, "model_name", None):
            return draft_translator
        try:
            return translator_registry.get_translator(
                translator_name, **{**kwargs, "model": polish_model}
            )
        except Exception:
            return draft_translator

    @staticmethod
    def voice_prompt_translator():
        """The translator used to turn voice descriptions into English control prompts.

        Returns None when no LLM is configured, which leaves voice design on the
        offline keyword table.
        """
        import os

        if not os.getenv("OPENROUTER_API_KEY"):
            return None
        try:
            return translator_registry.get_translator(
                "openrouter", model=os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
            )
        except Exception:
            return None
