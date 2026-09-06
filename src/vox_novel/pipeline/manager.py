from pathlib import Path
from typing import List, Optional, Tuple
from vox_novel.models.domain import Chapter, Novel
from vox_novel.models.series_knowledge import SeriesKnowledge
from vox_novel.scrapers.registry import registry as default_scraper_registry
from vox_novel.storage.file import StorageManager
from vox_novel.storage.knowledge import KnowledgeManager
from vox_novel.translators.openrouter import OpenRouterTranslator
from vox_novel.translators.registry import translator_registry


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

        chosen_model = model or os.getenv("OPENROUTER_MODEL", "minimax/minimax-m3:free")
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
                    chapter.paragraphs = await translator.polish_paragraphs_agent(
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
