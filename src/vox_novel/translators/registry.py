from typing import Dict, Type
from vox_novel.translators.base import BaseTranslator
from vox_novel.translators.dummy import DummyTranslator
from vox_novel.translators.gemini import GeminiTranslator
from vox_novel.translators.openrouter import OpenRouterTranslator


class TranslatorRegistry:
    """Registry allowing pluggable translation backends."""

    def __init__(self):
        self._translators: Dict[str, Type[BaseTranslator]] = {}

    def register(self, key: str, translator_cls: Type[BaseTranslator]) -> None:
        self._translators[key.lower()] = translator_cls

    def get_translator(self, name: str, **kwargs) -> BaseTranslator:
        key = name.lower()
        if key not in self._translators:
            available = list(self._translators.keys())
            raise ValueError(f"Translator '{name}' not found. Available: {available}")
        return self._translators[key](**kwargs)


translator_registry = TranslatorRegistry()
translator_registry.register("openrouter", OpenRouterTranslator)
translator_registry.register("gemini", GeminiTranslator)
translator_registry.register("dummy", DummyTranslator)
