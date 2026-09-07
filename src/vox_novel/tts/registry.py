from typing import Dict, Type
from vox_novel.tts.base import BaseTTS
from vox_novel.tts.dummy import DummyTTS
from vox_novel.tts.voxcpm import VoxCPM2TTS


class TTSRegistry:
    """Registry allowing pluggable Text-to-Speech backends."""

    def __init__(self):
        self._engines: Dict[str, Type[BaseTTS]] = {}

    def register(self, key: str, engine_cls: Type[BaseTTS]) -> None:
        self._engines[key.lower()] = engine_cls

    def get_tts(self, name: str, **kwargs) -> BaseTTS:
        key = name.lower()
        if key not in self._engines:
            available = list(self._engines.keys())
            raise ValueError(f"TTS engine '{name}' not found. Available: {available}")
        return self._engines[key](**kwargs)


tts_registry = TTSRegistry()
tts_registry.register("voxcpm2", VoxCPM2TTS)
tts_registry.register("voxcpm", VoxCPM2TTS)
tts_registry.register("dummy", DummyTTS)
