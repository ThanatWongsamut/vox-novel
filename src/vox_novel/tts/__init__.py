from vox_novel.tts.base import BaseTTS
from vox_novel.tts.dummy import DummyTTS
from vox_novel.tts.registry import tts_registry
from vox_novel.tts.voxcpm import VoxCPM2TTS

__all__ = ["BaseTTS", "VoxCPM2TTS", "DummyTTS", "tts_registry"]
