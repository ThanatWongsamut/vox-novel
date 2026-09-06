from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, List, Optional
from vox_novel.models.domain import Chapter, Paragraph


class BaseTTS(ABC):
    """Abstract Base Class for Text-to-Speech engines."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        output_file: Path,
        voice_description: Optional[str] = None,
        reference_audio: Optional[Path] = None,
        emotion: Optional[str] = None,
    ) -> Path:
        """Synthesize text into an audio file."""
        pass

    @abstractmethod
    async def synthesize_chapter(
        self,
        chapter: Chapter,
        output_dir: Path,
        use_translated: bool = True,
        voice_description: Optional[str] = None,
        reference_audio: Optional[Path] = None,
        knowledge: Optional[Any] = None,
        progress_callback: Optional[Callable[[int, str], Any]] = None,
    ) -> Path:
        """Synthesize an entire chapter into audio file(s)."""
        pass
