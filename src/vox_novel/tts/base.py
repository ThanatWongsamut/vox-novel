from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from vox_novel.models.domain import Chapter, Paragraph


class BaseTTS(ABC):
    """Abstract Base Class for Text-to-Speech engines."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    async def synthesize(self, text: str, output_file: Path, voice: Optional[str] = None) -> Path:
        """Synthesize text into an audio file."""
        pass

    @abstractmethod
    async def synthesize_chapter(
        self,
        chapter: Chapter,
        output_dir: Path,
        use_translated: bool = True,
        voice: Optional[str] = None,
    ) -> Path:
        """Synthesize an entire chapter into audio file(s)."""
        pass
