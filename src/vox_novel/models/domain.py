from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Paragraph(BaseModel):
    id: Optional[str] = None
    index: int
    text: str
    translated_text: Optional[str] = None
    audio_path: Optional[str] = None
    # e.g. "narrator" or a character name. Not populated yet -- the TTS engine's
    # per-character voice path reads it and stays inert until something does.
    speaker: Optional[str] = None
    emotion: Optional[str] = None  # e.g. "calm", "angry", "fearful", "whisper", "urgent"
    speech_type: Optional[str] = None  # "narration", "dialogue" or "thought"
    # True once a human has confirmed or corrected the attribution. Detection
    # leaves it False, so a re-run can overwrite its own guesses without
    # discarding reviewed lines -- and a verified set doubles as gold labels.
    speaker_verified: bool = False
    # What detection guessed, kept even after a human corrects `speaker`.
    # Without it a review destroys the evidence needed to score the model, which
    # is the whole point of reviewing; with it, verified lines are a gold set.
    speaker_detected: Optional[str] = None
    speech_type_detected: Optional[str] = None


class Chapter(BaseModel):
    id: str
    book_id: str
    title: str
    url: str
    chapter_number: Optional[float] = None
    paragraphs: List[Paragraph] = Field(default_factory=list)
    translated_title: Optional[str] = None
    source_language: str = "en"
    target_language: Optional[str] = None
    audio_path: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.paragraphs if p.text.strip())

    @property
    def translated_full_text(self) -> str:
        return "\n\n".join(
            p.translated_text or p.text for p in self.paragraphs if (p.translated_text or p.text).strip()
        )


class ChapterSummary(BaseModel):
    id: str
    book_id: str
    title: str
    url: str
    chapter_number: Optional[float] = None
    is_locked: bool = False
    has_audio: bool = False


class Novel(BaseModel):
    id: str
    title: str
    url: str
    author: Optional[str] = None
    cover_url: Optional[str] = None
    synopsis: Optional[str] = None
    source: str = "webnovel"
    chapters: List[ChapterSummary] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
