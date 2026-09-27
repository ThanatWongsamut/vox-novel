from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# Per-paragraph work done after a chapter is fetched: attribution, a human's
# verdict on it, and voice choices. None of it can be fetched again, so a
# re-fetch carries it over -- see Chapter.inherit_attribution.
INHERITED_PARAGRAPH_FIELDS = (
    "speaker", "speech_type", "speaker_verified", "speaker_detected",
    "speech_type_detected", "voice_override", "emotion",
)


class Paragraph(BaseModel):
    id: Optional[str] = None
    index: int
    text: str
    translated_text: Optional[str] = None
    audio_path: Optional[str] = None
    # The mind responsible for the line, or None for narration. TTS separately
    # resolves which body (and therefore which voice) that mind has here.
    speaker: Optional[str] = None
    # An exceptional line may deliberately use another voice (e.g. telepathy or
    # imitation). A canonical character name, or "narrator".
    voice_override: Optional[str] = None
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

    def inherit_attribution(self, previous: Optional["Chapter"]) -> int:
        """Keep a previous copy's attribution on paragraphs whose text is unchanged.

        Fetching a chapter again builds it from scratch, which used to discard
        every speaker label on it -- including the ones a human verified, which
        are the only gold labels speaker detection is scored against.

        Paragraphs are matched by their text, not their position, so a line
        added or removed upstream shifts nothing after it. A paragraph whose text
        changed keeps nothing: a label is only known to be right for the words it
        was made on. Repeated lines pair up in order. Returns how many paragraphs
        kept their attribution.
        """
        if previous is None:
            return 0

        def key(p: "Paragraph") -> str:
            # Detection and review both work on the text as it will be read.
            return (p.translated_text or p.text or "").strip()

        unused: Dict[str, List["Paragraph"]] = {}
        for old in previous.paragraphs:
            unused.setdefault(key(old), []).append(old)

        kept = 0
        for new in self.paragraphs:
            candidates = unused.get(key(new))
            if not candidates:
                continue
            old = candidates.pop(0)
            for field in INHERITED_PARAGRAPH_FIELDS:
                setattr(new, field, getattr(old, field))
            kept += 1
        return kept


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
