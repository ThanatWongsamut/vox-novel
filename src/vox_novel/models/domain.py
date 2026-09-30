from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
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
        changed keeps nothing. Repeated text is carried only inside an unchanged
        sequence bounded by matching unique paragraphs or chapter boundaries;
        ambiguous occurrences need review again. Returns how many paragraphs
        kept their attribution.
        """
        if previous is None:
            return 0

        def key(p: "Paragraph") -> str:
            # Detection and review both work on the text as it will be read.
            return (p.translated_text or p.text or "").strip()

        old_keys = [key(p) for p in previous.paragraphs]
        new_keys = [key(p) for p in self.paragraphs]
        old_counts, new_counts = Counter(old_keys), Counter(new_keys)
        unique = {text for text in old_counts if text and
                  old_counts[text] == new_counts[text] == 1}
        old_positions = {text: i for i, text in enumerate(old_keys) if text in unique}
        matches = {i: old_positions[text] for i, text in enumerate(new_keys) if text in unique}

        # Unique paragraphs anchor the sequence. Only identical regions between
        # ordered anchors can safely preserve repeated dialogue. An insertion or
        # deletion in such a region leaves its duplicates unlabelled.
        anchors = [(-1, -1)]
        for block in SequenceMatcher(None, old_keys, new_keys, autojunk=False).get_matching_blocks():
            anchors.extend((block.a + offset, block.b + offset)
                           for offset in range(block.size)
                           if old_keys[block.a + offset] in unique)
        anchors.append((len(old_keys), len(new_keys)))
        for (old_left, new_left), (old_right, new_right) in zip(anchors, anchors[1:]):
            old_region = old_keys[old_left + 1:old_right]
            new_region = new_keys[new_left + 1:new_right]
            if old_region == new_region:
                for offset, text in enumerate(new_region, 1):
                    if text:
                        matches[new_left + offset] = old_left + offset

        for new_index, old_index in matches.items():
            new, old = self.paragraphs[new_index], previous.paragraphs[old_index]
            for field in INHERITED_PARAGRAPH_FIELDS:
                setattr(new, field, getattr(old, field))
        return len(matches)


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
