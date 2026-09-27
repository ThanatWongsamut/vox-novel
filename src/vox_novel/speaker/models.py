"""Schemas the model is asked to fill in.

These are sent as JSON schemas with strict structured output, so they carry no
bookkeeping fields: a strict provider must emit every declared property, and
each extra one costs tokens and invites invented values. Persisted state lives
on CharacterProfile in series_knowledge instead.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# Paragraph-level classification. "thought" is interior monologue: attributed to
# a character like dialogue, but delivered differently.
SpeechType = Literal["dialogue", "thought", "narration"]


class CharacterDraft(BaseModel):
    """A character as the model reports it."""

    name: str = Field(description="Canonical name, in the novel's language, as most commonly written")
    aliases: List[str] = Field(default_factory=list, description="Other forms used for this character: titles, nicknames, full names")
    gender: Literal["female", "male", "unknown"] = "unknown"
    speech_style: str = Field(default="", description="Pronouns, politeness particles, register -- e.g. 'ดิฉัน/ค่ะ, formal female'")
    description: str = Field(default="", description="One line on who this character is")
    is_narrator: bool = Field(default=False, description="True if this character is the point-of-view narrator")


class RegistryDraft(BaseModel):
    """Output of the extraction pass."""

    characters: List[CharacterDraft]
    narration_note: str = Field(
        default="",
        description="How the narration works, e.g. 'first-person narration by X'",
    )


class Segment(BaseModel):
    """One annotated paragraph."""

    paragraph: int = Field(description="1-based index of the source paragraph")
    type: SpeechType
    speaker: Optional[str] = Field(
        default=None,
        description="Canonical character name for dialogue/thought; null for narration. Use 'UNKNOWN' only if truly undeterminable.",
    )
    confidence: float = Field(default=1.0, description="0-1 for dialogue/thought; 1.0 for narration")
    evidence: str = Field(default="", description="Brief note on how the speaker was determined")


class ChunkAnnotation(BaseModel):
    segments: List[Segment]
    new_characters: List[CharacterDraft] = Field(
        default_factory=list,
        description="Characters who speak in this chunk but are missing from the registry",
    )
