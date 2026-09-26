import re
from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, FiniteFloat, model_validator


class TermMapping(BaseModel):
    source: str
    target: str
    category: str = "general"  # gaming, character, location, item, skill, monster, honorific
    aliases: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class StoryPosition(BaseModel):
    """A paragraph boundary in story order. The paragraph at this position is included."""

    chapter_number: FiniteFloat
    paragraph_index: int = Field(default=1, ge=1)

    def key(self) -> tuple[float, int]:
        return self.chapter_number, self.paragraph_index


class OccupancyInterval(BaseModel):
    """A mind occupies a body from start until end (exclusive)."""

    body: str
    start: StoryPosition
    end: Optional[StoryPosition] = None

    @model_validator(mode="after")
    def valid_range(self):
        if self.end is not None and self.end.key() <= self.start.key():
            raise ValueError("occupancy end must be after start")
        return self


class CharacterProfile(BaseModel):
    name_en: str
    name_target: str
    gender: Optional[str] = None
    role: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    relationship_tones: Dict[str, str] = Field(default_factory=dict)
    notes: Optional[str] = None
    voice_description: Optional[str] = None  # Text prompt for Voice Design
    voice_ref_audio: Optional[str] = None    # Path to saved reference audio clip
    # English control prompt derived from voice_description. VoxCPM2 only acts on
    # English, so this is derived once (LLM when available, keyword table otherwise)
    # and reused for every paragraph rather than recomputed per synthesis.
    voice_control_prompt: Optional[str] = None
    # Speaker attribution. is_narrator is the highest-value field in the registry:
    # the benchmark that drove this design went from 87% to 100% accuracy by
    # correcting nothing but the narrator entry. speech_style carries the pronouns
    # and particles that distinguish speakers in Thai (ดิฉัน/ค่ะ vs ฉัน).
    is_narrator: bool = False
    speech_style: Optional[str] = None
    line_count: int = 0
    last_seen_chapter: Optional[str] = None
    # Stored on the mind. Multiple bounded intervals support changes within a
    # chapter, returns, and later swaps. The old fields remain as a compatibility
    # mirror for a single open-ended interval in existing knowledge files/API.
    occupancy_intervals: List[OccupancyInterval] = Field(default_factory=list)
    inhabiting: Optional[str] = None
    inhabiting_from_chapter: Optional[float] = None

    @model_validator(mode="after")
    def migrate_legacy_occupancy(self):
        if self.inhabiting and self.inhabiting_from_chapter is not None and not self.occupancy_intervals:
            self.occupancy_intervals = [OccupancyInterval(
                body=self.inhabiting,
                start=StoryPosition(chapter_number=self.inhabiting_from_chapter),
            )]
        return self


class SeriesKnowledge(BaseModel):
    """Cumulative glossary and lore memory for a novel series."""

    series_id: str
    target_language: str = "th"
    source_language: str = "en"
    terms: Dict[str, TermMapping] = Field(default_factory=dict)
    characters: Dict[str, CharacterProfile] = Field(default_factory=dict)
    narrator_voice_description: Optional[str] = (
        "เสียงบรรยายผู้ชาย นุ่มลึก มีชีวิตชีวา ชัดถ้อยชัดคำ เหมาะกับการเล่านิยายแฟนตาซี"
    )
    narrator_voice_ref_audio: Optional[str] = None
    narrator_voice_control_prompt: Optional[str] = None
    # e.g. "first-person narration by X; third-person following Y in some sections"
    narration_note: Optional[str] = None
    style_guidelines: List[str] = Field(default_factory=list)
    custom_system_prompt: Optional[str] = None
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def validate_occupancy_timeline(self):
        """Refuse persisted timelines that would make voice choice ambiguous."""
        entries = []
        for mind in self.characters.values():
            for interval in mind.occupancy_intervals:
                body = self.find_character(interval.body, allow_prefix=False)
                if body is None or body.name_en != interval.body:
                    raise ValueError(f"unknown occupancy body {interval.body!r}")
                if body.name_en == mind.name_en:
                    raise ValueError("a character cannot inhabit themselves")
                for other_mind, other in entries:
                    if not self._overlap(interval, other):
                        continue
                    if other_mind == mind.name_en or other.body == interval.body:
                        raise ValueError("overlapping occupancy periods")
                entries.append((mind.name_en, interval))
        return self

    def add_term(
        self,
        source: str,
        target: str,
        category: str = "general",
        aliases: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ):
        key = source.strip().lower()
        alias_list = [a.strip() for a in (aliases or []) if a.strip() and a.strip().lower() != key]
        self.terms[key] = TermMapping(
            source=source.strip(),
            target=target.strip(),
            category=category,
            aliases=alias_list,
            notes=notes,
        )
        self.last_updated = datetime.utcnow()

    @staticmethod
    def _normalize_name_key(name: str) -> str:
        """Normalize a name for matching.

        Separators collapse to a single underscore rather than vanishing: deleting
        them made "An Na" and "Anna", or "Li Wei" and "Liwei", the same character,
        which silently merges two people onto one voice. Characters outside Thai
        and ASCII are dropped -- models occasionally corrupt Thai names with stray
        CJK tokens, and those spellings should still resolve.
        """
        kept = [
            ch for ch in name.strip().lower()
            if ch.isascii() or "\u0e00" <= ch <= "\u0e7f"
        ]
        return re.sub(r"[\s\-_]+", "_", "".join(kept)).strip("_")

    @classmethod
    def _prefix_compatible(cls, a: str, b: str) -> bool:
        """True when two normalized names look like the same name, one truncated.

        Long enough that the overlap means something -- this is what catches a
        corrupted or shortened spelling without merging genuinely short names.
        """
        return len(a) >= 4 and len(b) >= 4 and (a.startswith(b) or b.startswith(a))

    def near_duplicate_characters(self) -> List[tuple]:
        """Entries that look like one character under two spellings.

        Registration deliberately does not fold these -- the difference between a
        corrupted spelling and a second character with a similar name is a
        judgement call, and merging the wrong pair costs a voice.
        """
        pairs = []
        entries = [
            (c.name_en, self._normalize_name_key(c.name_en))
            for c in self.characters.values()
        ]
        for i, (name_a, a) in enumerate(entries):
            for name_b, b in entries[i + 1:]:
                if not a or not b or a == b:
                    continue
                if a in b or b in a:
                    pairs.append((name_a, name_b))
        return pairs

    def find_character(
        self, query: str, allow_prefix: bool = True
    ) -> Optional[CharacterProfile]:
        """Resolve a name to a character: exact, then alias, then unambiguous prefix.

        Prefix matching rescues a corrupted or truncated spelling, but it is the
        loosest rule, so it only applies when nothing matched exactly and exactly
        one candidate is compatible -- two candidates means the name is ambiguous
        and guessing would put a line in the wrong voice.

        Pass allow_prefix=False when deciding whether a name is a NEW character.
        "Kim Kiryeong" is prefix compatible with "Kim Kiryeo" but may be a
        different person; folding them on registration is unrecoverable, whereas
        keeping both lets near_duplicate_characters() put it to a human.
        """
        norm_query = self._normalize_name_key(query)
        if not norm_query:
            return None

        for key, char in self.characters.items():
            if norm_query in {
                self._normalize_name_key(key),
                self._normalize_name_key(char.name_en),
                self._normalize_name_key(char.name_target),
            }:
                return char

        for char in self.characters.values():
            if any(self._normalize_name_key(a) == norm_query for a in char.aliases):
                return char

        if not allow_prefix:
            return None

        candidates = [
            char for char in self.characters.values()
            if self._prefix_compatible(norm_query, self._normalize_name_key(char.name_en))
            or self._prefix_compatible(norm_query, self._normalize_name_key(char.name_target))
        ]
        return candidates[0] if len(candidates) == 1 else None

    def update_narrator_voice(
        self,
        voice_description: Optional[str] = None,
        voice_ref_audio: Optional[str] = None,
        voice_control_prompt: Optional[str] = None,
    ):
        """Update narrator voice prompt and/or reference audio anchor."""
        if voice_description is not None:
            self.narrator_voice_description = voice_description
            # The cached English control no longer describes the new text; an
            # explicit control in this same call overrides that below.
            self.narrator_voice_control_prompt = None
        if voice_ref_audio is not None:
            self.narrator_voice_ref_audio = voice_ref_audio
        if voice_control_prompt is not None:
            self.narrator_voice_control_prompt = voice_control_prompt
        self.last_updated = datetime.utcnow()

    @staticmethod
    def _overlap(a: OccupancyInterval, b: OccupancyInterval) -> bool:
        a_end = a.end.key() if a.end else (float("inf"), 1)
        b_end = b.end.key() if b.end else (float("inf"), 1)
        return a.start.key() < b_end and b.start.key() < a_end

    @staticmethod
    def _active(interval: OccupancyInterval, position: StoryPosition) -> bool:
        return interval.start.key() <= position.key() and (
            interval.end is None or position.key() < interval.end.key()
        )

    def add_occupancy(
        self, mind_name: str, body_name: str, start: StoryPosition,
        end: Optional[StoryPosition] = None,
    ) -> OccupancyInterval:
        """Add one occupancy period, rejecting ambiguous simultaneous assignments."""
        mind = self.find_character(mind_name, allow_prefix=False)
        body = self.find_character(body_name, allow_prefix=False)
        if mind is None or body is None:
            raise ValueError("mind and body must both be registered characters")
        if mind.name_en == body.name_en:
            raise ValueError("a character cannot inhabit themselves")
        interval = OccupancyInterval(body=body.name_en, start=start, end=end)
        for other in self.characters.values():
            for existing in other.occupancy_intervals:
                if not self._overlap(interval, existing):
                    continue
                if other.name_en == mind.name_en:
                    raise ValueError("one mind cannot occupy two bodies at once")
                if existing.body == body.name_en:
                    raise ValueError("two minds cannot occupy one body at once")
        mind.occupancy_intervals.append(interval)
        mind.occupancy_intervals.sort(key=lambda item: item.start.key())
        # The old fields can describe only one open-ended chapter-boundary rule.
        if len(mind.occupancy_intervals) == 1 and end is None and start.paragraph_index == 1:
            mind.inhabiting = body.name_en
            mind.inhabiting_from_chapter = start.chapter_number
        else:
            mind.inhabiting = None
            mind.inhabiting_from_chapter = None
        self.last_updated = datetime.utcnow()
        return interval

    def remove_occupancy(self, mind_name: str, index: int) -> bool:
        mind = self.find_character(mind_name, allow_prefix=False)
        if mind is None or not 0 <= index < len(mind.occupancy_intervals):
            return False
        mind.occupancy_intervals.pop(index)
        mind.inhabiting = None
        mind.inhabiting_from_chapter = None
        if len(mind.occupancy_intervals) == 1:
            only = mind.occupancy_intervals[0]
            if only.end is None and only.start.paragraph_index == 1:
                mind.inhabiting = only.body
                mind.inhabiting_from_chapter = only.start.chapter_number
        self.last_updated = datetime.utcnow()
        return True

    def occupancy_at(self, position: StoryPosition) -> Dict[str, str]:
        """Active mind -> body assignments, with canonical character names."""
        return {
            mind.name_en: interval.body
            for mind in self.characters.values()
            for interval in mind.occupancy_intervals
            if self._active(interval, position)
        }

    def voice_for(
        self,
        character: CharacterProfile,
        speech_type: Optional[str],
        chapter_number: Optional[float] = None,
        paragraph_index: int = 1,
    ) -> Optional[CharacterProfile]:
        """Resolve a mind's voice at a paragraph. None means review is required."""
        if speech_type == "thought":
            return character
        if speech_type != "dialogue":
            return character
        if chapter_number is None:
            # A chapter without a position cannot be placed against any swap.
            affected = bool(character.occupancy_intervals) or any(
                interval.body == character.name_en
                for c in self.characters.values() for interval in c.occupancy_intervals
            )
            return None if affected else character
        position = StoryPosition(chapter_number=chapter_number, paragraph_index=paragraph_index)
        active = self.occupancy_at(position)
        if character.name_en in active:
            return self.find_character(active[character.name_en], allow_prefix=False)
        # The default body is unavailable if another mind occupies it. Do not
        # silently voice the displaced mind as the occupant's body.
        if character.name_en in active.values():
            return None
        return character

    def update_character_voice(
        self,
        name_en: str,
        voice_description: Optional[str] = None,
        voice_ref_audio: Optional[str] = None,
        voice_control_prompt: Optional[str] = None,
    ) -> bool:
        """Update character voice prompt and/or reference audio anchor."""
        char = self.find_character(name_en)
        if not char:
            return False
        if voice_description is not None:
            char.voice_description = voice_description or None
            # The cached English control no longer describes the new text.
            char.voice_control_prompt = None
        if voice_ref_audio is not None:
            char.voice_ref_audio = voice_ref_audio
        if voice_control_prompt is not None:
            char.voice_control_prompt = voice_control_prompt
        self.last_updated = datetime.utcnow()
        return True

    def set_inhabiting(
        self, name_en: str, inhabiting: Optional[str], from_chapter: Optional[float]
    ) -> bool:
        """Legacy API: replace a single open-ended chapter-boundary period."""
        char = self.find_character(name_en, allow_prefix=False)
        if not char:
            return False
        if char.occupancy_intervals and char.inhabiting is None:
            raise ValueError("edit this mind's occupancy timeline instead of replacing it with the legacy setting")
        if not inhabiting:
            char.occupancy_intervals = []
            char.inhabiting = None
            char.inhabiting_from_chapter = None
        else:
            if from_chapter is None:
                raise ValueError("a swap needs its start chapter")
            previous = char.occupancy_intervals
            char.occupancy_intervals = []
            try:
                self.add_occupancy(
                    char.name_en, inhabiting,
                    StoryPosition(chapter_number=from_chapter),
                )
            except ValueError:
                char.occupancy_intervals = previous
                raise
        self.last_updated = datetime.utcnow()
        return True

    def add_character(
        self,
        name_en: str,
        name_target: str,
        gender: Optional[str] = None,
        role: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        notes: Optional[str] = None,
        voice_description: Optional[str] = None,
        voice_ref_audio: Optional[str] = None,
    ):
        clean_name = name_en.strip()
        key = clean_name.lower()
        alias_list = [a.strip() for a in (aliases or []) if a.strip() and a.strip().lower() != key]

        # Check if this character already exists under a normalized variation (e.g. Ahn Yoonseung vs Ahn Yoon-seung)
        existing = self.find_character(clean_name, allow_prefix=False)
        if existing and existing.name_en.lower() != key:
            # Merge into the existing character
            if clean_name not in existing.aliases:
                existing.aliases.append(clean_name)
            for a in alias_list:
                if a not in existing.aliases and a.lower() != existing.name_en.lower():
                    existing.aliases.append(a)
            if gender and not existing.gender:
                existing.gender = gender
            if role and not existing.role:
                existing.role = role
            if notes and not existing.notes:
                existing.notes = notes
            if voice_description and not existing.voice_description:
                existing.voice_description = voice_description
                # A control prompt derived from an older description no longer
                # describes this voice; drop it so it is re-derived.
                existing.voice_control_prompt = None
            if voice_ref_audio and not existing.voice_ref_audio:
                existing.voice_ref_audio = voice_ref_audio
            self.last_updated = datetime.utcnow()
            return

        if key in self.characters:
            char = self.characters[key]
            char.name_target = name_target.strip()
            if gender:
                char.gender = gender
            if role:
                char.role = role
            if notes:
                char.notes = notes
            if voice_description and voice_description.strip() != (char.voice_description or "").strip():
                char.voice_description = voice_description
                # Same invalidation as update_character_voice -- the glossary edit
                # modal reaches this path, not that one.
                char.voice_control_prompt = None
            if voice_ref_audio:
                char.voice_ref_audio = voice_ref_audio
            for a in alias_list:
                if a not in char.aliases and a.lower() != key:
                    char.aliases.append(a)
        else:
            self.characters[key] = CharacterProfile(
                name_en=clean_name,
                name_target=name_target.strip(),
                gender=gender,
                role=role,
                aliases=alias_list,
                notes=notes,
                voice_description=voice_description,
                voice_ref_audio=voice_ref_audio,
            )
        self.last_updated = datetime.utcnow()

    def remove_character(self, name_en: str) -> bool:
        char = self.find_character(name_en)
        if not char:
            return False
        for other in self.characters.values():
            if any(period.body == char.name_en for period in other.occupancy_intervals):
                raise ValueError(f"{char.name_en} is used as a body in the occupancy timeline")
        del self.characters[char.name_en.strip().lower()]
        self.last_updated = datetime.utcnow()
        return True

    def rename_character(self, old_name: str, new_name: str) -> bool:
        """Preserve the character and all timeline references when renaming."""
        char = self.find_character(old_name, allow_prefix=False)
        if char is None:
            return False
        new_name = new_name.strip()
        if not new_name:
            raise ValueError("new character name is required")
        collision = self.find_character(new_name, allow_prefix=False)
        if collision is not None and collision is not char:
            raise ValueError(f"character {new_name!r} already exists")
        previous = char.name_en
        del self.characters[previous.lower()]
        char.name_en = new_name
        if previous not in char.aliases:
            char.aliases.append(previous)
        self.characters[new_name.lower()] = char
        for other in self.characters.values():
            for period in other.occupancy_intervals:
                if period.body == previous:
                    period.body = new_name
            if other.inhabiting == previous:
                other.inhabiting = new_name
        self.last_updated = datetime.utcnow()
        return True

    def remove_term(self, source: str) -> bool:
        key = source.strip().lower()
        if key in self.terms:
            del self.terms[key]
            self.last_updated = datetime.utcnow()
            return True
        return False

    def format_glossary_prompt(self) -> str:
        """Render terms, aliases, characters and their gender guards into prompt guidelines."""
        lines = []
        if self.terms:
            lines.append("### Key Term Preservation & Glossary:")
            by_cat: Dict[str, List[TermMapping]] = {}
            for t in self.terms.values():
                by_cat.setdefault(t.category, []).append(t)
            for cat, term_list in by_cat.items():
                lines.append(f"#### {cat.capitalize()} Terms:")
                for t in sorted(term_list, key=lambda x: x.source.lower()):
                    alias_str = ""
                    if t.aliases:
                        alias_str = f" (also referred to as: {', '.join(repr(a) for a in t.aliases)})"
                    note_str = f" [{t.notes}]" if t.notes else ""
                    lines.append(f'- **"{t.source}"**{alias_str} → **"{t.target}"**{note_str}')

        if self.characters:
            lines.append("\n### Character Profiles & Gender Particle Rules:")
            for c in sorted(self.characters.values(), key=lambda x: x.name_en.lower()):
                alias_str = ""
                if c.aliases:
                    alias_str = f" (also known as: {', '.join(repr(a) for a in c.aliases)})"
                
                gender_tag = ""
                if c.gender:
                    g_lower = c.gender.lower()
                    if g_lower in ("male", "m", "ชาย"):
                        gender_tag = " [เพศชาย (MALE) - บังคับหางเสียง: ครับ (ห้ามใช้ ค่ะ/นะคะ)]"
                    elif g_lower in ("female", "f", "หญิง"):
                        gender_tag = " [เพศหญิง (FEMALE) - หางเสียง: ค่ะ/คะ]"

                desc = f" ({c.role})" if c.role else ""
                note_str = f" [{c.notes}]" if c.notes else ""
                lines.append(f'- **"{c.name_en}"**{alias_str} → **"{c.name_target}"**{gender_tag}{desc}{note_str}')

        return "\n".join(lines)
