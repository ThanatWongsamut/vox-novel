import re
from datetime import datetime
from typing import Dict, List, NamedTuple, Optional
from pydantic import BaseModel, Field


class TermMapping(BaseModel):
    source: str
    target: str
    category: str = "general"  # gaming, character, location, item, skill, monster, honorific
    aliases: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class VoiceChoice(NamedTuple):
    """Which of a character's voices a line is heard in, and where it is cached.

    `slug` distinguishes the two voices everywhere a name becomes a filename or a
    cache key, so a body voice never serves the character's own prompt or clones
    from their anchor.
    """

    description: Optional[str]
    ref_audio: Optional[str]
    control_prompt: Optional[str]
    slug: str
    is_body: bool


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
    # A character whose spoken lines are heard in a different voice than their
    # inner ones -- a body swap, a possession, a disguise. Attribution stays
    # about identity: every line is still this character's. Only dialogue is
    # voiced from here; thought keeps the character's own voice, which is the
    # point of the distinction.
    #
    # This is a voice, not a second character, because the other body's name is
    # usually already an alias of this one. Registering it separately would make
    # `find_character` ambiguous on a name the attribution model actually emits.
    body_voice_description: Optional[str] = None
    body_voice_ref_audio: Optional[str] = None
    body_voice_control_prompt: Optional[str] = None
    # The chapter the swap begins at, by chapter_number. Earlier chapters keep
    # the character's own voice, so re-synthesizing one after the plot moves does
    # not silently re-voice it. Unset means the body voice never applies.
    body_voice_from_chapter: Optional[float] = None

    def voice_for(
        self, speech_type: Optional[str], chapter_number: Optional[float] = None
    ) -> "VoiceChoice":
        """Pick which of this character's voices a line is heard in.

        Only spoken lines take the body voice. Thought keeps the character's own,
        which is the whole point of the distinction: the listener hears the body
        others hear, and the mind the character actually is.

        A chapter with no number cannot be placed relative to the swap, so it
        falls back to the character's own voice rather than guessing.
        """
        swapped = (
            self.body_voice_description is not None
            and self.body_voice_from_chapter is not None
            and chapter_number is not None
            and chapter_number >= self.body_voice_from_chapter
        )
        if swapped and speech_type == "dialogue":
            return VoiceChoice(
                self.body_voice_description,
                self.body_voice_ref_audio,
                self.body_voice_control_prompt,
                f"{self.name_en}__body",
                True,
            )
        return VoiceChoice(
            self.voice_description,
            self.voice_ref_audio,
            self.voice_control_prompt,
            self.name_en,
            False,
        )

    def remember_control_prompt(self, choice: "VoiceChoice", control: str) -> None:
        """Cache a derived control prompt onto whichever voice produced it."""
        if choice.is_body:
            self.body_voice_control_prompt = control
        else:
            self.voice_control_prompt = control


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

    def update_character_voice(
        self,
        name_en: str,
        voice_description: Optional[str] = None,
        voice_ref_audio: Optional[str] = None,
        voice_control_prompt: Optional[str] = None,
        slot: str = "own",
        from_chapter: Optional[float] = None,
    ) -> bool:
        """Update character voice prompt and/or reference audio anchor.

        `slot` picks which of the character's two voices is written -- "own", or
        "body" for the voice their spoken lines take while they are in someone
        else's body. The body voice also needs `from_chapter`, without which it
        never applies.
        """
        if slot not in ("own", "body"):
            raise ValueError(f"slot must be 'own' or 'body', not {slot!r}")
        char = self.find_character(name_en)
        if not char:
            return False
        prefix = "body_voice" if slot == "body" else "voice"
        if voice_description is not None:
            setattr(char, f"{prefix}_description", voice_description or None)
            # The cached English control no longer describes the new text.
            setattr(char, f"{prefix}_control_prompt", None)
        if voice_ref_audio is not None:
            setattr(char, f"{prefix}_ref_audio", voice_ref_audio)
        if voice_control_prompt is not None:
            setattr(char, f"{prefix}_control_prompt", voice_control_prompt)
        if slot == "body":
            char.body_voice_from_chapter = from_chapter
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
        key = name_en.strip().lower()
        if key in self.characters:
            del self.characters[key]
            self.last_updated = datetime.utcnow()
            return True
        # Also try finding by normalized key
        char = self.find_character(name_en)
        if char:
            k = char.name_en.strip().lower()
            if k in self.characters:
                del self.characters[k]
                self.last_updated = datetime.utcnow()
                return True
        return False

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
