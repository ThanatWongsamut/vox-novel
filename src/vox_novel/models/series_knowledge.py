from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class TermMapping(BaseModel):
    source: str
    target: str
    category: str = "general"  # gaming, character, location, item, skill, monster, honorific
    aliases: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


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
        """Strip hyphens, extra whitespace, and lowercase for robust identity matching."""
        import re
        return re.sub(r"[\s\-_]+", "", name.strip().lower())

    def find_character(self, query: str) -> Optional[CharacterProfile]:
        """Find an existing character by exact name, normalized key, target name, or alias."""
        norm_query = self._normalize_name_key(query)
        # Check direct keys and target name
        for key, char in self.characters.items():
            if (
                self._normalize_name_key(key) == norm_query
                or self._normalize_name_key(char.name_en) == norm_query
                or self._normalize_name_key(char.name_target) == norm_query
            ):
                return char
            for alias in char.aliases:
                if self._normalize_name_key(alias) == norm_query:
                    return char
        return None

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
    ) -> bool:
        """Update character voice prompt and/or reference audio anchor."""
        char = self.find_character(name_en)
        if not char:
            return False
        if voice_description is not None:
            char.voice_description = voice_description
            # The cached English control no longer describes the new text.
            char.voice_control_prompt = None
        if voice_ref_audio is not None:
            char.voice_ref_audio = voice_ref_audio
        if voice_control_prompt is not None:
            char.voice_control_prompt = voice_control_prompt
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
        existing = self.find_character(clean_name)
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
            if voice_description and voice_description != char.voice_description:
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
