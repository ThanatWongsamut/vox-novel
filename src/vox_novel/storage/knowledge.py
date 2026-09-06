import json
from pathlib import Path
from typing import Optional
from vox_novel.models.series_knowledge import SeriesKnowledge


DEFAULT_THAI_PRESETS = {
    "terms": [
        ("gate", "เกท", "gaming", "dungeon gate"),
        ("dungeon", "ดันเจี้ยน", "gaming", "dungeon location"),
        ("dungeon break", "ดันเจี้ยนเบรค", "gaming", "dungeon break event"),
        ("skill", "สกิล", "gaming", "special skill"),
        ("item", "ไอเทม", "gaming", "game item"),
        ("boss", "บอส", "gaming", "monster boss"),
        ("Magic Tower", "หอคอยเวทย์", "gaming", "magic structure"),
        ("hunter", "ฮันเตอร์", "gaming", "hunter class/job"),
        ("awakener", "ผู้อเวค", "gaming", "awakened person"),
        ("awake", "การอเวค", "gaming", "act of awakening"),
        ("F-Class", "คลาส F", "gaming", "hunter ranking"),
        ("S-Class", "คลาส S", "gaming", "hunter ranking"),
        ("Pyromancer", "นักเวทย์ไฟ", "gaming", "magic class"),
        ("potion", "โพชั่น", "gaming", "healing or mana drink"),
        ("earthlings", "ชาวโลก", "general", "people of Earth"),
        ("alien", "เอเลี่ยน", "general", "extraterrestrial"),
    ],
    "characters": [
        ("Kim Kiryeo", "คิม กีรยอง", "Protagonist"),
        ("Seonwoo Yeon", "ซอนอูยอน", "High-level hunter"),
        ("Ahn Yoonseung", "อันยุนซึง", "Hunter"),
    ],
}


class KnowledgeManager:
    """Manages persistence of series lore, glossaries, and self-learning updates."""

    def __init__(self, base_dir: Path = Path("output")):
        self.base_dir = Path(base_dir)

    def _get_path(self, series_id: str, target_lang: str) -> Path:
        series_dir = self.base_dir / series_id
        series_dir.mkdir(parents=True, exist_ok=True)
        return series_dir / f"knowledge_{target_lang}.json"

    def load_or_init(self, series_id: str, target_lang: str = "th") -> SeriesKnowledge:
        path = self._get_path(series_id, target_lang)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return SeriesKnowledge.model_validate(data)

        # Initialize with series presets if available
        knowledge = SeriesKnowledge(series_id=series_id, target_language=target_lang)
        if target_lang == "th":
            for src, tgt, cat, note in DEFAULT_THAI_PRESETS["terms"]:
                knowledge.add_term(src, tgt, cat, note)
            for name_en, name_tgt, role in DEFAULT_THAI_PRESETS["characters"]:
                knowledge.add_character(name_en, name_tgt, role=role)

        self.save(knowledge)
        return knowledge

    def save(self, knowledge: SeriesKnowledge) -> Path:
        path = self._get_path(knowledge.series_id, knowledge.target_language)
        with open(path, "w", encoding="utf-8") as f:
            f.write(knowledge.model_dump_json(indent=2))
        return path
