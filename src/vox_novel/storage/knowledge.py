import json
from pathlib import Path
from typing import Optional
from vox_novel.models.series_knowledge import SeriesKnowledge


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

        # Initialize clean, isolated series knowledge (no shared or hardcoded presets)
        knowledge = SeriesKnowledge(series_id=series_id, target_language=target_lang)
        self.save(knowledge)
        return knowledge

    def save(self, knowledge: SeriesKnowledge) -> Path:
        path = self._get_path(knowledge.series_id, knowledge.target_language)
        with open(path, "w", encoding="utf-8") as f:
            f.write(knowledge.model_dump_json(indent=2))
        return path
