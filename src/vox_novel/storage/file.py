import json
import re
from pathlib import Path
from typing import Dict, List, Optional
import httpx
from vox_novel.models.domain import Chapter, Novel


def sanitize_filename(name: str) -> str:
    clean = re.sub(r'[\\/*?:"<>|]', "", name)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:80]


def get_chapter_file_prefix(chapter: Chapter) -> str:
    if chapter.chapter_number is not None:
        if chapter.chapter_number.is_integer():
            num_str = f"ch_{int(chapter.chapter_number):04d}"
        else:
            num_str = f"ch_{chapter.chapter_number:06.1f}"
    else:
        num_str = f"ch_{chapter.id[:8]}"

    raw_title = chapter.title or f"Chapter {chapter.id}"
    safe_title = sanitize_filename(raw_title)
    return f"{num_str} - {safe_title}"


class StorageManager:
    """Manages saving and loading Novel/Chapter data, local cover caching, and translation status."""

    def __init__(self, base_dir: Path = Path("output")):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def download_and_cache_cover(self, series_id: str, cover_url: Optional[str]) -> Optional[str]:
        """Download remote cover and cache locally to prevent hotlinking/CORS issues."""
        if not cover_url:
            return None
        novel_dir = self.base_dir / series_id
        novel_dir.mkdir(parents=True, exist_ok=True)
        local_cover_path = novel_dir / "cover.jpg"

        if local_cover_path.exists() and local_cover_path.stat().st_size > 0:
            return f"/api/cover/{series_id}"

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Referer": "https://www.webnovel.com/",
            }
            resp = httpx.get(cover_url, headers=headers, timeout=10.0, follow_redirects=True)
            if resp.status_code == 200 and len(resp.content) > 500:
                with open(local_cover_path, "wb") as f:
                    f.write(resp.content)
                return f"/api/cover/{series_id}"
        except Exception:
            pass
        return cover_url

    def get_local_cover_file(self, series_id: str) -> Optional[Path]:
        p = self.base_dir / series_id / "cover.jpg"
        return p if p.exists() else None

    def list_saved_series(self) -> List[Novel]:
        novels = []
        if not self.base_dir.exists():
            return novels
        for child in self.base_dir.iterdir():
            if child.is_dir():
                info_path = child / "novel_info.json"
                if info_path.exists():
                    try:
                        with open(info_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            novel = Novel.model_validate(data)
                            # If local cover cached, serve it
                            if (child / "cover.jpg").exists():
                                novel.cover_url = f"/api/cover/{novel.id}"
                            novels.append(novel)
                    except Exception:
                        pass
        return novels

    def get_novel(self, series_id: str) -> Optional[Novel]:
        info_path = self.base_dir / series_id / "novel_info.json"
        if info_path.exists():
            with open(info_path, "r", encoding="utf-8") as f:
                novel = Novel.model_validate(json.load(f))
                if (self.base_dir / series_id / "cover.jpg").exists():
                    novel.cover_url = f"/api/cover/{novel.id}"
                return novel
        return None

    def save_novel_metadata(self, novel: Novel) -> Path:
        novel_dir = self.base_dir / novel.id
        novel_dir.mkdir(parents=True, exist_ok=True)

        # Cache cover locally
        if novel.cover_url and not novel.cover_url.startswith("/api/cover/"):
            cached_url = self.download_and_cache_cover(novel.id, novel.cover_url)
            if cached_url:
                novel.cover_url = cached_url

        meta_file = novel_dir / "novel_info.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            f.write(novel.model_dump_json(indent=2))
        return meta_file

    def get_translated_chapter_ids(self, series_id: str, target_lang: str = "th") -> Dict[str, dict]:
        novel_dir = self.base_dir / series_id
        chapters_dir = novel_dir / "chapters"
        translated = {}
        if not chapters_dir.exists():
            return translated

        for json_file in chapters_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("target_language") == target_lang or data.get("translated_title"):
                        translated[data["id"]] = {
                            "title": data.get("translated_title") or data.get("title"),
                            "original_title": data.get("title"),
                            "target_language": data.get("target_language"),
                            "chapter_number": data.get("chapter_number"),
                            "json_path": str(json_file),
                            "md_path": str(json_file.with_suffix(".md")),
                        }
            except Exception:
                pass
        return translated

    def save_chapter(self, chapter: Chapter, as_markdown: bool = True) -> Path:
        novel_dir = self.base_dir / chapter.book_id
        chapters_dir = novel_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

        prefix = get_chapter_file_prefix(chapter)

        legacy_json = chapters_dir / f"chapter_{chapter.id}.json"
        legacy_md = chapters_dir / f"chapter_{chapter.id}.md"
        if legacy_json.exists():
            legacy_json.unlink()
        if legacy_md.exists():
            legacy_md.unlink()

        json_file = chapters_dir / f"{prefix}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            f.write(chapter.model_dump_json(indent=2))

        if as_markdown:
            md_file = chapters_dir / f"{prefix}.md"
            with open(md_file, "w", encoding="utf-8") as f:
                display_title = chapter.translated_title or chapter.title
                f.write(f"# {display_title}\n\n")
                if chapter.translated_title:
                    f.write(f"*Original: {chapter.title}*\n\n---\n\n")

                for p in chapter.paragraphs:
                    text_to_show = p.translated_text if p.translated_text else p.text
                    f.write(f"{text_to_show}\n\n")

        return json_file
