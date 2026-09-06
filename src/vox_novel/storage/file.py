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
            referer = "https://readtoon.com/" if any(k in cover_url.lower() for k in ["nobuild.pro", "readtoon"]) else "https://www.webnovel.com/"
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
                "Referer": referer,
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
        for ext in [".jpg", ".jpeg", ".webp", ".png"]:
            p = self.base_dir / series_id / f"cover{ext}"
            if p.exists() and p.stat().st_size > 0:
                return p
        return None


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

    def get_voices_dir(self, series_id: str) -> Path:
        voices_dir = self.base_dir / series_id / "voices"
        voices_dir.mkdir(parents=True, exist_ok=True)
        return voices_dir

    def get_narrator_voice_file(self, series_id: str) -> Optional[Path]:
        voices_dir = self.get_voices_dir(series_id)
        for ext in [".wav", ".mp3", ".m4a", ".flac"]:
            p = voices_dir / f"narrator_ref{ext}"
            if p.exists() and p.stat().st_size > 0:
                return p
        return None

    def get_character_voice_file(self, series_id: str, character_name: str) -> Optional[Path]:
        voices_dir = self.get_voices_dir(series_id)
        safe_key = re.sub(r"[\s\-_]+", "_", character_name.strip().lower())
        for ext in [".wav", ".mp3", ".m4a", ".flac"]:
            p = voices_dir / f"{safe_key}_ref{ext}"
            if p.exists() and p.stat().st_size > 0:
                return p
        for f in voices_dir.glob(f"*{safe_key}*"):
            if f.is_file() and f.stat().st_size > 0 and f.suffix.lower() in [".wav", ".mp3", ".m4a", ".flac"]:
                return f
        return None

    def delete_voice_file(self, series_id: str, voice_type: str, character_name: Optional[str] = None) -> bool:
        voices_dir = self.get_voices_dir(series_id)
        if voice_type == "narrator":
            target = self.get_narrator_voice_file(series_id)
            if target and target.exists():
                target.unlink()
                return True
        elif character_name:
            target = self.get_character_voice_file(series_id, character_name)
            if target and target.exists():
                target.unlink()
                return True
        return False

    def get_chapter(self, series_id: str, chapter_id: str) -> Optional[Chapter]:
        chapters_dir = self.base_dir / series_id / "chapters"
        if not chapters_dir.exists():
            return None
        for jf in chapters_dir.glob("*.json"):
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("id") == chapter_id:
                        return Chapter.model_validate(data)
            except Exception:
                pass
        return None

    def get_chapter_audio_file(self, series_id: str, chapter_id: str) -> Optional[Path]:
        chapters_dir = self.base_dir / series_id / "chapters"
        if not chapters_dir.exists():
            return None
        # Check standard chapter_<id>.wav
        p1 = chapters_dir / f"chapter_{chapter_id}.wav"
        if p1.exists():
            return p1
        p1_mp3 = chapters_dir / f"chapter_{chapter_id}.mp3"
        if p1_mp3.exists():
            return p1_mp3
        # Check matching prefix.wav
        for wf in chapters_dir.glob("*.wav"):
            if chapter_id in wf.stem:
                return wf
        for mf in chapters_dir.glob("*.mp3"):
            if chapter_id in mf.stem:
                return mf
        return None

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
                    cid = data.get("id")
                    if data.get("target_language") == target_lang or data.get("translated_title"):
                        has_audio = (
                            (chapters_dir / f"chapter_{cid}.wav").exists()
                            or (chapters_dir / f"{json_file.stem}.wav").exists()
                            or bool(data.get("audio_path") and Path(data["audio_path"]).exists())
                        )
                        translated[cid] = {
                            "title": data.get("translated_title") or data.get("title"),
                            "original_title": data.get("title"),
                            "target_language": data.get("target_language"),
                            "chapter_number": data.get("chapter_number"),
                            "json_path": str(json_file),
                            "md_path": str(json_file.with_suffix(".md")),
                            "has_audio": has_audio,
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
