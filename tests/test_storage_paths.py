import shutil
import tempfile
import unittest
from pathlib import Path

from vox_novel.models.domain import Chapter, Paragraph
from vox_novel.storage.file import (
    StorageManager,
    UnsafePathSegment,
    character_voice_key,
    validate_path_segment,
)


class TestValidatePathSegment(unittest.TestCase):
    def test_accepts_normal_ids(self):
        for ok in ["novel-slug", "36119734008764305", "a.b_c-1", "X"]:
            self.assertEqual(validate_path_segment(ok), ok)

    def test_strips_surrounding_whitespace(self):
        self.assertEqual(validate_path_segment("  slug  "), "slug")

    def test_rejects_traversal_and_separators(self):
        bad = [
            "..",
            ".",
            "../../etc/passwd",
            "..%2f..%2fetc",
            "a/b",
            "a\\b",
            "/abs",
            "",
            "   ",
            None,
            ".hidden",
            "x" * 200,
        ]
        for value in bad:
            with self.assertRaises(UnsafePathSegment, msg=f"accepted {value!r}"):
                validate_path_segment(value)


class TestCharacterVoiceKey(unittest.TestCase):
    def test_never_yields_path_separators(self):
        for name in ["../../evil", "a/b\\c", "..", "/etc/passwd", "  "]:
            key = character_voice_key(name)
            self.assertNotIn("/", key)
            self.assertNotIn("\\", key)
            self.assertNotEqual(key, "")
            # Joining onto a root must not escape it.
            joined = (Path("/root") / f"{key}_ref.wav").resolve()
            self.assertTrue(str(joined).startswith("/root/"), joined)

    def test_normalizes_spacing_and_case(self):
        self.assertEqual(character_voice_key("  Ada  Lovelace "), "ada_lovelace")
        self.assertEqual(character_voice_key("Jean-Luc"), "jean_luc")

    def test_preserves_non_ascii_names(self):
        self.assertEqual(character_voice_key("อาเธอร์"), "อาเธอร์")


class TestSaveChapterCleanup(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.storage = StorageManager(base_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _chapter(self, title: str) -> Chapter:
        return Chapter(
            id="170",
            book_id="series-a",
            title=title,
            url="https://readtoon.com/content/series-a/170",
            chapter_number=170.0,
            paragraphs=[Paragraph(id="p1", index=1, text="hello")],
        )

    def test_retitle_drops_stale_text_files(self):
        self.storage.save_chapter(self._chapter("Old Title"))
        self.storage.save_chapter(self._chapter("New Title"))

        chapters_dir = self.tmp / "series-a" / "chapters"
        names = sorted(f.name for f in chapters_dir.iterdir())
        self.assertEqual(names, ["ch_0170 - New Title.json", "ch_0170 - New Title.md"])

    def test_reimport_preserves_generated_audio(self):
        self.storage.save_chapter(self._chapter("Old Title"))
        chapters_dir = self.tmp / "series-a" / "chapters"

        # Audio can live under either naming scheme; both must survive a re-import.
        prefixed_audio = chapters_dir / "ch_0170 - Old Title.wav"
        prefixed_audio.write_bytes(b"RIFF-fake")
        chapter_audio = chapters_dir / "chapter_170.wav"
        chapter_audio.write_bytes(b"RIFF-fake")
        para_audio = chapters_dir / "para_170_1.wav"
        para_audio.write_bytes(b"RIFF-fake")

        self.storage.save_chapter(self._chapter("New Title"))

        self.assertTrue(prefixed_audio.exists(), "re-import deleted chapter audio")
        self.assertTrue(chapter_audio.exists())
        self.assertTrue(para_audio.exists())


if __name__ == "__main__":
    unittest.main()
