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


class TestCharacterNameResolution(unittest.TestCase):
    """Attribution makes this reachable: a wrong match puts a line in another voice."""

    def registry(self, *names):
        from vox_novel.models.series_knowledge import SeriesKnowledge

        k = SeriesKnowledge(series_id="s", target_language="th")
        for n in names:
            k.add_character(n, n)
        return k

    def test_names_differing_only_by_a_space_stay_distinct(self):
        # Deleting separators made these one character, silently merging two
        # people onto one voice -- plausible in Chinese and Thai web novels.
        for a, b in [("An Na", "Anna"), ("Li Wei", "Liwei"), ("Bai Lu", "Bailu")]:
            k = self.registry(a, b)
            self.assertEqual(len(k.characters), 2, f"{a} and {b} were merged")
            self.assertEqual(k.find_character(a).name_en, a)
            self.assertEqual(k.find_character(b).name_en, b)

    def test_case_and_separator_variants_still_resolve(self):
        k = self.registry("Ye Chen")
        for q in ("ye chen", "YE_CHEN", "Ye-Chen", "  Ye Chen  "):
            self.assertEqual(k.find_character(q).name_en, "Ye Chen", q)

    def test_a_corrupted_thai_spelling_resolves(self):
        # Models emit stray CJK tokens inside Thai names.
        k = self.registry("เทเนเบรย์")
        self.assertEqual(k.find_character("เทเนเบร义").name_en, "เทเนเบรย์")

    def test_an_ambiguous_prefix_refuses_to_guess(self):
        k = self.registry("Kim Kiryeo", "Kim Kiryeong")
        self.assertIsNone(
            k.find_character("Kim Kiry"), "guessing here puts a line in the wrong voice"
        )

    def test_registration_does_not_fold_a_similar_name(self):
        k = self.registry("Kim Kiryeo", "Kim Kiryeong")
        self.assertEqual(len(k.characters), 2)
        self.assertEqual(
            k.near_duplicate_characters(), [("Kim Kiryeo", "Kim Kiryeong")],
            "the pair should be surfaced for a human instead",
        )

    def test_an_alias_resolves_before_a_prefix(self):
        from vox_novel.models.series_knowledge import SeriesKnowledge

        k = SeriesKnowledge(series_id="s", target_language="th")
        k.add_character("Arthur", "อาเธอร์", aliases=["Art"])
        k.add_character("Arthurian", "อาเธอเรียน")
        self.assertEqual(k.find_character("Art").name_en, "Arthur")


if __name__ == "__main__":
    unittest.main()
