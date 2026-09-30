"""Fetching a chapter again must not throw away work done on it since.

Both fetch paths -- the extension import and the server's own scraper -- build
the chapter from scratch and overwrite the stored copy. That discarded every
speaker label on it, including the ones a human verified: the only gold labels
speaker detection is scored against.
"""
import asyncio
import shutil
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from vox_novel.models.domain import INHERITED_PARAGRAPH_FIELDS, Chapter, Paragraph
from vox_novel.speaker.detector import score_attribution
from vox_novel.web import app as web


def chapter(*texts, **labels):
    return Chapter(
        id="5", book_id="s", title="ตอนที่ 5", url="https://readtoon.com/content/s/5",
        chapter_number=5.0, source_language="th", target_language="th",
        paragraphs=[
            Paragraph(index=i, text=t, translated_text=t, **labels.get(t, {}))
            for i, t in enumerate(texts, 1)
        ],
    )


VERIFIED = {
    "speaker": "Rico", "speech_type": "dialogue", "speaker_verified": True,
    "speaker_detected": "Daisy", "speech_type_detected": "dialogue",
    "voice_override": "narrator", "emotion": "angry",
}


class TestInheritAttribution(unittest.TestCase):
    def test_an_unchanged_paragraph_keeps_everything(self):
        old = chapter("“ไปกันเถอะ”", **{"“ไปกันเถอะ”": VERIFIED})
        new = chapter("“ไปกันเถอะ”")
        self.assertEqual(new.inherit_attribution(old), 1)
        for field in INHERITED_PARAGRAPH_FIELDS:
            self.assertEqual(getattr(new.paragraphs[0], field), VERIFIED[field], field)

    def test_a_changed_paragraph_keeps_nothing(self):
        # A label is only known to be right for the words it was made on.
        old = chapter("“ไปกันเถอะ”", **{"“ไปกันเถอะ”": VERIFIED})
        new = chapter("“ไปกันเถอะ!”")
        self.assertEqual(new.inherit_attribution(old), 0)
        self.assertIsNone(new.paragraphs[0].speaker)
        self.assertFalse(new.paragraphs[0].speaker_verified)

    def test_an_inserted_line_does_not_shift_the_labels_after_it(self):
        old = chapter("a", "b", **{"a": {"speaker": "A"}, "b": {"speaker": "B"}})
        new = chapter("a", "inserted", "b")
        self.assertEqual(new.inherit_attribution(old), 2)
        self.assertEqual([p.speaker for p in new.paragraphs], ["A", None, "B"])

    def test_repeated_lines_pair_up_in_order(self):
        old = chapter("...", "...", **{})
        old.paragraphs[0].speaker, old.paragraphs[1].speaker = "First", "Second"
        new = chapter("...", "...")
        new.inherit_attribution(old)
        self.assertEqual([p.speaker for p in new.paragraphs], ["First", "Second"])

    def test_removing_an_exchange_keeps_the_remaining_speakers_context(self):
        old = chapter("Alice nodded", "Yes", "Bob answered", "Yes")
        for p, name in ((old.paragraphs[1], "Alice"), (old.paragraphs[3], "Bob")):
            p.speaker, p.speaker_verified = name, True
        new = chapter("Bob answered", "Yes")
        new.inherit_attribution(old)
        self.assertEqual(new.paragraphs[1].speaker, "Bob")
        self.assertTrue(new.paragraphs[1].speaker_verified)

    def test_inserted_identical_line_does_not_inherit_a_verified_label(self):
        old = chapter("context", "Yes", "end")
        old.paragraphs[1].speaker, old.paragraphs[1].speaker_verified = "Alice", True
        new = chapter("context", "Yes", "Yes", "end")
        new.inherit_attribution(old)
        for p in new.paragraphs[1:3]:
            self.assertIsNone(p.speaker)
            self.assertFalse(p.speaker_verified)

    def test_repeated_lines_without_matching_context_remain_unreviewed(self):
        old = chapter("Yes", "Yes")
        old.paragraphs[0].speaker, old.paragraphs[0].speaker_verified = "Alice", True
        new = chapter("Yes")
        new.inherit_attribution(old)
        self.assertIsNone(new.paragraphs[0].speaker)
        self.assertFalse(new.paragraphs[0].speaker_verified)

    def test_nothing_to_inherit_from(self):
        self.assertEqual(chapter("a").inherit_attribution(None), 0)


class RefetchTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        roots = [web.storage, web.knowledge_mgr, web.pipeline.storage, web.pipeline.knowledge]
        originals = [m.base_dir for m in roots]
        for m in roots:
            m.base_dir = self.tmp
        self.addCleanup(lambda: [setattr(m, "base_dir", o) for m, o in zip(roots, originals)])

        # A reviewed chapter: one verified line, one detected line, narration.
        stored = chapter(
            "narration line", "“ไปกันเถอะ”", "“รอด้วย”",
            **{"“ไปกันเถอะ”": VERIFIED,
               "“รอด้วย”": {"speaker": "Daisy", "speech_type": "dialogue",
                            "speaker_detected": "Daisy", "speech_type_detected": "dialogue"}},
        )
        web.storage.save_chapter(stored, True)
        self.score_before = score_attribution(stored)

    def stored(self):
        return web.storage.get_chapter("s", "5")


class TestTheExtensionImportKeepsLabels(RefetchTestCase):
    def reimport(self, *paragraphs):
        return TestClient(web.web_app).post("/api/extension/import", json={
            "url": "https://readtoon.com/content/s/5", "series_id": "s", "chapter_no": 5,
            "chapter_title": "ตอนที่ 5", "paragraphs": list(paragraphs),
            "source": "readtoon", "source_language": "th",
        })

    def test_a_verified_label_survives_a_reimport(self):
        res = self.reimport("narration line", "“ไปกันเถอะ”", "“รอด้วย”")
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(res.json()["attribution_kept"], 3)
        p = self.stored().paragraphs[1]
        self.assertEqual((p.speaker, p.speaker_verified, p.speaker_detected), ("Rico", True, "Daisy"))
        self.assertEqual(score_attribution(self.stored()), self.score_before)

    def test_an_edited_paragraph_loses_only_its_own_label(self):
        self.reimport("narration line", "“ไปกันเถอะ”", "“รอด้วยสิ”")
        paragraphs = self.stored().paragraphs
        self.assertTrue(paragraphs[1].speaker_verified)
        self.assertIsNone(paragraphs[2].speaker)


class TestTheServerFetchKeepsLabels(RefetchTestCase):
    def test_a_verified_label_survives_a_server_refetch(self):
        fresh = chapter("narration line", "“ไปกันเถอะ”", "“รอด้วย”")
        asyncio.run(web.pipeline.scrape_and_translate_chapter(
            url=fresh.url, target_lang="th", chapter_obj=fresh,
        ))
        p = self.stored().paragraphs[1]
        self.assertEqual((p.speaker, p.speaker_verified), ("Rico", True))
        self.assertEqual(score_attribution(self.stored()), self.score_before)


if __name__ == "__main__":
    unittest.main()
