import asyncio
import shutil
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from vox_novel.models.domain import Chapter, Paragraph
from vox_novel.models.series_knowledge import SeriesKnowledge, StoryPosition
from vox_novel.speaker.detector import _occupancy_block, _scoped_registry, annotate_chapter
from vox_novel.speaker.models import ChunkAnnotation, Segment
from vox_novel.web import app as web


def pos(chapter, paragraph=1):
    return StoryPosition(chapter_number=chapter, paragraph_index=paragraph)


class TestOccupancyTimeline(unittest.TestCase):
    def setUp(self):
        self.k = SeriesKnowledge(series_id="s")
        for name in ("A", "B", "C"):
            self.k.add_character(name, name)

    def voice(self, mind, chapter, paragraph, kind="dialogue"):
        result = self.k.voice_for(self.k.find_character(mind), kind, chapter, paragraph)
        return result.name_en if result else None

    def test_mid_chapter_start_end_and_recurrence(self):
        self.k.add_occupancy("A", "B", pos(10, 3), pos(10, 6))
        self.k.add_occupancy("A", "C", pos(11, 2), pos(12))
        self.assertEqual([self.voice("A", 10, i) for i in (2, 3, 5, 6)], ["A", "B", "B", "A"])
        self.assertEqual([self.voice("A", 11, i) for i in (1, 2)], ["A", "C"])
        self.assertEqual(self.voice("A", 12, 1), "A")
        self.assertEqual(self.voice("A", 10, 4, "thought"), "A")
        self.assertIsNone(self.voice("B", 10, 4))
        self.assertEqual(self.voice("B", 10, 4, "thought"), "B")

    def test_reciprocal_swap_and_unaffected_character(self):
        self.k.add_occupancy("A", "B", pos(10), pos(11))
        self.k.add_occupancy("B", "A", pos(10), pos(11))
        self.assertEqual(self.voice("A", 10, 1), "B")
        self.assertEqual(self.voice("B", 10, 1), "A")
        self.assertEqual(self.voice("C", 10, 1), "C")
        self.assertEqual(self.voice("A", 11, 1), "A")

    def test_overlaps_and_invalid_ranges_are_rejected(self):
        self.k.add_occupancy("A", "B", pos(10, 3), pos(10, 6))
        with self.assertRaises(ValueError):
            self.k.add_occupancy("A", "C", pos(10, 5), pos(11))
        with self.assertRaises(ValueError):
            self.k.add_occupancy("C", "B", pos(10, 4), pos(11))
        with self.assertRaises(ValueError):
            self.k.add_occupancy("A", "C", pos(10, 7), pos(10, 7))
        self.assertEqual(len(self.k.find_character("A").occupancy_intervals), 1)

    def test_legacy_setting_is_migrated_on_load(self):
        raw = self.k.model_dump()
        raw["characters"]["a"]["inhabiting"] = "B"
        raw["characters"]["a"]["inhabiting_from_chapter"] = 10
        raw["characters"]["a"].pop("occupancy_intervals")
        loaded = SeriesKnowledge.model_validate(raw)
        self.assertEqual(loaded.voice_for(loaded.find_character("A"), "dialogue", 10, 1).name_en, "B")
        self.assertEqual(len(loaded.find_character("A").occupancy_intervals), 1)

    def test_an_overlapping_saved_period_is_dropped_on_load_not_fatal(self):
        # Load runs on every page; refusing the file would 500 the whole series,
        # including the glossary where the conflict would be fixed.
        self.k.add_occupancy("A", "B", pos(10, 2), pos(10, 5))
        raw = self.k.model_dump()
        raw["characters"]["c"]["occupancy_intervals"] = [{
            "body": "B", "start": {"chapter_number": 10, "paragraph_index": 3},
            "end": {"chapter_number": 10, "paragraph_index": 6},
        }]
        with self.assertLogs("vox_novel.models.series_knowledge", "WARNING"):
            loaded = SeriesKnowledge.model_validate(raw)
        self.assertEqual(len(loaded.find_character("A").occupancy_intervals), 1)
        self.assertEqual(loaded.find_character("C").occupancy_intervals, [])

    def test_a_legacy_swap_onto_a_deleted_body_loads(self):
        # Before the edit paths guarded it, deleting or renaming the body left
        # the mind's legacy field pointing at nothing.
        raw = self.k.model_dump()
        raw["characters"]["a"]["inhabiting"] = "gone"
        raw["characters"]["a"]["inhabiting_from_chapter"] = 100.0
        with self.assertLogs("vox_novel.models.series_knowledge", "WARNING"):
            loaded = SeriesKnowledge.model_validate(raw)
        a = loaded.find_character("A")
        self.assertEqual(a.occupancy_intervals, [])
        # Cleared too, or the next load would migrate the dead period back in.
        self.assertIsNone(a.inhabiting)
        self.assertIsNone(a.inhabiting_from_chapter)

    def test_two_legacy_open_claims_on_one_body_keep_the_first(self):
        raw = self.k.model_dump()
        for key in ("a", "c"):
            raw["characters"][key]["inhabiting"] = "B"
            raw["characters"][key]["inhabiting_from_chapter"] = 100.0
        with self.assertLogs("vox_novel.models.series_knowledge", "WARNING"):
            loaded = SeriesKnowledge.model_validate(raw)
        self.assertEqual(loaded.find_character("A").inhabiting, "B")
        self.assertIsNone(loaded.find_character("C").inhabiting)

    def test_a_legacy_swap_naming_the_body_by_alias_is_kept(self):
        self.k.find_character("B").aliases.append("bee")
        raw = self.k.model_dump()
        raw["characters"]["a"]["inhabiting"] = "bee"
        raw["characters"]["a"]["inhabiting_from_chapter"] = 100.0
        loaded = SeriesKnowledge.model_validate(raw)
        self.assertEqual(loaded.find_character("A").occupancy_intervals[0].body, "B")

    def test_the_edit_path_still_refuses_an_overlap(self):
        self.k.add_occupancy("A", "B", pos(10, 2), pos(10, 5))
        with self.assertRaises(ValueError):
            self.k.add_occupancy("C", "B", pos(10, 3), pos(10, 6))

    def test_prompt_context_marks_the_exact_transition(self):
        self.k.add_occupancy("A", "B", pos(10, 3), pos(10, 5))
        block = _occupancy_block(self.k, 10, [1, 2, 3, 4, 5])
        self.assertIn("[1]-[2]: no active body change", block)
        self.assertIn("[3]-[4]: A -> B", block)
        self.assertIn("[5]: no active body change", block)

    def test_annotation_receives_positioned_occupancy_context(self):
        self.k.add_occupancy("A", "B", pos(10, 2))
        chapter = Chapter(id="c", book_id="s", title="t", url="u", chapter_number=10,
                          paragraphs=[Paragraph(index=i, text=f"line {i}") for i in (1, 2)])
        class Translator:
            async def structured(self, system, user, model):
                assert "[1]: no active body change" in user
                assert "[2]: A -> B" in user
                return ChunkAnnotation(segments=[
                    Segment(paragraph=i, type="dialogue", speaker="A") for i in (1, 2)
                ])
        asyncio.run(annotate_chapter(chapter, self.k, Translator(), use_translated=False))
        self.assertEqual([p.speaker for p in chapter.paragraphs], ["A", "A"])

    def test_active_mind_and_body_survive_a_large_registry_limit(self):
        for i in range(50):
            name = f"Extra {i}"
            self.k.add_character(name, name)
            self.k.find_character(name).line_count = 100
        self.k.add_occupancy("A", "B", pos(10))
        names = {entry["name"] for entry in _scoped_registry(self.k, {"A", "B"})}
        self.assertEqual(len(names), 40)
        self.assertTrue({"A", "B"}.issubset(names))

    def test_registry_exposes_names_used_by_the_occupancy_timeline(self):
        self.k.find_character("A").name_target = "Translated A"
        self.k.add_occupancy("A", "B", pos(10))
        entry = next(item for item in _scoped_registry(self.k) if item["canonical_name"] == "A")
        self.assertEqual(entry["name"], "Translated A")
        self.assertIn("A -> B", _occupancy_block(self.k, 10, [1]))

    def test_renaming_a_body_preserves_the_timeline_and_deletion_is_guarded(self):
        self.k.add_occupancy("A", "B", pos(10, 3))
        self.assertTrue(self.k.rename_character("B", "New B"))
        self.assertEqual(self.voice("A", 10, 3), "New B")
        with self.assertRaises(ValueError):
            self.k.remove_character("New B")
        self.assertIsNotNone(self.k.find_character("New B"))


class TestOccupancyWeb(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        roots = [web.storage, web.knowledge_mgr, web.pipeline.storage, web.pipeline.knowledge]
        originals = [m.base_dir for m in roots]
        for m in roots:
            m.base_dir = self.tmp
        self.addCleanup(lambda: [setattr(m, "base_dir", o) for m, o in zip(roots, originals)])
        self.client = TestClient(web.web_app)
        k = web.knowledge_mgr.load_or_init("s")
        for name in ("A", "B"):
            k.add_character(name, name)
            k.find_character(name).voice_description = f"Voice for {name}"
        web.knowledge_mgr.save(k)

    def test_add_remove_and_review_voice(self):
        payload = {
            "action": "add", "series_id": "s", "mind": "A", "body": "B",
            "start_chapter": 10, "start_paragraph": 2,
            "end_chapter": 10, "end_paragraph": 4,
        }
        self.assertEqual(self.client.post("/api/voices/occupancy", json=payload).status_code, 200)
        k = web.knowledge_mgr.load_or_init("s")
        self.assertEqual(len(k.find_character("A").occupancy_intervals), 1)
        chapter = Chapter(id="ch", book_id="s", title="t", url="u", chapter_number=10,
                          paragraphs=[Paragraph(index=i, text=f"line {i}", speaker="A", speech_type="dialogue") for i in range(1, 5)])
        web.storage.save_chapter(chapter)
        page = self.client.get("/series/s/speakers/ch")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Voice:", page.text)
        self.assertIn("voice: auto", page.text)
        self.assertIn('<span class="resolved-voice">B</span>', page.text)
        glossary = self.client.get("/series/s/glossary")
        self.assertEqual(glossary.status_code, 200)
        self.assertIn("Body occupancy timeline", glossary.text)
        self.assertIn("Start ch", glossary.text)
        response = self.client.post("/api/speakers/update", json={
            "series_id": "s", "chapter_id": "ch", "paragraph": 2,
            "speaker": "A", "speech_type": "dialogue", "voice_override": "narrator",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["voice"], "narrator (override)")
        self.assertEqual(self.client.post("/api/voices/occupancy", json={
            "action": "remove", "series_id": "s", "mind": "A", "index": 0,
        }).status_code, 200)
        self.assertEqual(web.knowledge_mgr.load_or_init("s").find_character("A").occupancy_intervals, [])

    def test_invalid_or_overlapping_period_does_not_change_saved_timeline(self):
        base = {
            "action": "add", "series_id": "s", "mind": "A", "body": "B",
            "start_chapter": 10, "start_paragraph": 2,
            "end_chapter": 10, "end_paragraph": 4,
        }
        self.assertEqual(self.client.post("/api/voices/occupancy", json=base).status_code, 200)
        self.assertEqual(self.client.post("/api/voices/occupancy", json={
            **base, "start_paragraph": 3, "end_paragraph": 5,
        }).status_code, 400)
        self.assertEqual(self.client.post("/api/voices/occupancy", json={
            **base, "start_paragraph": 5, "end_paragraph": 4,
        }).status_code, 400)
        self.assertEqual(len(web.knowledge_mgr.load_or_init("s").find_character("A").occupancy_intervals), 1)

    def _override_chapter(self):
        chapter = Chapter(id="ov", book_id="s", title="t", url="u", chapter_number=10,
                          paragraphs=[Paragraph(index=1, text="line", speaker="A",
                                                speech_type="dialogue", voice_override="B")])
        web.storage.save_chapter(chapter)
        k = web.knowledge_mgr.load_or_init("s")
        self.assertTrue(k.rename_character("B", "New B"))
        web.knowledge_mgr.save(k)

    def test_an_override_on_a_renamed_character_still_shows_as_selected(self):
        # The stored name is now an alias. Unless the page maps it to the current
        # name, no option is selected and the row reads "voice: auto".
        self._override_chapter()
        page = self.client.get("/series/s/speakers/ov")
        self.assertEqual(page.status_code, 200)
        self.assertIn('<option value="New B" selected>', page.text)

    def test_editing_the_speaker_does_not_touch_the_override(self):
        # The page now omits voice_override unless that dropdown changed; the
        # endpoint must leave it alone when the key is absent.
        self._override_chapter()
        response = self.client.post("/api/speakers/update", json={
            "series_id": "s", "chapter_id": "ov", "paragraph": 1,
            "speaker": "A", "speech_type": "thought",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(web.storage.get_chapter("s", "ov").paragraphs[0].voice_override, "B")

    def test_the_page_sends_the_override_only_when_it_changed(self):
        self._override_chapter()
        page = self.client.get("/series/s/speakers/ov").text
        self.assertIn("changed === voiceOverride ? { voice_override", page)
        self.assertNotIn("voice_override: voiceOverride.value,\n", page)


if __name__ == "__main__":
    unittest.main()
