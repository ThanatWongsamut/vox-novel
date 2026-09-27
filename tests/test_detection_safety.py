"""Regressions from the full-branch review of speaker detection.

Each of these damaged data silently: a deletion that took out the wrong
character, a registration that merged two people onto one voice, a run that
overwrote the labels a human was entering, a re-run that reported the previous
run's guesses as its own.
"""
import asyncio
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vox_novel.models.domain import Chapter, Paragraph
from vox_novel.models.series_knowledge import SeriesKnowledge
from vox_novel.speaker.detector import annotate_chapter, score_attribution
from vox_novel.speaker.models import ChunkAnnotation, Segment


def chapter(n=3, cid="1"):
    return Chapter(
        id=cid, book_id="b", title="t", url="u", chapter_number=1.0, target_language="th",
        paragraphs=[
            Paragraph(id=f"p{i}", index=i, text=f"line {i}", translated_text=f"บรรทัด {i}")
            for i in range(1, n + 1)
        ],
    )


def knowledge(*names):
    k = SeriesKnowledge(series_id="b", target_language="th")
    for n in names:
        k.add_character(n, n)
    return k


def seg(i, speaker="Annabelle", type_="dialogue"):
    return Segment(paragraph=i, type=type_, speaker=speaker, confidence=1.0)


class Fake:
    def __init__(self, *responses, before_each=None):
        self.responses = list(responses)
        self.before_each = before_each

    async def structured(self, system_prompt, user_prompt, output_model, temperature=0.0):
        if self.before_each:
            self.before_each()
        nxt = self.responses.pop(0) if self.responses else RuntimeError("chunk failed")
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class TestPrefixMatchingCannotMergePeople(unittest.TestCase):
    def test_deleting_a_missing_name_does_not_delete_a_longer_one(self):
        k = knowledge("Annabelle")
        self.assertFalse(k.remove_character("Anna"))
        self.assertIsNotNone(k.find_character("Annabelle"))

    def test_a_new_speaker_is_not_folded_into_a_longer_name(self):
        chap, k = chapter(1), knowledge("Annabelle")
        asyncio.run(annotate_chapter(chap, k, Fake(ChunkAnnotation(segments=[seg(1, "Anna")]))))
        self.assertEqual(chap.paragraphs[0].speaker, "Anna")
        self.assertIsNotNone(k.find_character("Anna", allow_prefix=False))
        self.assertEqual(len(k.characters), 2)

    def test_a_corrupted_thai_name_is_still_rescued_by_prefix(self):
        # The case prefix matching exists for: the CJK token replaced the tail.
        chap, k = chapter(1), knowledge("เทเนเบรย์")
        asyncio.run(annotate_chapter(chap, k, Fake(ChunkAnnotation(segments=[seg(1, "เทเนเบร义")]))))
        self.assertEqual(chap.paragraphs[0].speaker, "เทเนเบรย์")
        self.assertEqual(len(k.characters), 1)

    def test_translation_prescan_does_not_learn_a_prefix_as_an_alias(self):
        # That path saves the name as a permanent alias, so a prefix match there
        # would merge two people for good.
        from vox_novel.translators.openrouter import OpenRouterTranslator

        t = OpenRouterTranslator(api_key="k", model="m")
        k = knowledge("Annabelle")
        reply = json.dumps({"new_terms": [], "new_characters": [{"name_en": "Anna"}]})
        with mock.patch.object(t, "_call_chat_completion", mock.AsyncMock(return_value=reply)):
            _, new_chars = asyncio.run(t.detect_new_entities_for_review("text", k))
        self.assertEqual([c["name_en"] for c in new_chars], ["Anna"])
        self.assertNotIn("Anna", k.find_character("Annabelle").aliases)


class TestNamesInEveryScriptAreKept(unittest.TestCase):
    def test_accents_are_not_stripped(self):
        k = knowledge("Zoë")
        self.assertIsNone(k.find_character("Zo", allow_prefix=False))
        self.assertEqual(k.find_character("zoë").name_en, "Zoë")

    def test_a_hangul_name_can_be_found(self):
        k = knowledge("김기령")
        self.assertNotEqual(SeriesKnowledge._normalize_name_key("김기령"), "")
        self.assertEqual(k.find_character("김기령").name_en, "김기령")

    def test_cjk_noise_inside_a_thai_name_is_still_ignored(self):
        self.assertEqual(
            SeriesKnowledge._normalize_name_key("ริโก้义"),
            SeriesKnowledge._normalize_name_key("ริโก้"),
        )


class TestALocalServerNeedsNoKey(unittest.TestCase):
    """Importing the translator loads .env, so the key is pinned empty around
    each call rather than removed up front -- otherwise a developer's real key
    leaks into the assertion and the test proves nothing."""

    def headers_without_a_key(self, **kw):
        from vox_novel.translators.openrouter import OpenRouterTranslator

        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}):
            t = OpenRouterTranslator(model="m", **kw)
            t.api_key = None
            return t._get_headers()

    def test_no_key_against_a_local_server_sends_no_authorization(self):
        headers = self.headers_without_a_key(base_url="http://localhost:11434/v1")
        self.assertNotIn("Authorization", headers)

    def test_no_key_against_openrouter_still_refuses(self):
        with self.assertRaises(ValueError):
            self.headers_without_a_key()


class TestARerunReportsOnlyItsOwnGuesses(unittest.TestCase):
    def test_a_failed_chunk_clears_the_previous_guess(self):
        chap, k = chapter(2), knowledge("Annabelle", "Rico")
        asyncio.run(annotate_chapter(chap, k, Fake(ChunkAnnotation(segments=[seg(1), seg(2)]))))
        chap.paragraphs[1].speaker_verified = True

        # Second run: the chunk fails and so does the repair pass.
        asyncio.run(annotate_chapter(chap, k, Fake(RuntimeError("down"), RuntimeError("down"))))

        unverified, verified = chap.paragraphs
        self.assertIsNone(unverified.speaker, "the previous run's attribution survived")
        self.assertEqual(unverified.speech_type, "narration")
        self.assertIsNone(unverified.speaker_detected)
        self.assertEqual(verified.speaker, "Annabelle", "a human verdict was cleared")
        self.assertIsNone(verified.speaker_detected, "an old guess would be scored as this run's")
        self.assertEqual(score_attribution(chap)["scored"], 0)


class TestAgreementCount(unittest.TestCase):
    def test_disagreements_are_not_counted_as_agreed(self):
        chap, k = chapter(3), knowledge("Annabelle", "Rico")
        primary = Fake(ChunkAnnotation(segments=[seg(1), seg(2), seg(3)]))
        second = Fake(ChunkAnnotation(segments=[seg(1), seg(2), seg(3, "Rico")]))
        result = asyncio.run(annotate_chapter(chap, k, primary, confirm_with=second))
        self.assertEqual(result["agreed"], 2)
        self.assertEqual(len(result["disagreements"]), 1)


class TestARunDoesNotOverwriteConcurrentEdits(unittest.TestCase):
    """detect_speakers takes minutes; edits made meanwhile must survive it."""

    def setUp(self):
        from vox_novel.pipeline.manager import NovelPipeline
        from vox_novel.storage.file import StorageManager
        from vox_novel.storage.knowledge import KnowledgeManager

        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.storage = StorageManager(base_dir=self.tmp)
        self.km = KnowledgeManager(base_dir=self.tmp)
        self.pipeline = NovelPipeline(storage_manager=self.storage, knowledge_manager=self.km)
        self.storage.save_chapter(chapter(2))
        k = self.km.load_or_init("b", target_lang="th")
        k.add_character("Annabelle", "Annabelle")
        k.add_character("Rico", "Rico")
        self.km.save(k)

    def human_edits(self):
        """What a reviewer does in the web UI while the run is in flight."""
        if getattr(self, "_edited", False):
            return
        self._edited = True
        chap = self.storage.get_chapter("b", "1")
        chap.paragraphs[0].speaker = "Rico"
        chap.paragraphs[0].speech_type = "dialogue"
        chap.paragraphs[0].speaker_verified = True
        self.storage.save_chapter(chap)
        k = self.km.load_or_init("b", target_lang="th")
        k.find_character("Rico").voice_description = "ชายหนุ่ม"
        self.km.save(k)

    def run_detection(self):
        t = Fake(
            ChunkAnnotation(segments=[seg(1), seg(2)]),
            before_each=self.human_edits,
        )
        with mock.patch.object(type(self.pipeline), "speaker_translator",
                               staticmethod(lambda model=None: t)):
            return asyncio.run(self.pipeline.detect_speakers("b", "1"))

    def test_a_verdict_entered_during_the_run_survives(self):
        self.run_detection()
        p = self.storage.get_chapter("b", "1").paragraphs[0]
        self.assertEqual(p.speaker, "Rico", "the run overwrote a human correction")
        self.assertTrue(p.speaker_verified)
        self.assertEqual(p.speaker_detected, "Annabelle", "the run's own guess was not recorded")

    def test_an_unverified_line_still_takes_the_new_attribution(self):
        self.run_detection()
        self.assertEqual(self.storage.get_chapter("b", "1").paragraphs[1].speaker, "Annabelle")

    def test_a_glossary_edit_during_the_run_survives(self):
        self.run_detection()
        k = self.km.load_or_init("b", target_lang="th")
        self.assertEqual(k.find_character("Rico").voice_description, "ชายหนุ่ม")
        self.assertGreater(k.find_character("Annabelle").line_count, 0, "the run's counts were lost")


class TestLineCountsDoNotPileUp(unittest.TestCase):
    """The registry the model sees is trimmed by line count once the cast is
    large, so a count that grows on every re-run pushes real speakers out."""

    def counts(self, k):
        return {c.name_en: c.line_count for c in k.characters.values()}

    def test_rerunning_the_same_result_leaves_counts_unchanged(self):
        chap, k = chapter(2), knowledge("Annabelle", "Rico")
        for _ in range(3):
            asyncio.run(annotate_chapter(chap, k, Fake(ChunkAnnotation(segments=[seg(1), seg(2, "Rico")]))))
        self.assertEqual(self.counts(k), {"Annabelle": 1, "Rico": 1})

    def test_a_changed_attribution_moves_the_count(self):
        chap, k = chapter(1), knowledge("Annabelle", "Rico")
        asyncio.run(annotate_chapter(chap, k, Fake(ChunkAnnotation(segments=[seg(1)]))))
        asyncio.run(annotate_chapter(chap, k, Fake(ChunkAnnotation(segments=[seg(1, "Rico")]))))
        self.assertEqual(self.counts(k), {"Annabelle": 0, "Rico": 1})

    def test_a_line_falling_back_to_narration_gives_its_count_back(self):
        chap, k = chapter(1), knowledge("Annabelle")
        asyncio.run(annotate_chapter(chap, k, Fake(ChunkAnnotation(segments=[seg(1)]))))
        asyncio.run(annotate_chapter(chap, k, Fake(RuntimeError("down"), RuntimeError("down"))))
        self.assertEqual(self.counts(k), {"Annabelle": 0})

    def test_a_verified_line_is_not_recounted(self):
        chap, k = chapter(1), knowledge("Annabelle", "Rico")
        asyncio.run(annotate_chapter(chap, k, Fake(ChunkAnnotation(segments=[seg(1)]))))
        chap.paragraphs[0].speaker_verified = True
        asyncio.run(annotate_chapter(chap, k, Fake(ChunkAnnotation(segments=[seg(1, "Rico")]))))
        self.assertEqual(self.counts(k), {"Annabelle": 1, "Rico": 0})


class TestOneDefinitionOfAReferenceClip(unittest.TestCase):
    """The review page labels a line with the voice synthesis will use. Both
    now call find_reference_clip; these pin what it accepts."""

    def setUp(self):
        self.voices = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.voices, True)
        self.char = knowledge("Rico").find_character("Rico")

    def test_an_upload_is_found(self):
        from vox_novel.storage.file import find_reference_clip
        clip = self.voices / "rico_ref.mp3"
        clip.write_bytes(b"x")
        self.assertEqual(find_reference_clip(self.voices, self.char), clip)

    def test_an_empty_upload_is_ignored(self):
        from vox_novel.storage.file import find_reference_clip
        (self.voices / "rico_ref.wav").write_bytes(b"")
        self.assertIsNone(find_reference_clip(self.voices, self.char))

    def test_a_generated_anchor_is_not_a_reference_clip(self):
        # Synthesis builds anchors from the description; treating one as an
        # upload would outrank a description edit made since.
        from vox_novel.storage.file import find_reference_clip
        (self.voices / "rico_ref.1a2b3c4d.wav").write_bytes(b"x")
        self.assertIsNone(find_reference_clip(self.voices, self.char))

    def test_an_explicit_path_wins(self):
        from vox_novel.storage.file import find_reference_clip
        (self.voices / "rico_ref.wav").write_bytes(b"x")
        explicit = self.voices / "chosen.wav"
        explicit.write_bytes(b"x")
        self.char.voice_ref_audio = str(explicit)
        self.assertEqual(find_reference_clip(self.voices, self.char), explicit)

    def test_both_callers_use_it(self):
        import inspect
        from vox_novel.tts import voxcpm
        from vox_novel.web import app
        self.assertIn("find_reference_clip(", inspect.getsource(voxcpm.VoxCPM2TTS))
        self.assertIn("find_reference_clip(", inspect.getsource(app._resolved_voice_name))


class TestTheGlossaryFindsOnlyThisCharactersClip(unittest.TestCase):
    """get_character_voice_file used to accept any file whose name contained
    the character's key, so the glossary played, and reset deleted, another
    character's voice."""

    def setUp(self):
        from vox_novel.storage.file import StorageManager, character_voice_key

        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.storage = StorageManager(base_dir=self.tmp)
        self.voices = self.storage.get_voices_dir("s")
        self.key = character_voice_key

    def put(self, name, data=b"x"):
        path = self.voices / name
        path.write_bytes(data)
        return path

    def test_a_longer_names_clip_is_not_this_characters(self):
        self.put(f"{self.key('Annabelle')}_ref.wav")
        self.assertIsNone(self.storage.get_character_voice_file("s", "Anna"))

    def test_a_generated_anchor_is_played_when_there_is_no_upload(self):
        anchor = self.put(f"{self.key('Anna')}_ref.1a2b3c4d.wav")
        self.assertEqual(self.storage.get_character_voice_file("s", "Anna"), anchor)

    def test_an_upload_outranks_an_anchor(self):
        self.put(f"{self.key('Anna')}_ref.1a2b3c4d.wav")
        upload = self.put(f"{self.key('Anna')}_ref.mp3")
        self.assertEqual(self.storage.get_character_voice_file("s", "Anna"), upload)

    def test_a_half_written_anchor_is_never_offered(self):
        self.put(f"{self.key('Anna')}_ref.1a2b3c4d.abc123.partial.wav")
        self.assertIsNone(self.storage.get_character_voice_file("s", "Anna"))

    def test_the_newest_anchor_is_current(self):
        import time
        old = self.put(f"{self.key('Anna')}_ref.00000000.wav")
        new = self.put(f"{self.key('Anna')}_ref.ffffffff.wav")
        os.utime(old, (time.time() - 60, time.time() - 60))
        self.assertEqual(self.storage.get_character_voice_file("s", "Anna"), new)

    def test_reset_removes_every_clip_and_nobody_elses(self):
        mine = [
            self.put(f"{self.key('Anna')}_ref.wav"),
            self.put(f"{self.key('Anna')}_ref.00000000.wav"),
            self.put(f"{self.key('Anna')}_ref.ffffffff.wav"),
        ]
        theirs = self.put(f"{self.key('Annabelle')}_ref.wav")
        self.assertTrue(self.storage.delete_voice_file("s", "character", "Anna"))
        self.assertFalse(any(p.exists() for p in mine), "an older anchor would come back as current")
        self.assertTrue(theirs.exists(), "reset deleted another character's voice")
        self.assertIsNone(self.storage.get_character_voice_file("s", "Anna"))


if __name__ == "__main__":
    unittest.main()
