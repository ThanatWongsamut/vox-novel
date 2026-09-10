import asyncio
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vox_novel.models.domain import Chapter, Paragraph
from vox_novel.tts.voxcpm import VoxCPM2TTS


def chapter(cid="1", n=2):
    return Chapter(
        id=cid,
        book_id="b",
        title="t",
        url="u",
        chapter_number=1.0,
        paragraphs=[
            Paragraph(id=f"p{i}", index=i, text=f"line {i}", translated_text=f"line {i}")
            for i in range(1, n + 1)
        ],
    )


class Translator:
    def __init__(self, reply):
        self.reply = reply

    async def complete(self, system_prompt, user_prompt):
        return self.reply


class AnchorTestCase(unittest.TestCase):
    """The narrator anchor is what every paragraph clones from.

    The local engine ignores a control prompt once reference_wav_path is set, so
    if the anchor is built from the wrong prompt the derived voice reaches nothing.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "chapters"
        self.voices = self.tmp / "voices"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def offline_engine(self):
        engine = VoxCPM2TTS(api_url=None)
        engine.api_url = None
        patcher = mock.patch.object(engine, "_get_local_model", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)
        return engine

    def run_chapter(self, engine, translator=None, description="เสียงบรรยายผู้ชาย", cid="1"):
        """Synthesize, recording the control prompt handed to each synthesize call."""
        calls = []
        original = engine.synthesize

        async def spy(**kwargs):
            calls.append((Path(kwargs["output_file"]).name, kwargs.get("control_prompt")))
            return await original(**kwargs)

        with mock.patch.object(engine, "synthesize", spy):
            asyncio.run(
                engine.synthesize_chapter(
                    chapter=chapter(cid),
                    output_dir=self.out,
                    voice_description=description,
                    translator=translator,
                )
            )
        return dict(calls)


class TestAnchorReceivesDerivedPrompt(AnchorTestCase):
    DERIVED = "45-year-old male narrator, gravelly, weary"

    def test_anchor_is_built_from_the_derived_prompt(self):
        calls = self.run_chapter(self.offline_engine(), Translator(self.DERIVED))
        self.assertEqual(
            calls.get("narrator_ref.wav"),
            self.DERIVED,
            "the anchor must carry the derived prompt, or deriving it is pointless",
        )

    def test_paragraphs_use_the_same_prompt(self):
        calls = self.run_chapter(self.offline_engine(), Translator(self.DERIVED))
        for name, control in calls.items():
            self.assertEqual(control, self.DERIVED, name)

    def test_without_a_translator_the_table_prompt_is_used(self):
        calls = self.run_chapter(self.offline_engine(), translator=None)
        self.assertTrue(calls.get("narrator_ref.wav"), "anchor still needs a control prompt")
        self.assertTrue(calls["narrator_ref.wav"].isascii())


class TestAnchorStaleness(AnchorTestCase):
    """A description edit must reach chapter 2, not just the first synthesis."""

    def test_anchor_is_rebuilt_when_the_control_changes(self):
        engine = self.offline_engine()
        first = self.run_chapter(engine, Translator("old male narrator, slow"), cid="1")
        self.assertEqual(first.get("narrator_ref.wav"), "old male narrator, slow")

        second = self.run_chapter(engine, Translator("young female narrator, bright"), cid="2")
        self.assertEqual(
            second.get("narrator_ref.wav"),
            "young female narrator, bright",
            "a stale anchor pins the whole series to the previous voice",
        )

    def test_anchor_is_reused_when_the_control_is_unchanged(self):
        engine = self.offline_engine()
        self.run_chapter(engine, Translator("steady male narrator"), cid="1")
        second = self.run_chapter(engine, Translator("steady male narrator"), cid="2")
        self.assertNotIn(
            "narrator_ref.wav", second, "an unchanged voice must not be re-synthesized"
        )

    def test_user_supplied_reference_is_never_overwritten(self):
        import numpy as np
        import soundfile as sf

        self.voices.mkdir(parents=True, exist_ok=True)
        supplied = self.voices / "narrator_ref.wav"
        sf.write(supplied, np.zeros(2400, dtype=np.float32), 48000)
        before = supplied.read_bytes()

        engine = self.offline_engine()
        asyncio.run(
            engine.synthesize_chapter(
                chapter=chapter(),
                output_dir=self.out,
                reference_audio=supplied,
                translator=Translator("some other voice"),
            )
        )
        self.assertEqual(supplied.read_bytes(), before, "uploaded reference was clobbered")


class TestEmotionTraits(unittest.TestCase):
    def test_only_affect_traits_are_returned(self):
        # "เสียงหญิงชราโกรธ" also yields "elderly", which would append an age to
        # whoever is speaking this one paragraph.
        self.assertEqual(VoxCPM2TTS._emotion_traits("เสียงหญิงชราโกรธ"), "angry")
        self.assertEqual(VoxCPM2TTS._emotion_traits("นุ่มลึก โกรธ"), "angry")

    def test_plain_emotions_pass_through(self):
        self.assertEqual(VoxCPM2TTS._emotion_traits("โกรธ"), "angry")
        self.assertEqual(VoxCPM2TTS._emotion_traits("กระซิบ"), "whispering")

    def test_unrecognised_emotion_yields_nothing(self):
        self.assertEqual(VoxCPM2TTS._emotion_traits("ไม่มีคำนี้เลย"), "")
        self.assertEqual(VoxCPM2TTS._emotion_traits(None), "")
        self.assertEqual(VoxCPM2TTS._emotion_traits(""), "")


class TestReferenceCache(unittest.TestCase):
    def test_cache_is_bounded(self):
        engine = VoxCPM2TTS()
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        for i in range(VoxCPM2TTS.MAX_REF_CACHE_ENTRIES + 8):
            f = tmp / f"ref{i}.wav"
            f.write_bytes(b"x" * 16)
            engine._encoded_reference(f)
        self.assertLessEqual(len(engine._ref_cache), VoxCPM2TTS.MAX_REF_CACHE_ENTRIES)

    def test_hot_entry_survives_eviction(self):
        """The narrator anchor is reused constantly; FIFO would evict it each cycle."""
        engine = VoxCPM2TTS()
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)

        anchor = tmp / "narrator_ref.wav"
        anchor.write_bytes(b"anchor")
        engine._encoded_reference(anchor)

        for i in range(VoxCPM2TTS.MAX_REF_CACHE_ENTRIES + 4):
            f = tmp / f"other{i}.wav"
            f.write_bytes(b"y" * 16)
            engine._encoded_reference(f)
            engine._encoded_reference(anchor)  # touched every round, as in a chapter

        stat = anchor.stat()
        self.assertIn((str(anchor), stat.st_mtime_ns, stat.st_size), engine._ref_cache)


if __name__ == "__main__":
    unittest.main()
