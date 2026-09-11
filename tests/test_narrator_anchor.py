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
            name = Path(kwargs["output_file"]).name
            # The anchor is written to a staging file and renamed into place, so
            # normalize it back to the name callers reason about.
            if name.startswith("narrator_ref"):
                name = "narrator_ref.wav"   # content-addressed; normalize for assertions
            calls.append((name, kwargs.get("control_prompt")))
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

    def write_supplied_anchor(self):
        import numpy as np
        import soundfile as sf

        self.voices.mkdir(parents=True, exist_ok=True)
        supplied = self.voices / "narrator_ref.wav"
        sf.write(supplied, np.zeros(2400, dtype=np.float32), 48000)
        return supplied

    def test_explicit_reference_argument_is_never_overwritten(self):
        supplied = self.write_supplied_anchor()
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

    def test_unstamped_file_is_kept_when_there_is_nothing_to_judge_by(self):
        """No knowledge object means no way to tell an upload from a legacy file.
        Losing a user's clip is worse than keeping a stale voice."""
        supplied = self.write_supplied_anchor()
        before = supplied.read_bytes()

        engine = self.offline_engine()
        asyncio.run(
            engine.synthesize_chapter(
                chapter=chapter(),
                output_dir=self.out,
                voice_description="เสียงบรรยายผู้ชาย",
                translator=Translator("a completely different voice"),
            )
        )
        self.assertEqual(
            supplied.read_bytes(), before, "an unstamped user file was regenerated"
        )

    def test_uploaded_reference_is_kept_even_with_knowledge_present(self):
        from vox_novel.models.series_knowledge import SeriesKnowledge

        supplied = self.write_supplied_anchor()
        before = supplied.read_bytes()

        k = SeriesKnowledge(series_id="s", target_language="th")
        k.update_narrator_voice(voice_ref_audio=str(supplied))   # what upload records

        engine = self.offline_engine()
        asyncio.run(
            engine.synthesize_chapter(
                chapter=chapter(), output_dir=self.out, knowledge=k,
                translator=Translator("a completely different voice"),
            )
        )
        self.assertEqual(supplied.read_bytes(), before, "an upload was regenerated")

    def test_legacy_anchor_is_adopted_not_replaced(self):
        """An install predating content-addressed anchors keeps the voice it has.

        Re-voicing it silently would change how an in-progress audiobook sounds;
        adopting it records the file as the series reference instead.
        """
        from vox_novel.models.series_knowledge import SeriesKnowledge

        legacy = self.write_supplied_anchor()
        before = legacy.read_bytes()

        k = SeriesKnowledge(series_id="s", target_language="th")
        k.update_narrator_voice(voice_description="เสียงเด็กหญิง หวานใส")

        engine = self.offline_engine()
        asyncio.run(
            engine.synthesize_chapter(
                chapter=chapter(), output_dir=self.out, knowledge=k,
                translator=Translator("young girl narrator, sweet"),
            )
        )
        self.assertEqual(legacy.read_bytes(), before, "a legacy anchor was overwritten")
        self.assertIsNone(
            k.narrator_voice_ref_audio,
            "recording it as the series reference would outrank the control prompt "
            "and make later description edits do nothing",
        )

    def test_a_legacy_anchor_does_not_freeze_later_edits(self):
        """Using a legacy file for one run must not pin the series voice."""
        from vox_novel.models.series_knowledge import SeriesKnowledge

        self.write_supplied_anchor()
        k = SeriesKnowledge(series_id="s", target_language="th")
        k.update_narrator_voice(voice_description="เสียงชายแก่")

        engine = self.offline_engine()
        asyncio.run(
            engine.synthesize_chapter(
                chapter=chapter("1"), output_dir=self.out, knowledge=k,
                translator=Translator("old male narrator"),
            )
        )
        # The description changes; the next chapter must not still use the legacy clip.
        k.update_narrator_voice(voice_description="เสียงเด็กหญิง หวานใส")
        (self.voices / "narrator_ref.wav").unlink()
        asyncio.run(
            engine.synthesize_chapter(
                chapter=chapter("2"), output_dir=self.out, knowledge=k,
                translator=Translator("young girl narrator, sweet"),
            )
        )
        self.assertTrue(
            list(self.voices.glob("narrator_ref.*.wav")),
            "a new anchor was never generated for the edited description",
        )


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


class TestConcurrentAnchorBuild(AnchorTestCase):
    """Two chapters of one series may synthesize at once (two browser tabs).

    Both find the anchor absent and both build it. Whoever lands second must not
    replace a file the other is already cloning from, or that chapter's narrator
    changes part-way through.
    """

    def test_concurrent_first_build_yields_one_anchor_and_one_voice(self):
        import soundfile as sf
        from vox_novel.models.series_knowledge import SeriesKnowledge

        k = SeriesKnowledge(series_id="s", target_language="th")
        k.update_narrator_voice(
            voice_description="เสียงชายแก่", voice_control_prompt="steady narrator"
        )
        e1, e2 = self.offline_engine(), self.offline_engine()

        async def both():
            await asyncio.gather(
                e1.synthesize_chapter(
                    chapter=chapter("c1", n=4), output_dir=self.out, knowledge=k,
                    translator=Translator("steady narrator"),
                ),
                e2.synthesize_chapter(
                    chapter=chapter("c2", n=4), output_dir=self.out, knowledge=k,
                    translator=Translator("steady narrator"),
                ),
            )

        asyncio.run(both())

        anchors = list(self.voices.glob("narrator_ref.*.wav"))
        self.assertEqual(len(anchors), 1, f"expected one anchor, got {[a.name for a in anchors]}")
        self.assertEqual(
            [f.name for f in self.voices.iterdir() if ".partial." in f.name],
            [],
            "a staging file was left behind",
        )

        # Every paragraph of one chapter clones the same anchor, so levels match.
        levels = {
            round(float(abs(sf.read(f)[0]).max()), 3)
            for f in sorted(self.out.glob("para_c1_*.wav"))
        }
        self.assertEqual(len(levels), 1, f"the narrator changed mid-chapter: {sorted(levels)}")


class TestAnchorIsVisibleToTheWebLayer(AnchorTestCase):
    """A generated anchor must be reachable by the Voice Studio, or the player is
    empty and the reset button hidden for every series voiced by plain synthesis."""

    def storage(self):
        from vox_novel.storage.file import StorageManager

        base = self.tmp / "store"
        (base / "s1" / "voices").mkdir(parents=True)
        return StorageManager(base_dir=base), base / "s1" / "voices"

    def test_a_content_addressed_anchor_is_found(self):
        st, voices = self.storage()
        (voices / "narrator_ref.8443cafd.wav").write_bytes(b"anchor")
        self.assertIsNotNone(st.get_narrator_voice_file("s1"))

    def test_an_upload_outranks_a_generated_anchor(self):
        st, voices = self.storage()
        (voices / "narrator_ref.8443cafd.wav").write_bytes(b"anchor")
        (voices / "narrator_ref.wav").write_bytes(b"upload")
        self.assertEqual(st.get_narrator_voice_file("s1").name, "narrator_ref.wav")

    def test_reset_removes_every_anchor(self):
        st, voices = self.storage()
        for name in ("narrator_ref.wav", "narrator_ref.aaaa1111.wav", "narrator_ref.bbbb2222.wav"):
            (voices / name).write_bytes(b"x")
        self.assertTrue(st.delete_voice_file("s1", "narrator"))
        self.assertIsNone(
            st.get_narrator_voice_file("s1"), "an older anchor resurfaced after a reset"
        )

    def test_a_staging_file_is_never_offered(self):
        st, voices = self.storage()
        (voices / "narrator_ref.8443cafd.1.partial.wav").write_bytes(b"half written")
        self.assertIsNone(st.get_narrator_voice_file("s1"))


class TestControlPromptCaching(AnchorTestCase):
    """The cache is what keeps a series' narrator voice stable between chapters.

    Re-deriving per chapter returns varying phrasings (the LLM runs at
    temperature 0.3), which changes the stamp, rebuilds the anchor, and drifts the
    voice chapter to chapter.
    """

    def synth(self, engine, knowledge, translator, override=None, cid="1"):
        asyncio.run(
            engine.synthesize_chapter(
                chapter=chapter(cid),
                output_dir=self.out,
                voice_description=override,
                knowledge=knowledge,
                translator=translator,
            )
        )

    def knowledge_with_cached_prompt(self):
        from vox_novel.models.series_knowledge import SeriesKnowledge

        k = SeriesKnowledge(series_id="s", target_language="th")
        k.update_narrator_voice(
            voice_description="เสียงชายแก่", voice_control_prompt="cached male narrator"
        )
        return k

    def counting_translator(self):
        calls = []

        class T:
            async def complete(self, system_prompt, user_prompt):
                calls.append(user_prompt)
                return "freshly derived narrator"

        return T(), calls

    def test_normal_path_uses_the_cache_and_makes_no_llm_call(self):
        k = self.knowledge_with_cached_prompt()
        t, calls = self.counting_translator()
        self.synth(self.offline_engine(), k, t)
        self.assertEqual(calls, [], "the cached prompt should have been reused")
        self.assertEqual(k.narrator_voice_control_prompt, "cached male narrator")

    def test_an_explicit_override_bypasses_the_cache(self):
        k = self.knowledge_with_cached_prompt()
        t, calls = self.counting_translator()
        self.synth(self.offline_engine(), k, t, override="เด็กหญิง เสียงหวานใส")
        self.assertEqual(len(calls), 1, "an override must be derived, not read from cache")
        self.assertIn("เด็กหญิง", calls[0])

    def anchors(self):
        return sorted(f.name for f in self.voices.glob("narrator_ref.*.wav"))

    def test_repeated_chapters_reuse_one_anchor(self):
        k = self.knowledge_with_cached_prompt()
        t, _ = self.counting_translator()
        engine = self.offline_engine()
        for cid in ("1", "2", "3"):
            self.synth(engine, k, t, cid=cid)
        self.assertEqual(len(self.anchors()), 1, f"the voice drifted: {self.anchors()}")

    def test_an_override_does_not_disturb_the_series_anchor(self):
        """A one-off `--voice` run gets its own anchor; the series voice is untouched."""
        k = self.knowledge_with_cached_prompt()
        t, _ = self.counting_translator()
        engine = self.offline_engine()

        self.synth(engine, k, t, cid="1")
        series_anchor = self.anchors()[0]
        series_bytes = (self.voices / series_anchor).read_bytes()

        self.synth(engine, k, t, override="เด็กหญิง เสียงหวานใส", cid="2")
        self.synth(engine, k, t, cid="3")

        self.assertEqual(
            (self.voices / series_anchor).read_bytes(),
            series_bytes,
            "the override re-rolled the series narrator",
        )


class TestAnchorArtifacts(AnchorTestCase):
    def test_no_anchor_artifact_is_left_beside_the_chapter_audio(self):
        engine = self.offline_engine()
        self.run_chapter(engine, Translator("steady narrator"))
        strays = [f.name for f in self.out.iterdir() if "anchor" in f.name.lower()]
        self.assertEqual(strays, [], "a pinned anchor copy was left in the chapters dir")

    def test_a_stray_artifact_is_not_served_as_chapter_audio(self):
        from vox_novel.storage.file import StorageManager

        base = self.tmp / "store"
        chapters = base / "s1" / "chapters"
        chapters.mkdir(parents=True)
        (chapters / ".anchor_2.wav").write_bytes(b"not a chapter")
        (chapters / "para_2_1.wav").write_bytes(b"not a chapter either")

        got = StorageManager(base_dir=base).get_chapter_audio_file("s1", "2")
        self.assertIsNone(got, f"served {got} as chapter audio")


class TestPipelineDoesNotDefeatTheCache(unittest.TestCase):
    """Exercise the layer the regression actually lived in.

    synthesize_chapter_audio used to collapse the per-chapter `voice` override and
    the series description into one argument, so every call looked like an override
    and the prompt cache was bypassed -- one LLM call per chapter, a new phrasing
    each time, and a rebuilt anchor.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

        from vox_novel.pipeline.manager import NovelPipeline
        from vox_novel.storage.file import StorageManager
        from vox_novel.storage.knowledge import KnowledgeManager

        self.storage = StorageManager(base_dir=self.tmp)
        self.km = KnowledgeManager(base_dir=self.tmp)
        self.pipeline = NovelPipeline(storage_manager=self.storage, knowledge_manager=self.km)

        self.storage.save_chapter(chapter("7"))
        k = self.km.load_or_init("b")
        k.update_narrator_voice(
            voice_description="เสียงชายแก่", voice_control_prompt="cached male narrator"
        )
        self.km.save(k)

    def synthesize(self, voice=None):
        calls = []

        class T:
            async def complete(self, system_prompt, user_prompt):
                calls.append(user_prompt)
                return "freshly derived narrator"

        engine = VoxCPM2TTS(api_url=None)
        engine.api_url = None
        with mock.patch.object(engine, "_get_local_model", return_value=None), \
             mock.patch("vox_novel.pipeline.manager.tts_registry.get_tts", return_value=engine), \
             mock.patch.object(
                 type(self.pipeline), "voice_prompt_translator", staticmethod(lambda: T())
             ):
            asyncio.run(
                self.pipeline.synthesize_chapter_audio(
                    series_id="b", chapter_id="7", voice_description=voice
                )
            )
        return calls

    def test_a_plain_chapter_reuses_the_cached_prompt(self):
        self.assertEqual(
            self.synthesize(), [], "the pipeline bypassed the cache on a normal chapter"
        )

    def test_an_override_is_still_derived(self):
        calls = self.synthesize(voice="เด็กหญิง เสียงหวานใส")
        self.assertEqual(len(calls), 1)
        self.assertIn("เด็กหญิง", calls[0])

    def test_an_override_is_not_persisted_as_the_series_voice(self):
        self.synthesize(voice="เด็กหญิง เสียงหวานใส")
        k = self.km.load_or_init("b")
        self.assertEqual(k.narrator_voice_description, "เสียงชายแก่")
        self.assertEqual(
            k.narrator_voice_control_prompt,
            "cached male narrator",
            "a one-off override overwrote the series voice",
        )


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

    def test_hot_entry_is_never_re_encoded(self):
        """Count disk reads, not membership.

        Asserting the anchor is still cached at the end passes under FIFO too,
        because the final touch reinserts it. What distinguishes LRU is that a
        constantly-used entry is never evicted, so it is read from disk once.
        """
        engine = VoxCPM2TTS()
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)

        anchor = tmp / "narrator_ref.wav"
        anchor.write_bytes(b"anchor")

        reads = []
        real_read = Path.read_bytes

        def counting_read(self):
            reads.append(str(self))
            return real_read(self)

        with mock.patch.object(Path, "read_bytes", counting_read):
            engine._encoded_reference(anchor)
            for i in range(VoxCPM2TTS.MAX_REF_CACHE_ENTRIES + 4):
                f = tmp / f"other{i}.wav"
                f.write_bytes(b"y" * 16)
                engine._encoded_reference(f)
                engine._encoded_reference(anchor)  # reused every round, as in a chapter

        self.assertEqual(
            reads.count(str(anchor)), 1, "the hot entry was evicted and re-encoded"
        )


if __name__ == "__main__":
    unittest.main()
