import asyncio
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import soundfile as sf

from vox_novel.models.domain import Chapter, Paragraph
from vox_novel.tts.dummy import DummyTTS
from vox_novel.tts.registry import tts_registry
from vox_novel.tts.voxcpm import VoxCPM2TTS


def make_chapter() -> Chapter:
    # Index 2 is blank: it is skipped for synthesis, so chunk file names must follow
    # Paragraph.index rather than the position in the filtered list.
    texts = ["first", "", "third", "fourth"]
    return Chapter(
        id="170",
        book_id="series-a",
        title="Test",
        url="https://example.com/1",
        chapter_number=170.0,
        paragraphs=[
            Paragraph(id=f"p{i}", index=i, text=t, translated_text=t)
            for i, t in enumerate(texts, 1)
        ],
    )


class TTSTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "chapters"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def offline_voxcpm(self) -> VoxCPM2TTS:
        """A VoxCPM2 engine that never loads weights or calls a remote server.

        Keeps these tests hermetic and fast: with real weights installed the
        engine would otherwise run full inference for every paragraph.
        """
        engine = VoxCPM2TTS(api_url=None)
        engine.api_url = None
        patcher = mock.patch.object(engine, "_get_local_model", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)
        return engine


class TestChunkNaming(TTSTestCase):
    def test_dummy_names_chunks_by_paragraph_index(self):
        chapter = make_chapter()
        asyncio.run(DummyTTS().synthesize_chapter(chapter=chapter, output_dir=self.out))

        written = sorted(f.name for f in self.out.glob("para_*.wav"))
        self.assertEqual(
            written,
            ["para_170_1.wav", "para_170_3.wav", "para_170_4.wav"],
            "chunk names must follow Paragraph.index, skipping the empty paragraph",
        )

    def test_voxcpm_names_chunks_by_paragraph_index(self):
        chapter = make_chapter()
        engine = self.offline_voxcpm()  # no weights, no API -> placeholder tones
        asyncio.run(engine.synthesize_chapter(chapter=chapter, output_dir=self.out))

        written = sorted(f.name for f in self.out.glob("para_*.wav"))
        self.assertEqual(written, ["para_170_1.wav", "para_170_3.wav", "para_170_4.wav"])

    def test_paragraph_audio_path_matches_its_index(self):
        chapter = make_chapter()
        asyncio.run(DummyTTS().synthesize_chapter(chapter=chapter, output_dir=self.out))

        for p in chapter.paragraphs:
            if not p.text:
                continue
            self.assertEqual(Path(p.audio_path).name, f"para_170_{p.index}.wav")


class TestChapterAudioOutput(TTSTestCase):
    def test_master_file_written_at_the_chunk_sample_rate(self):
        chapter = make_chapter()
        engine = self.offline_voxcpm()
        asyncio.run(engine.synthesize_chapter(chapter=chapter, output_dir=self.out))

        final = self.out / "chapter_170.wav"
        self.assertTrue(final.exists())

        info = sf.info(str(final))
        chunk_info = sf.info(str(self.out / "para_170_1.wav"))
        self.assertEqual(
            info.samplerate,
            chunk_info.samplerate,
            "master audio rate must match the chunks or playback speed is wrong",
        )

    def test_placeholder_use_is_reported(self):
        chapter = make_chapter()
        engine = self.offline_voxcpm()
        messages = []
        asyncio.run(
            engine.synthesize_chapter(
                chapter=chapter,
                output_dir=self.out,
                progress_callback=lambda pct, msg: messages.append((pct, msg)),
            )
        )
        self.assertTrue(engine.used_placeholder)
        self.assertIn("PLACEHOLDER", messages[-1][1])

    def test_sync_progress_callback_is_supported(self):
        chapter = make_chapter()
        seen = []

        def sync_cb(pct, msg):
            seen.append(pct)

        asyncio.run(
            DummyTTS().synthesize_chapter(
                chapter=chapter, output_dir=self.out, progress_callback=sync_cb
            )
        )
        self.assertEqual(seen[-1], 100)

    def test_async_progress_callback_is_supported(self):
        chapter = make_chapter()
        seen = []

        async def async_cb(pct, msg):
            seen.append(pct)

        asyncio.run(
            DummyTTS().synthesize_chapter(
                chapter=chapter, output_dir=self.out, progress_callback=async_cb
            )
        )
        self.assertEqual(seen[-1], 100)


class TestEngineInterfaceParity(TTSTestCase):
    """Every engine must accept what the pipeline actually passes.

    The pipeline calls synthesize_chapter with translator=..., so an engine that
    omits the parameter raises TypeError at runtime while the rest of the suite
    still passes.
    """

    PIPELINE_KWARGS = dict(
        use_translated=True,
        voice_description=None,
        reference_audio=None,
        knowledge=None,
        progress_callback=None,
        translator=None,
    )

    def test_every_registered_engine_accepts_the_pipeline_call(self):
        for name in ("dummy", "voxcpm2"):
            engine = tts_registry.get_tts(name)
            if name == "voxcpm2":
                engine.api_url = None
                patcher = mock.patch.object(engine, "_get_local_model", return_value=None)
                patcher.start()
                self.addCleanup(patcher.stop)
            out = self.tmp / name
            try:
                asyncio.run(
                    engine.synthesize_chapter(
                        chapter=make_chapter(), output_dir=out, **self.PIPELINE_KWARGS
                    )
                )
            except TypeError as e:
                self.fail(f"{name} rejects the pipeline's call: {e}")

    def test_base_class_declares_the_same_parameters(self):
        import inspect

        from vox_novel.tts.base import BaseTTS

        base = set(inspect.signature(BaseTTS.synthesize_chapter).parameters)
        for name in ("dummy", "voxcpm2"):
            impl = set(inspect.signature(type(tts_registry.get_tts(name)).synthesize_chapter).parameters)
            missing = base - impl
            self.assertFalse(missing, f"{name} is missing {missing} from the BaseTTS contract")


class TestRegistry(TTSTestCase):
    def test_engines_accept_shared_cli_options(self):
        # The CLI forwards --api-url/--device to whichever engine is selected;
        # constructing an engine must not eagerly load any model.
        for name in ["dummy", "voxcpm2"]:
            engine = tts_registry.get_tts(name, api_url="http://localhost:9", device="cpu")
            self.assertIsNotNone(engine.name)

    def test_unknown_engine_raises(self):
        with self.assertRaises(ValueError):
            tts_registry.get_tts("nope")


if __name__ == "__main__":
    unittest.main()
