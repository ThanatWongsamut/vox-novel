"""Generate in the Voice Studio saves a designed voice, not an upload.

It used to write the sample to `<key>_ref.wav` -- the upload's name -- and
record it as the character's reference clip. An upload outranks the voice
description, so after one Generate, editing the description and saving it
changed nothing: synthesis and the preview kept using the old sample until
Generate was clicked again. It also wrote straight over a clip the user had
uploaded.
"""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from vox_novel.storage.file import anchor_digest, character_voice_key, find_reference_clip
from vox_novel.tts.voxcpm import VoxCPM2TTS
from vox_novel.web import app as web


class GenerateTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        roots = [web.storage, web.knowledge_mgr, web.pipeline.storage, web.pipeline.knowledge]
        originals = [m.base_dir for m in roots]
        for m in roots:
            m.base_dir = self.tmp
        self.addCleanup(lambda: [setattr(m, "base_dir", o) for m, o in zip(roots, originals)])

        # Offline: no model weights, no remote TTS, and no LLM -- the prompt
        # translator would otherwise call OpenRouter with a developer's real key.
        real_init = VoxCPM2TTS.__init__

        def offline_init(self, *args, **kwargs):
            real_init(self, *args, **kwargs)
            self.api_url = None  # set per instance from the environment

        for target, attr, value in (
            (VoxCPM2TTS, "__init__", offline_init),
            (VoxCPM2TTS, "_get_local_model", lambda self: None),
            (type(web.pipeline), "voice_prompt_translator", lambda self: None),
        ):
            patcher = mock.patch.object(target, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.client = TestClient(web.web_app)
        k = web.knowledge_mgr.load_or_init("s", target_lang="th")
        k.add_character("Rico", "ริโก้", gender="female")
        web.knowledge_mgr.save(k)
        self.voices = web.storage.get_voices_dir("s")
        self.key = character_voice_key("Rico")

    def generate(self, description, voice_type="character"):
        body = {"series_id": "s", "target_lang": "th", "voice_type": voice_type,
                "voice_description": description}
        if voice_type == "character":
            body["character_name"] = "Rico"
        return self.client.post("/api/voices/generate", json=body)

    def save_description(self, description):
        return self.client.post("/api/voices/update-prompt", json={
            "series_id": "s", "target_lang": "th", "voice_type": "character",
            "character_name": "Rico", "voice_description": description,
        })

    def rico(self):
        return web.knowledge_mgr.load_or_init("s", target_lang="th").find_character("Rico")


class TestGenerateSavesADesignedVoice(GenerateTestCase):
    def test_the_sample_is_the_anchor_synthesis_will_clone(self):
        self.assertEqual(self.generate("หญิงสาว เสียงหวาน").status_code, 200)
        rico = self.rico()
        anchor = self.voices / f"{self.key}_ref.{anchor_digest(rico.voice_control_prompt)}.wav"
        self.assertTrue(anchor.is_file(), "the sample was not saved under the anchor name")
        self.assertFalse((self.voices / f"{self.key}_ref.wav").exists(), "saved as an upload")
        self.assertIsNone(rico.voice_ref_audio)

    def test_a_description_edit_after_generate_takes_effect(self):
        # The reported bug: the old sample outranked the new description.
        self.generate("หญิงสาว เสียงหวาน")
        before = self.rico().voice_control_prompt
        self.assertEqual(self.save_description("ชายชรา เสียงแหบ").status_code, 200)
        rico = self.rico()
        self.assertNotEqual(rico.voice_control_prompt, before)
        self.assertIsNone(
            find_reference_clip(self.voices, rico),
            "a stale clip still outranks the new description",
        )
        # Synthesis will build the new voice rather than clone the old sample.
        self.assertFalse(
            (self.voices / f"{self.key}_ref.{anchor_digest(rico.voice_control_prompt)}.wav").exists()
        )

    def test_the_preview_plays_the_generated_sample(self):
        self.generate("หญิงสาว เสียงหวาน")
        res = self.client.get("/api/voices/s/character/Rico?lang=th")
        self.assertEqual(res.status_code, 200)

    def test_the_preview_stops_playing_it_after_a_description_edit(self):
        self.generate("หญิงสาว เสียงหวาน")
        self.save_description("ชายชรา เสียงแหบ")
        self.assertEqual(self.client.get("/api/voices/s/character/Rico?lang=th").status_code, 404)

    def test_generate_replaces_an_upload(self):
        # Choosing a designed voice; an upload would outrank it forever.
        (self.voices / f"{self.key}_ref.wav").write_bytes(b"uploaded")
        self.generate("หญิงสาว เสียงหวาน")
        self.assertFalse((self.voices / f"{self.key}_ref.wav").exists())
        self.assertIsNone(find_reference_clip(self.voices, self.rico()))

    def test_a_failed_generate_keeps_the_upload(self):
        upload = self.voices / f"{self.key}_ref.wav"
        upload.write_bytes(b"uploaded")
        with mock.patch.object(VoxCPM2TTS, "design_anchor", side_effect=RuntimeError("gpu")):
            with self.assertRaises(RuntimeError):
                self.generate("หญิงสาว เสียงหวาน")
        self.assertEqual(upload.read_bytes(), b"uploaded")

    def test_generating_again_re_rolls_in_place(self):
        self.generate("หญิงสาว เสียงหวาน")
        self.generate("หญิงสาว เสียงหวาน")
        anchors = list(self.voices.glob(f"{self.key}_ref.*.wav"))
        self.assertEqual(len(anchors), 1, anchors)
        self.assertFalse(list(self.voices.glob("*.partial.wav")))


class TestNarratorGenerate(GenerateTestCase):
    def test_the_narrator_sample_is_an_anchor_too(self):
        self.assertEqual(self.generate("เสียงบรรยายผู้ชาย", voice_type="narrator").status_code, 200)
        k = web.knowledge_mgr.load_or_init("s", target_lang="th")
        anchor = self.voices / f"narrator_ref.{anchor_digest(k.narrator_voice_control_prompt)}.wav"
        self.assertTrue(anchor.is_file())
        self.assertFalse((self.voices / "narrator_ref.wav").exists())
        self.assertIsNone(k.narrator_voice_ref_audio)
        self.assertEqual(web.storage.get_narrator_voice_file("s"), anchor)


if __name__ == "__main__":
    unittest.main()
