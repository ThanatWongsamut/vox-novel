import json
import os
import shutil
import tempfile
import time
import unittest
import uuid
from pathlib import Path

from unittest import mock

from fastapi.testclient import TestClient

from vox_novel.pipeline.manager import NovelPipeline
from vox_novel.tts.voxcpm import VoxCPM2TTS
from vox_novel.web import app as web


class ExtensionApiTestCase(unittest.TestCase):
    """Drives the real FastAPI app against a throwaway storage root."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        # The web module and the pipeline each hold their own manager instances;
        # both roots must point at the throwaway directory.
        self._roots = [
            web.storage,
            web.knowledge_mgr,
            web.pipeline.storage,
            web.pipeline.knowledge,
        ]
        self._orig_roots = [m.base_dir for m in self._roots]
        for manager in self._roots:
            manager.base_dir = self.tmp
        self.client = TestClient(web.web_app)

    def tearDown(self):
        for manager, original in zip(self._roots, self._orig_roots):
            manager.base_dir = original
        shutil.rmtree(self.tmp, ignore_errors=True)

    def payload(self, **overrides):
        body = {
            "url": "https://readtoon.com/content/series-a/170",
            "series_id": "series-a",
            "series_title": "Test Novel",
            "chapter_no": 170,
            "chapter_title": "ตอนที่ 170 - บทเรียนแรก",
            "paragraphs": ["บรรทัดหนึ่ง", "“สวัสดี”", "บรรทัดสาม"],
            "source": "readtoon",
            "source_language": "th",
        }
        body.update(overrides)
        return body

    def saved_chapter(self, series_id="series-a"):
        files = list((self.tmp / series_id / "chapters").glob("*.json"))
        self.assertEqual(len(files), 1, f"expected one chapter file, got {files}")
        return json.loads(files[0].read_text(encoding="utf-8"))


class TestExtensionImport(ExtensionApiTestCase):
    def test_import_persists_chapter_and_novel(self):
        res = self.client.post("/api/extension/import", json=self.payload())
        self.assertEqual(res.status_code, 200, res.text)

        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["series_id"], "series-a")
        self.assertEqual(data["chapter_id"], "170")
        self.assertEqual(data["paragraph_count"], 3)
        self.assertEqual(data["read_url"], "/series/series-a/read/170")

        chapter = self.saved_chapter()
        self.assertEqual(chapter["chapter_number"], 170.0)
        self.assertEqual(chapter["source_language"], "th")

        novel = json.loads((self.tmp / "series-a" / "novel_info.json").read_text(encoding="utf-8"))
        self.assertEqual(novel["title"], "Test Novel")
        self.assertEqual(len(novel["chapters"]), 1)

    def test_paragraph_indices_are_one_based_and_contiguous(self):
        self.client.post("/api/extension/import", json=self.payload())
        paragraphs = self.saved_chapter()["paragraphs"]
        self.assertEqual([p["index"] for p in paragraphs], [1, 2, 3])

    def test_dialogue_detection_uses_shared_quote_set(self):
        body = self.payload(paragraphs=["เล่าเรื่อง", "『คำพูด』", "“คำพูด”"])
        self.client.post("/api/extension/import", json=body)
        types = [p["speech_type"] for p in self.saved_chapter()["paragraphs"]]
        self.assertEqual(types, ["narration", "dialogue", "dialogue"])

    def test_reimport_updates_in_place_without_duplicates(self):
        self.client.post("/api/extension/import", json=self.payload())
        res = self.client.post(
            "/api/extension/import",
            json=self.payload(paragraphs=["ก", "ข", "ค", "ง"]),
        )
        self.assertEqual(res.status_code, 200, res.text)

        novel = json.loads((self.tmp / "series-a" / "novel_info.json").read_text(encoding="utf-8"))
        self.assertEqual(len(novel["chapters"]), 1, "re-import duplicated the chapter entry")
        self.assertEqual(len(self.saved_chapter()["paragraphs"]), 4)

    def test_unnumbered_import_does_not_bind_to_another_unnumbered_chapter(self):
        first = self.payload(
            url="https://example.com/story/a",
            chapter_no=None,
            chapter_title="Prologue",
        )
        second = self.payload(
            url="https://example.com/story/b",
            chapter_no=None,
            chapter_title="Epilogue",
        )
        r1 = self.client.post("/api/extension/import", json=first)
        r2 = self.client.post("/api/extension/import", json=second)
        self.assertEqual(r1.status_code, 200, r1.text)
        self.assertEqual(r2.status_code, 200, r2.text)
        self.assertNotEqual(r1.json()["chapter_id"], r2.json()["chapter_id"])

        novel = json.loads((self.tmp / "series-a" / "novel_info.json").read_text(encoding="utf-8"))
        titles = sorted(c["title"] for c in novel["chapters"])
        self.assertEqual(titles, ["Epilogue", "Prologue"])

    def test_empty_content_is_rejected(self):
        res = self.client.post("/api/extension/import", json=self.payload(paragraphs=["", "   "]))
        self.assertEqual(res.status_code, 400)

    def test_health_endpoint(self):
        res = self.client.get("/api/extension/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "ok")


class TestPathTraversalRejected(ExtensionApiTestCase):
    def test_import_rejects_traversing_series_id(self):
        marker = f"vox-escape-{uuid.uuid4().hex}"
        escape_target = self.tmp.parent / marker
        self.addCleanup(shutil.rmtree, escape_target, True)

        res = self.client.post(
            "/api/extension/import",
            json=self.payload(series_id=f"../{marker}"),
        )
        self.assertEqual(res.status_code, 400, res.text)
        self.assertFalse(escape_target.exists(), "import escaped the storage root")

    def test_import_rejects_absolute_series_id(self):
        res = self.client.post("/api/extension/import", json=self.payload(series_id="/tmp/evil"))
        self.assertEqual(res.status_code, 400)

    def test_series_id_derived_from_url_is_also_validated(self):
        res = self.client.post(
            "/api/extension/import",
            json=self.payload(series_id=None, url="https://readtoon.com/content/..%2f..%2fetc/1"),
        )
        self.assertEqual(res.status_code, 400, res.text)

    # An encoded slash (%2F) is decoded before routing, so it can never smuggle a
    # separator into a single path parameter -- those requests 404 at the router.
    # An encoded dot (%2E) does survive as one segment, which is the vector that
    # actually reaches these handlers.
    def test_audio_endpoints_reject_traversal(self):
        for path in [
            "/api/audio/%2e%2e/1",
            "/api/audio/series-a/%2e%2e",
            "/api/audio/%2e%2e/1/para/1",
            "/api/voices/%2e%2e/narrator",
            "/api/voices/%2e%2e/character/bob",
            "/api/cover/%2e%2e",
        ]:
            res = self.client.get(path)
            self.assertEqual(res.status_code, 400, f"{path} -> {res.status_code}")

    def test_page_routes_reject_traversal(self):
        for path in [
            "/series/%2e%2e",
            "/series/series-a/read/%2e%2e",
            "/series/%2e%2e/glossary",
        ]:
            res = self.client.get(path)
            self.assertEqual(res.status_code, 400, f"{path} -> {res.status_code}")

    def test_encoded_slash_cannot_reach_a_path_parameter(self):
        # Documents the router-level protection the validation complements.
        res = self.client.get("/api/cover/..%2F..%2Fetc")
        self.assertEqual(res.status_code, 404)

    def test_negative_paragraph_index_rejected(self):
        res = self.client.get("/api/audio/series-a/170/para/-1")
        self.assertIn(res.status_code, (400, 422))


class TestSynthesisJob(ExtensionApiTestCase):
    def test_job_reports_placeholder_audio_instead_of_claiming_success(self):
        self.client.post("/api/extension/import", json=self.payload())

        # Patch before the request: the job starts synthesizing immediately, and
        # real weights would otherwise run full inference here.
        # No weights, no remote TTS, and no LLM: the suite must not touch the network.
        with mock.patch.object(VoxCPM2TTS, "_get_local_model", return_value=None), \
             mock.patch.object(NovelPipeline, "voice_prompt_translator", staticmethod(lambda: None)), \
             mock.patch.dict(os.environ, {"VOXCPM_API_URL": ""}, clear=False):
            res = self.client.post(
                "/api/synthesize-chapter",
                data={"series_id": "series-a", "chapter_id": "170", "engine": "voxcpm2"},
            )
            self.assertEqual(res.status_code, 200, res.text)
            job_id = res.json()["job_id"]

            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if web.JOBS[job_id]["status"] in ("completed", "failed"):
                    break
                time.sleep(0.05)

        job = web.JOBS[job_id]
        self.assertEqual(job["status"], "completed", job.get("error"))
        self.assertIn("PLACEHOLDER", job["message"])

    def test_synthesis_rejects_traversing_ids(self):
        res = self.client.post(
            "/api/synthesize-chapter",
            data={"series_id": "../evil", "chapter_id": "170"},
        )
        self.assertEqual(res.status_code, 400)


class TestVoicePromptValidation(ExtensionApiTestCase):
    def test_invalid_voice_type_is_a_client_error(self):
        # Previously fell through to an UnboundLocalError on `char` -> HTTP 500.
        res = self.client.post(
            "/api/voices/update-prompt",
            json={"series_id": "series-a", "voice_type": "bogus", "voice_description": "x"},
        )
        self.assertEqual(res.status_code, 400, res.text)

    def test_missing_voice_type_is_a_client_error(self):
        res = self.client.post(
            "/api/voices/update-prompt", json={"series_id": "series-a"}
        )
        self.assertEqual(res.status_code, 400)

    def test_character_without_a_name_is_a_client_error(self):
        res = self.client.post(
            "/api/voices/update-prompt",
            json={"series_id": "series-a", "voice_type": "character"},
        )
        self.assertEqual(res.status_code, 400)


class TestVoicePromptPersistence(ExtensionApiTestCase):
    """The description and its derived control prompt must land in one write.

    update_narrator_voice clears the cached control whenever the description
    changes, so writing them separately relies on ordering that is easy to break.
    """

    class FailingTranslator:
        async def complete(self, *a, **k):
            raise RuntimeError("LLM down")

    def knowledge(self, series="s1"):
        return json.loads((self.tmp / series / "knowledge_th.json").read_text(encoding="utf-8"))

    def test_description_and_control_are_both_persisted(self):
        from vox_novel.pipeline.manager import NovelPipeline

        # No translator: the keyword table fills in, and the suite stays offline.
        with mock.patch.object(
            NovelPipeline, "voice_prompt_translator", staticmethod(lambda: None)
        ):
            res = self.client.post(
            "/api/voices/update-prompt",
                json={
                    "series_id": "s1",
                    "voice_type": "narrator",
                    "voice_description": "เสียงหวานใส ร่าเริง",
                },
            )
        self.assertEqual(res.status_code, 200, res.text)
        k = self.knowledge()
        self.assertEqual(k["narrator_voice_description"], "เสียงหวานใส ร่าเริง")
        self.assertTrue(
            k["narrator_voice_control_prompt"],
            "the control prompt must survive the description write",
        )

    def test_llm_failure_still_saves_both(self):
        from vox_novel.pipeline.manager import NovelPipeline

        with mock.patch.object(
            NovelPipeline, "voice_prompt_translator",
            staticmethod(lambda: self.FailingTranslator()),
        ):
            res = self.client.post(
                "/api/voices/update-prompt",
                json={
                    "series_id": "s1",
                    "voice_type": "narrator",
                    "voice_description": "เสียงหวานใส ร่าเริง",
                },
            )
        self.assertEqual(res.status_code, 200, res.text)
        k = self.knowledge()
        self.assertTrue(k["narrator_voice_description"])
        self.assertTrue(k["narrator_voice_control_prompt"], "keyword table should fill in")


class TestCorsPolicy(ExtensionApiTestCase):
    def test_arbitrary_origin_is_not_allowed(self):
        res = self.client.get(
            "/api/extension/health",
            headers={"Origin": "https://evil.example"},
        )
        self.assertNotIn("access-control-allow-origin", res.headers)

    def test_lookalike_origin_is_not_allowed(self):
        for origin in [
            "https://readtoon.com.evil.example",
            "http://readtoon.com",
            "https://evilreadtoon.com",
        ]:
            res = self.client.get("/api/extension/health", headers={"Origin": origin})
            self.assertNotIn(
                "access-control-allow-origin",
                res.headers,
                f"{origin} was allowed",
            )

    def test_readtoon_origin_is_allowed_without_credentials(self):
        res = self.client.get(
            "/api/extension/health",
            headers={"Origin": "https://readtoon.com"},
        )
        self.assertEqual(res.headers.get("access-control-allow-origin"), "https://readtoon.com")
        # Credentialed cross-origin reads must stay off.
        self.assertNotIn("access-control-allow-credentials", res.headers)

    def test_localhost_origin_is_allowed(self):
        res = self.client.get(
            "/api/extension/health",
            headers={"Origin": "http://127.0.0.1:8000"},
        )
        self.assertEqual(res.headers.get("access-control-allow-origin"), "http://127.0.0.1:8000")

    def test_preflight_from_evil_origin_is_rejected(self):
        res = self.client.options(
            "/api/extension/import",
            headers={
                "Origin": "https://evil.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        self.assertNotIn("access-control-allow-origin", res.headers)


if __name__ == "__main__":
    unittest.main()
