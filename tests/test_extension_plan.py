"""GET /api/extension/plan: which chapters a batch import should load."""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from vox_novel.models.domain import Chapter, ChapterSummary, Novel, Paragraph
from vox_novel.web import app as web


def summary(no, locked=False, url=None):
    return ChapterSummary(
        id=str(no), book_id="s", title=f"ตอนที่ {no}", chapter_number=float(no),
        url=url or f"https://readtoon.com/content/s/{no}", is_locked=locked,
    )


def novel(*chapters, source="readtoon"):
    return Novel(id="s", title="Series", url="https://readtoon.com/content/s",
                 source=source, chapters=list(chapters))


class PlanTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        roots = [web.storage, web.knowledge_mgr, web.pipeline.storage, web.pipeline.knowledge]
        originals = [m.base_dir for m in roots]
        for m in roots:
            m.base_dir = self.tmp
        self.addCleanup(lambda: [setattr(m, "base_dir", o) for m, o in zip(roots, originals)])
        self.client = TestClient(web.web_app)
        # Never reach ReadToon from a test.
        self.scrape = mock.AsyncMock(side_effect=AssertionError("catalog fetched unexpectedly"))
        patcher = mock.patch.object(web.pipeline, "scrape_novel", self.scrape)
        patcher.start()
        self.addCleanup(patcher.stop)

    def store(self, n):
        web.storage.save_novel_metadata(n)

    def import_chapter(self, no):
        web.storage.save_chapter(Chapter(
            id=str(no), book_id="s", title=f"ตอนที่ {no}", chapter_number=float(no),
            url=f"https://readtoon.com/content/s/{no}", source_language="th",
            target_language="th",
            paragraphs=[Paragraph(index=1, text="x", translated_text="x")],
        ), True)

    def plan(self, **params):
        params.setdefault("series_id", "s")
        return self.client.get("/api/extension/plan", params=params)


class TestPlan(PlanTestCase):
    def test_lists_only_chapters_not_yet_imported(self):
        self.store(novel(summary(1), summary(2), summary(3)))
        self.import_chapter(2)
        body = self.plan().json()
        self.assertEqual([c["no"] for c in body["chapters"]], [1.0, 3.0])
        self.assertEqual((body["in_range"], body["already_imported"]), (3, 1))

    def test_respects_the_range(self):
        self.store(novel(*(summary(n) for n in range(1, 11))))
        body = self.plan(**{"from": 3, "to": 5}).json()
        self.assertEqual([c["no"] for c in body["chapters"]], [3.0, 4.0, 5.0])

    def test_a_single_chapter_is_a_range_of_one(self):
        self.store(novel(*(summary(n) for n in range(1, 11))))
        body = self.plan(**{"from": 7, "to": 7}).json()
        self.assertEqual([c["no"] for c in body["chapters"]], [7.0])

    def test_carries_the_paid_flag(self):
        self.store(novel(summary(1), summary(2, locked=True)))
        self.assertEqual([c["is_locked"] for c in self.plan().json()["chapters"]], [False, True])

    def test_a_non_readtoon_url_never_reaches_the_extension(self):
        self.store(novel(summary(1), summary(2, url="https://evil.example/content/s/2")))
        self.assertEqual([c["no"] for c in self.plan().json()["chapters"]], [1.0])

    def test_a_reversed_range_is_refused(self):
        self.store(novel(summary(1)))
        self.assertEqual(self.plan(**{"from": 5, "to": 2}).status_code, 400)

    def test_a_traversing_series_id_is_refused(self):
        self.assertEqual(self.plan(series_id="../evil").status_code, 400)

    def test_a_non_readtoon_series_is_refused(self):
        self.store(novel(summary(1), source="webnovel"))
        self.assertEqual(self.plan().status_code, 400)

    def test_a_refetch_plans_imported_chapters_too(self):
        self.store(novel(summary(1), summary(2)))
        self.import_chapter(1)
        body = self.plan(include_imported="true").json()
        self.assertEqual([(c["no"], c["imported"]) for c in body["chapters"]], [(1.0, True), (2.0, False)])
        self.assertEqual([c["no"] for c in self.plan().json()["chapters"]], [2.0], "default changed")


class TestCatalogRefresh(PlanTestCase):
    def test_an_unknown_series_fetches_its_catalog(self):
        self.scrape.side_effect = None
        self.scrape.return_value = novel(summary(1), summary(2))
        body = self.plan().json()
        self.scrape.assert_awaited_once_with("https://readtoon.com/content/s")
        self.assertEqual(len(body["chapters"]), 2)

    def test_a_range_past_the_known_end_refreshes(self):
        self.store(novel(summary(1), summary(2)))
        self.scrape.side_effect = None
        self.scrape.return_value = novel(summary(1), summary(2), summary(3))
        body = self.plan(**{"from": 3, "to": 3}).json()
        self.assertEqual([c["no"] for c in body["chapters"]], [3.0])

    def test_a_range_within_the_known_catalog_does_not_refresh(self):
        self.store(novel(summary(1), summary(2)))
        self.plan(**{"from": 1, "to": 2})
        self.scrape.assert_not_awaited()

    def test_a_failed_refresh_falls_back_to_the_stored_catalog(self):
        self.store(novel(summary(1), summary(2)))
        self.scrape.side_effect = RuntimeError("readtoon down")
        body = self.plan(refresh="true").json()
        self.assertEqual(len(body["chapters"]), 2)

    def test_a_failed_fetch_with_nothing_stored_is_reported(self):
        self.scrape.side_effect = RuntimeError("readtoon down")
        self.assertEqual(self.plan().status_code, 502)


class TestSeriesPage(PlanTestCase):
    """The batch panel is ReadToon-only and hidden until the extension appears."""

    def page(self):
        res = self.client.get("/series/s")
        self.assertEqual(res.status_code, 200, res.text)
        return res.text

    def test_a_readtoon_series_gets_the_panel(self):
        self.store(novel(summary(1), summary(2, locked=True)))
        html = self.page()
        self.assertIn('id="vox-batch"', html)
        self.assertIn('const SERIES_ID = "s";', html)
        # The old manual route stays for browsers without the extension.
        self.assertIn('class="vox-no-ext', html)

    def test_every_readtoon_fetch_goes_through_one_function(self):
        # Fetch Chapter, Re-Fetch and Fetch Next all route through the extension
        # when it is installed, and the server otherwise.
        self.store(novel(summary(1), summary(2)))
        self.import_chapter(1)
        html = self.page()
        self.assertNotIn("startTranslation('s'", html, "a ReadToon fetch bypasses the router")
        self.assertEqual(html.count('onclick="fetchReadtoonChapter(this)"'), 3)  # 2 rows + Fetch Next
        self.assertIn('data-no="1.0"', html)
        self.assertIn('data-imported="true"', html)
        self.assertIn('data-imported="false"', html)

    def test_a_title_with_quotes_cannot_break_out_of_the_button(self):
        self.store(novel(ChapterSummary(
            id="1", book_id="s", chapter_number=1.0, url="https://readtoon.com/content/s/1",
            title='ตอนที่ 1 "x" onmouseover="alert(1)',
        )))
        html = self.page()
        self.assertNotIn('onmouseover="alert(1)', html)
        self.assertIn("&#34;x&#34; onmouseover=&#34;alert(1)", html)

    def test_a_reload_shows_only_a_live_job(self):
        # Showing a finished job again made "Refresh chapter list" look broken:
        # the page reloaded straight back into the same "Finished" panel.
        self.store(novel(summary(1)))
        html = self.page()
        self.assertIn('job.status === "running" || job.status === "paused")) {\n                    render(job);', html)
        self.assertIn('onclick="voxBatch.finish()"', html)

    def test_the_page_never_requires_an_extension_newer_than_the_one_shipped(self):
        # Otherwise every user would be told to reload an extension that cannot
        # satisfy the page, and the batch panel would never work.
        import json, re
        self.store(novel(summary(1)))
        required = re.search(r'const REQUIRED_EXTENSION = "([0-9.]+)";', self.page()).group(1)
        manifest = json.loads((Path(__file__).parent.parent / "extension" / "manifest.json").read_text())
        as_tuple = lambda v: tuple(int(x) for x in v.split("."))
        self.assertLessEqual(as_tuple(required), as_tuple(manifest["version"]))

    def test_other_sources_do_not(self):
        self.store(novel(summary(1), source="webnovel"))
        html = self.page()
        self.assertNotIn('id="vox-batch"', html)
        self.assertNotIn("voxBatch", html)


if __name__ == "__main__":
    unittest.main()
