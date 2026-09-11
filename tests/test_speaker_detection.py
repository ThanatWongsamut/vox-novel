import asyncio
import unittest

from vox_novel.models.domain import Chapter, Paragraph
from vox_novel.models.series_knowledge import SeriesKnowledge
from vox_novel.speaker.chunking import format_chunk, make_chunks
from vox_novel.speaker.detector import annotate_chapter, extract_registry
from vox_novel.speaker.models import (
    CharacterDraft,
    ChunkAnnotation,
    RegistryDraft,
    Segment,
)


class FakeTranslator:
    """Returns queued structured responses, recording what it was asked."""

    name = "fake"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    async def structured(self, system_prompt, user_prompt, output_model, temperature=0.0):
        self.calls.append(user_prompt)
        if not self.responses:
            raise AssertionError("structured() called more times than responses queued")
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def chapter(n=4, cid="1"):
    return Chapter(
        id=cid,
        book_id="b",
        title="t",
        url="u",
        chapter_number=1.0,
        paragraphs=[
            Paragraph(id=f"p{i}", index=i, text=f"line {i}", translated_text=f"บรรทัด {i}")
            for i in range(1, n + 1)
        ],
    )


def knowledge_with(*names):
    k = SeriesKnowledge(series_id="s", target_language="th")
    for n in names:
        k.add_character(n, n)
    return k


def seg(i, type_="dialogue", speaker="ริโก้", confidence=1.0):
    return Segment(paragraph=i, type=type_, speaker=speaker, confidence=confidence)


class TestChunking(unittest.TestCase):
    def test_indices_are_the_caller_s_not_positional(self):
        # Paragraph.index is 1-based and may have gaps once empties are filtered.
        indexed = [(1, "a"), (2, "b"), (5, "c")]
        chunks = make_chunks(indexed, chunk_size=2, context_size=1)
        self.assertEqual([c.indices for c in chunks], [[1, 2], [5]])

    def test_context_precedes_the_target_and_is_not_annotated(self):
        indexed = [(i, f"p{i}") for i in range(1, 8)]
        chunks = make_chunks(indexed, chunk_size=3, context_size=2)
        self.assertEqual(chunks[1].indices, [4, 5, 6])
        self.assertEqual([i for i, _ in chunks[1].context], [2, 3])
        rendered = format_chunk(chunks[1])
        self.assertIn("do not repeat", rendered)
        self.assertLess(rendered.index("[2]"), rendered.index("[4]"))

    def test_a_rejected_chunk_size_is_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            make_chunks([(1, "a")], chunk_size=0)


class TestAttribution(unittest.TestCase):
    def run_annotate(self, chap, k, *responses, **kw):
        t = FakeTranslator(*responses)
        result = asyncio.run(annotate_chapter(chap, k, t, **kw))
        return result, t

    def test_speakers_and_types_land_on_the_paragraphs(self):
        chap, k = chapter(3), knowledge_with("ริโก้")
        self.run_annotate(
            chap, k,
            ChunkAnnotation(segments=[
                seg(1, "narration", None), seg(2, "dialogue", "ริโก้"), seg(3, "thought", "ริโก้"),
            ]),
        )
        got = [(p.index, p.speech_type, p.speaker) for p in chap.paragraphs]
        self.assertEqual(got, [(1, "narration", None), (2, "dialogue", "ริโก้"), (3, "thought", "ริโก้")])

    def test_narration_never_carries_a_speaker(self):
        chap, k = chapter(1), knowledge_with("ริโก้")
        self.run_annotate(chap, k, ChunkAnnotation(segments=[seg(1, "narration", "ริโก้")]))
        self.assertIsNone(chap.paragraphs[0].speaker)

    def test_an_invented_paragraph_number_is_ignored(self):
        chap, k = chapter(2), knowledge_with("ริโก้")
        self.run_annotate(
            chap, k,
            ChunkAnnotation(segments=[seg(1), seg(99)]),
            ChunkAnnotation(segments=[seg(2)]),      # repair pass for the real gap
        )
        self.assertEqual(chap.paragraphs[0].speaker, "ริโก้")
        self.assertEqual(chap.paragraphs[1].speaker, "ริโก้")

    def test_a_skipped_paragraph_is_repaired(self):
        chap, k = chapter(3), knowledge_with("ริโก้")
        result, t = self.run_annotate(
            chap, k,
            ChunkAnnotation(segments=[seg(1), seg(2)]),          # 3 skipped
            ChunkAnnotation(segments=[seg(3, "dialogue", "ริโก้")]),
        )
        self.assertEqual(len(t.calls), 2, "the repair pass did not run")
        self.assertEqual(chap.paragraphs[2].speaker, "ริโก้")
        self.assertEqual(result["missing"], [])

    def test_a_paragraph_never_annotated_falls_back_to_narration(self):
        chap, k = chapter(2), knowledge_with("ริโก้")
        result, _ = self.run_annotate(
            chap, k,
            ChunkAnnotation(segments=[seg(1)]),
            ChunkAnnotation(segments=[]),            # repair returns nothing
        )
        self.assertEqual(result["missing"], [2])
        self.assertEqual(chap.paragraphs[1].speech_type, "narration")
        self.assertIsNone(chap.paragraphs[1].speaker)

    def test_a_failed_chunk_does_not_abort_the_chapter(self):
        chap, k = chapter(2), knowledge_with("ริโก้")
        result, _ = self.run_annotate(
            chap, k, RuntimeError("502"), ChunkAnnotation(segments=[seg(1), seg(2)]),
        )
        self.assertEqual(result["annotated"], 2)

    def test_low_confidence_lines_are_surfaced(self):
        chap, k = chapter(2), knowledge_with("ริโก้")
        result, _ = self.run_annotate(
            chap, k,
            ChunkAnnotation(segments=[seg(1, confidence=0.4), seg(2, confidence=0.99)]),
        )
        self.assertEqual([x["paragraph"] for x in result["low_confidence"]], [1])


class TestAgreementGate(unittest.TestCase):
    """Two models must agree before a paragraph gets a character voice.

    A wrong voice is worse for a listener than no voice: the narrator reading
    everything is consistent, a voice switching mid-conversation is not. So an
    uncertain attribution falls back to narration, which sounds exactly as it
    does today.
    """

    def gated(self, primary, secondary, n=3):
        chap, k = chapter(n), knowledge_with("ริโก้", "พุดดิ้ง")
        result = asyncio.run(
            annotate_chapter(
                chap, k, FakeTranslator(ChunkAnnotation(segments=primary)),
                confirm_with=FakeTranslator(ChunkAnnotation(segments=secondary)),
            )
        )
        return chap, result

    def test_agreement_keeps_the_attribution(self):
        chap, result = self.gated([seg(1, "dialogue", "ริโก้")], [seg(1, "dialogue", "ริโก้")], n=1)
        self.assertEqual(chap.paragraphs[0].speaker, "ริโก้")
        self.assertEqual(result["disagreements"], [])

    def test_a_different_speaker_falls_back_to_narration(self):
        chap, result = self.gated([seg(1, "dialogue", "ริโก้")], [seg(1, "dialogue", "พุดดิ้ง")], n=1)
        self.assertIsNone(chap.paragraphs[0].speaker, "a disputed line kept a character voice")
        self.assertEqual(chap.paragraphs[0].speech_type, "narration")
        self.assertEqual(len(result["disagreements"]), 1)

    def test_a_different_type_also_disagrees(self):
        # Same speaker, but one calls it thought and the other dialogue.
        chap, _ = self.gated([seg(1, "thought", "ริโก้")], [seg(1, "dialogue", "ริโก้")], n=1)
        self.assertEqual(chap.paragraphs[0].speech_type, "narration")

    def test_silence_from_the_second_model_counts_as_disagreement(self):
        # Absence is not assent: the second model saying nothing is not a vote.
        chap, result = self.gated([seg(1, "dialogue", "ริโก้")], [], n=1)
        self.assertIsNone(chap.paragraphs[0].speaker)
        self.assertEqual(len(result["disagreements"]), 1)

    def test_agreement_on_narration_needs_no_special_case(self):
        chap, result = self.gated([seg(1, "narration", None)], [seg(1, "narration", None)], n=1)
        self.assertEqual(chap.paragraphs[0].speech_type, "narration")
        self.assertEqual(result["disagreements"], [])

    def test_only_the_disputed_paragraph_is_downgraded(self):
        chap, result = self.gated(
            [seg(1, "dialogue", "ริโก้"), seg(2, "dialogue", "ริโก้"), seg(3, "narration", None)],
            [seg(1, "dialogue", "ริโก้"), seg(2, "dialogue", "พุดดิ้ง"), seg(3, "narration", None)],
        )
        self.assertEqual(chap.paragraphs[0].speaker, "ริโก้", "an agreed line was dropped")
        self.assertIsNone(chap.paragraphs[1].speaker)
        self.assertEqual([d["paragraph"] for d in result["disagreements"]], [2])

    def test_an_alias_is_not_counted_as_disagreement(self):
        # Both models mean the same character under different spellings; the
        # canonicaliser runs before the comparison, so they agree.
        chap = chapter(1)
        k = SeriesKnowledge(series_id="s", target_language="th")
        k.add_character("ริโก้", "ริโก้", aliases=["ริโก้ราดกา"])
        result = asyncio.run(
            annotate_chapter(
                chap, k,
                FakeTranslator(ChunkAnnotation(segments=[seg(1, "dialogue", "ริโก้")])),
                confirm_with=FakeTranslator(
                    ChunkAnnotation(segments=[seg(1, "dialogue", "ริโก้ราดกา")])
                ),
            )
        )
        self.assertEqual(chap.paragraphs[0].speaker, "ริโก้")
        self.assertEqual(result["disagreements"], [], "an alias was treated as a different speaker")

    def test_without_a_second_model_nothing_is_gated(self):
        chap, k = chapter(1), knowledge_with("ริโก้")
        result = asyncio.run(
            annotate_chapter(chap, k, FakeTranslator(ChunkAnnotation(segments=[seg(1)])))
        )
        self.assertEqual(chap.paragraphs[0].speaker, "ริโก้")
        self.assertEqual(result["disagreements"], [])


class TestRegistryFailureModes(unittest.TestCase):
    """The three failures the prototype found only appear across many chapters."""

    def test_a_speaker_the_model_did_not_declare_is_registered(self):
        # Models name a new speaker in `speaker` far more often than they fill
        # new_characters; without this the cast list freezes.
        chap, k = chapter(1), knowledge_with("ริโก้")
        t = FakeTranslator(ChunkAnnotation(segments=[seg(1, "dialogue", "พุดดิ้ง")]))
        asyncio.run(annotate_chapter(chap, k, t))
        self.assertIsNotNone(k.find_character("พุดดิ้ง"), "the registry froze")

    def test_a_corrupted_spelling_folds_onto_one_character(self):
        # Models occasionally emit Thai names with stray CJK tokens. Left alone,
        # one character is split across several voices.
        chap, k = chapter(1), knowledge_with("เทเนเบรย์")
        t = FakeTranslator(ChunkAnnotation(segments=[seg(1, "dialogue", "เทเนเบร义")]))
        asyncio.run(annotate_chapter(chap, k, t))
        self.assertEqual(chap.paragraphs[0].speaker, "เทเนเบรย์")
        self.assertEqual(len(k.characters), 1, "a corrupted spelling became a second character")

    def test_unknown_is_not_registered_as_a_character(self):
        chap, k = chapter(1), knowledge_with("ริโก้")
        t = FakeTranslator(ChunkAnnotation(segments=[seg(1, "dialogue", "UNKNOWN")]))
        asyncio.run(annotate_chapter(chap, k, t))
        self.assertIsNone(k.find_character("UNKNOWN"))
        self.assertEqual(len(k.characters), 1)

    def test_the_prompt_does_not_grow_without_bound(self):
        k = SeriesKnowledge(series_id="s", target_language="th")
        for i in range(120):
            k.add_character(f"ตัวละคร{i}", f"ตัวละคร{i}")
        chap = chapter(1)
        t = FakeTranslator(ChunkAnnotation(segments=[seg(1, "narration", None)]))
        asyncio.run(annotate_chapter(chap, k, t))
        shown = t.calls[0].count('"name"')
        self.assertLessEqual(shown, 40, f"{shown} characters were sent to the model")

    def test_line_counts_accumulate_for_prompt_scoping(self):
        chap, k = chapter(2), knowledge_with("ริโก้")
        t = FakeTranslator(ChunkAnnotation(segments=[seg(1), seg(2)]))
        asyncio.run(annotate_chapter(chap, k, t))
        self.assertEqual(k.find_character("ริโก้").line_count, 2)
        self.assertEqual(k.find_character("ริโก้").last_seen_chapter, "1")


class TestExtraction(unittest.TestCase):
    def test_the_narrator_flag_and_speech_style_are_stored(self):
        # Both are absent from CharacterProfile before this feature, and the
        # benchmark attributes its 87% -> 100% jump to the narrator entry alone.
        k = SeriesKnowledge(series_id="s", target_language="th")
        t = FakeTranslator(
            RegistryDraft(
                characters=[
                    CharacterDraft(
                        name="ริโก้", gender="female", speech_style="ดิฉัน/ค่ะ, formal",
                        description="lady's maid", is_narrator=True,
                    )
                ],
                narration_note="first-person narration by ริโก้",
            )
        )
        added = asyncio.run(extract_registry(chapter(2), k, t))
        self.assertEqual(added, ["ริโก้"])
        character = k.find_character("ริโก้")
        self.assertTrue(character.is_narrator)
        self.assertEqual(character.speech_style, "ดิฉัน/ค่ะ, formal")
        self.assertEqual(k.narration_note, "first-person narration by ริโก้")

    def test_extraction_does_not_duplicate_a_known_character(self):
        k = knowledge_with("ริโก้")
        t = FakeTranslator(RegistryDraft(characters=[CharacterDraft(name="ริโก้")]))
        self.assertEqual(asyncio.run(extract_registry(chapter(1), k, t)), [])
        self.assertEqual(len(k.characters), 1)


class TestPipelineWiring(unittest.TestCase):
    """detect_speakers writes through to disk and is re-runnable."""

    def setUp(self):
        import shutil, tempfile
        from pathlib import Path
        from vox_novel.pipeline.manager import NovelPipeline
        from vox_novel.storage.file import StorageManager
        from vox_novel.storage.knowledge import KnowledgeManager

        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.storage = StorageManager(base_dir=self.tmp)
        self.km = KnowledgeManager(base_dir=self.tmp)
        self.pipeline = NovelPipeline(storage_manager=self.storage, knowledge_manager=self.km)

        chap = chapter(3)
        chap.target_language = "th"
        self.storage.save_chapter(chap)

        k = self.km.load_or_init("b", target_lang="th")
        k.add_character("ริโก้", "ริโก้")
        self.km.save(k)

    def detect(self, *responses, **kw):
        from unittest import mock

        t = FakeTranslator(*responses)
        with mock.patch.object(
            type(self.pipeline), "speaker_translator", staticmethod(lambda: t)
        ):
            return asyncio.run(
                self.pipeline.detect_speakers("b", "1", **kw)
            )

    def test_attributions_are_persisted_to_the_chapter(self):
        self.detect(
            ChunkAnnotation(segments=[
                seg(1, "narration", None), seg(2, "dialogue", "ริโก้"), seg(3, "narration", None),
            ])
        )
        saved = self.storage.get_chapter("b", "1")
        self.assertEqual(
            [(p.index, p.speech_type, p.speaker) for p in saved.paragraphs],
            [(1, "narration", None), (2, "dialogue", "ริโก้"), (3, "narration", None)],
        )

    def test_a_newly_seen_speaker_is_persisted_to_the_registry(self):
        self.detect(
            ChunkAnnotation(segments=[seg(1, "dialogue", "พุดดิ้ง"), seg(2, "narration", None),
                                      seg(3, "narration", None)])
        )
        k = self.km.load_or_init("b", target_lang="th")
        self.assertIsNotNone(k.find_character("พุดดิ้ง"))

    def test_rerunning_overwrites_rather_than_accumulating(self):
        self.detect(ChunkAnnotation(segments=[seg(i, "dialogue", "ริโก้") for i in (1, 2, 3)]))
        self.detect(ChunkAnnotation(segments=[seg(i, "narration", None) for i in (1, 2, 3)]))
        saved = self.storage.get_chapter("b", "1")
        self.assertEqual({p.speech_type for p in saved.paragraphs}, {"narration"})
        self.assertTrue(all(p.speaker is None for p in saved.paragraphs))

    def test_no_llm_configured_is_an_actionable_error(self):
        from unittest import mock

        with mock.patch.object(
            type(self.pipeline), "speaker_translator", staticmethod(lambda: None)
        ):
            with self.assertRaises(RuntimeError) as ctx:
                asyncio.run(self.pipeline.detect_speakers("b", "1"))
        self.assertIn("OPENROUTER_BASE_URL", str(ctx.exception))

    def test_a_missing_chapter_is_reported_clearly(self):
        from unittest import mock

        t = FakeTranslator()
        with mock.patch.object(
            type(self.pipeline), "speaker_translator", staticmethod(lambda: t)
        ):
            with self.assertRaises(ValueError):
                asyncio.run(self.pipeline.detect_speakers("b", "nope"))


class TestLocalServerConfig(unittest.TestCase):
    def test_base_url_comes_from_the_environment(self):
        import os
        from unittest import mock
        from vox_novel.translators.openrouter import OpenRouterTranslator

        with mock.patch.dict(os.environ, {"OPENROUTER_BASE_URL": "http://localhost:11434/v1"}):
            self.assertEqual(
                OpenRouterTranslator(api_key="x").base_url, "http://localhost:11434/v1"
            )

    def test_an_explicit_argument_still_wins(self):
        import os
        from unittest import mock
        from vox_novel.translators.openrouter import OpenRouterTranslator

        with mock.patch.dict(os.environ, {"OPENROUTER_BASE_URL": "http://env/v1"}):
            self.assertEqual(
                OpenRouterTranslator(api_key="x", base_url="http://explicit/v1").base_url,
                "http://explicit/v1",
            )

    def test_a_local_server_needs_no_api_key(self):
        import os
        from unittest import mock
        from vox_novel.pipeline.manager import NovelPipeline

        env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
        env["OPENROUTER_BASE_URL"] = "http://localhost:11434/v1"
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertIsNotNone(NovelPipeline.speaker_translator())


class TestReviewEndpoint(unittest.TestCase):
    """Correcting a line is how a human fixes attribution -- and how gold labels
    get made, since the corrections are the labels."""

    def setUp(self):
        import shutil, tempfile
        from pathlib import Path
        from fastapi.testclient import TestClient
        from vox_novel.web import app as web

        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self._roots = [web.storage, web.knowledge_mgr, web.pipeline.storage, web.pipeline.knowledge]
        self._orig = [m.base_dir for m in self._roots]
        for m in self._roots:
            m.base_dir = self.tmp
        self.addCleanup(lambda: [setattr(m, "base_dir", o) for m, o in zip(self._roots, self._orig)])

        self.web = web
        self.client = TestClient(web.web_app)

        chap = chapter(3)
        chap.paragraphs[1].speech_type = "dialogue"
        chap.paragraphs[1].speaker = "ริโก้"
        web.storage.save_chapter(chap)
        k = web.knowledge_mgr.load_or_init("b", target_lang="th")
        k.add_character("ริโก้", "ริโก้")
        k.add_character("พุดดิ้ง", "พุดดิ้ง")
        web.knowledge_mgr.save(k)

    def update(self, **kw):
        body = {"series_id": "b", "chapter_id": "1", "paragraph": 2,
                "speech_type": "dialogue", "speaker": "ริโก้"}
        body.update(kw)
        return self.client.post("/api/speakers/update", json=body)

    def saved(self):
        return self.web.storage.get_chapter("b", "1")

    def test_the_page_renders(self):
        res = self.client.get("/series/b/speakers/1")
        self.assertEqual(res.status_code, 200, res.text)
        self.assertIn("ริโก้", res.text)

    def test_a_correction_is_persisted_and_marked_verified(self):
        self.assertEqual(self.update(speaker="พุดดิ้ง").status_code, 200)
        p = self.saved().paragraphs[1]
        self.assertEqual(p.speaker, "พุดดิ้ง")
        self.assertTrue(p.speaker_verified, "a human correction was not recorded as verified")

    def test_switching_to_narration_clears_the_speaker(self):
        # Leaving a speaker on a narration line would have the two disagree.
        self.update(speech_type="narration", speaker="ริโก้")
        p = self.saved().paragraphs[1]
        self.assertEqual(p.speech_type, "narration")
        self.assertIsNone(p.speaker)

    def test_an_unattributed_line_is_allowed(self):
        self.assertEqual(self.update(speaker="").status_code, 200)
        self.assertIsNone(self.saved().paragraphs[1].speaker)

    def test_an_invalid_type_is_rejected(self):
        self.assertEqual(self.update(speech_type="shouting").status_code, 400)

    def test_a_missing_paragraph_is_rejected(self):
        self.assertEqual(self.update(paragraph=999).status_code, 404)

    def test_a_traversing_series_id_is_rejected(self):
        self.assertEqual(self.update(series_id="../evil").status_code, 400)

    def test_a_non_numeric_paragraph_is_rejected(self):
        self.assertEqual(self.update(paragraph="two").status_code, 400)


if __name__ == "__main__":
    unittest.main()
