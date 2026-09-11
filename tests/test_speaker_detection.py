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


if __name__ == "__main__":
    unittest.main()
