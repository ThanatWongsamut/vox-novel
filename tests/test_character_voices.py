import asyncio
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vox_novel.models.domain import Chapter, Paragraph
from vox_novel.models.series_knowledge import SeriesKnowledge, StoryPosition
from vox_novel.tts.voxcpm import VoxCPM2TTS


#: Thai voice descriptions used across these tests, and the English control
#: prompt a translator is expected to return for each. Control prompts must be
#: ASCII -- VoxCPM2 speaks a non-English one aloud instead of acting on it --
#: so a fake that echoes the Thai back is silently rejected and the keyword
#: table answers instead, which would make these assertions test the table.
VOICES = {
    "หญิงสาว เสียงสูง": "young woman, bright and high pitched",
    "ชายหนุ่ม เสียงต่ำ": "young man, low and level",
    "เสียงบรรยายผู้ชาย": "male narrator, measured",
}


class Translator:
    """Returns the expected English prompt, recording every call.

    The call count is what proves a voice is derived once per chapter rather
    than once per paragraph.
    """

    def __init__(self):
        self.calls = []

    async def complete(self, system_prompt, user_prompt):
        self.calls.append(user_prompt)
        for thai, english in VOICES.items():
            if thai in user_prompt:
                return english
        raise AssertionError(f"no voice in prompt: {user_prompt!r}")


class CharacterVoiceTestCase(unittest.TestCase):
    """Which voice a paragraph is heard in, and where its timbre comes from.

    The reference clip decides timbre -- the engine re-rolls a voice per
    utterance otherwise -- so an assertion about a description is worthless
    unless it also pins the clip that was cloned from.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "chapters"
        self.voices = self.tmp / "voices"
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def engine(self):
        engine = VoxCPM2TTS(api_url=None)
        engine.api_url = None
        patcher = mock.patch.object(engine, "_get_local_model", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)
        return engine

    def knowledge(self):
        k = SeriesKnowledge(series_id="s", target_language="th")
        k.narrator_voice_description = "เสียงบรรยายผู้ชาย"
        return k

    def chapter(self, rows, number=169.0):
        """rows: (speaker, speech_type) per paragraph, 1-based."""
        return Chapter(
            id="1", book_id="b", title="t", url="u", chapter_number=number,
            paragraphs=[
                Paragraph(
                    id=f"p{i}", index=i, text=f"line {i}", translated_text=f"line {i}",
                    speaker=speaker, speech_type=speech_type,
                )
                for i, (speaker, speech_type) in enumerate(rows, 1)
            ],
        )

    def run_chapter(self, engine, chap, knowledge, translator=None):
        """Return {paragraph index: (control_prompt, reference clip name)}."""
        seen = {}
        original = engine.synthesize

        async def spy(**kwargs):
            name = Path(kwargs["output_file"]).name
            result = await original(**kwargs)
            if name.startswith("para_"):
                ref = kwargs.get("reference_audio")
                seen[int(name.rsplit("_", 1)[1].split(".")[0])] = (
                    kwargs.get("control_prompt"),
                    Path(ref).name if ref else None,
                )
            return result

        with mock.patch.object(engine, "synthesize", spy):
            asyncio.run(engine.synthesize_chapter(
                chapter=chap, output_dir=self.out,
                knowledge=knowledge, translator=translator,
            ))
        return seen


class TestADescribedCharacterGetsTheirOwnAnchor(CharacterVoiceTestCase):
    """The bug this fixes: a character with a description but no clip cloned
    from the narrator anchor, so they were heard in the narrator's timbre and
    the description changed almost nothing."""

    def setUp(self):
        super().setUp()
        self.k = self.knowledge()
        self.k.add_character("ริโก้", "ริโก้")
        self.k.find_character("ริโก้").voice_description = "หญิงสาว เสียงสูง"

    def test_the_character_does_not_clone_from_the_narrator(self):
        seen = self.run_chapter(
            self.engine(),
            self.chapter([("ริโก้", "dialogue"), (None, "narration")]),
            self.k, Translator(),
        )
        character_ref, narrator_ref = seen[1][1], seen[2][1]
        self.assertIsNotNone(character_ref)
        self.assertNotEqual(
            character_ref, narrator_ref,
            "the character was cloned from the narrator anchor, so the "
            "description reached nothing",
        )
        self.assertTrue(character_ref.startswith("ริโก้_ref."), character_ref)

    def test_the_anchor_carries_the_character_control_prompt(self):
        self.run_chapter(
            self.engine(), self.chapter([("ริโก้", "dialogue")]), self.k, Translator()
        )
        anchors = [f.name for f in self.voices.glob("ริโก้_ref.*.wav")]
        self.assertEqual(len(anchors), 1, anchors)

    def test_an_undescribed_character_still_uses_the_narrator(self):
        # No description means no voice was designed for them, and inventing one
        # would be worse than the consistent narrator.
        self.k.add_character("เดซี่", "เดซี่")
        seen = self.run_chapter(
            self.engine(),
            self.chapter([("เดซี่", "dialogue"), (None, "narration")]),
            self.k, Translator(),
        )
        self.assertEqual(seen[1][1], seen[2][1])

    def test_an_uploaded_clip_outranks_the_generated_anchor(self):
        self.voices.mkdir(parents=True, exist_ok=True)
        uploaded = self.voices / "ริโก้_ref.wav"
        uploaded.write_bytes(b"RIFF....WAVE")
        seen = self.run_chapter(
            self.engine(), self.chapter([("ริโก้", "dialogue")]), self.k, Translator()
        )
        self.assertEqual(seen[1][1], "ริโก้_ref.wav")

    def test_a_legacy_narrator_speaker_uses_the_narrator_voice(self):
        self.k.add_character("narrator", "narrator")
        self.k.find_character("narrator").voice_description = "หญิงสาว เสียงสูง"
        seen = self.run_chapter(
            self.engine(), self.chapter([("narrator", "narration")]), self.k, Translator()
        )
        self.assertEqual(seen[1][0], VOICES["เสียงบรรยายผู้ชาย"])


class TestASwappedBodySplitsSpeechFromThought(CharacterVoiceTestCase):
    """A body holding someone else's mind speaks as the body and thinks as the
    occupant. Occupancy is recorded on the mind, even when the prose names the
    visible body."""

    SWAP_AT = 100.0

    def setUp(self):
        super().setUp()
        self.k = self.knowledge()
        self.k.add_character("อาซาเซล", "อาซาเซล")
        self.k.add_character("เยเรเมีย", "เยเรเมีย")
        self.k.find_character("อาซาเซล").voice_description = "ชายหนุ่ม เสียงต่ำ"
        self.k.find_character("เยเรเมีย").voice_description = "หญิงสาว เสียงสูง"
        self.k.set_inhabiting("เยเรเมีย", "อาซาเซล", self.SWAP_AT)

    def rows(self):
        return [("เยเรเมีย", "dialogue"), ("เยเรเมีย", "thought")]

    def body(self):
        return VOICES["ชายหนุ่ม เสียงต่ำ"]

    def mind(self):
        return VOICES["หญิงสาว เสียงสูง"]

    def test_after_the_swap_speech_is_the_body_and_thought_is_the_occupant(self):
        seen = self.run_chapter(
            self.engine(), self.chapter(self.rows(), number=169.0), self.k, Translator()
        )
        self.assertEqual(seen[1][0], self.body())
        self.assertEqual(seen[2][0], self.mind())
        self.assertNotEqual(
            seen[1][1], seen[2][1], "both voices cloned from the same anchor"
        )

    def test_before_the_swap_the_body_is_simply_itself(self):
        # The case the previous design got wrong: this character existed and
        # spoke for a hundred chapters before anyone occupied him, and those
        # chapters need no configuration at all.
        seen = self.run_chapter(
            self.engine(), self.chapter([("อาซาเซล", "dialogue"), ("อาซาเซล", "thought")], number=50.0), self.k, Translator()
        )
        self.assertEqual(seen[1][0], self.body())
        self.assertEqual(seen[2][0], self.body())
        self.assertEqual(seen[1][1], seen[2][1])

    def test_the_body_name_is_not_an_alias_for_the_occupants_mind(self):
        """An original owner may still think; a body name cannot stand for both minds."""
        rows = [
            ("เยเรเมีย", "dialogue"), ("เยเรเมีย", "thought"),
            ("อาซาเซล", "dialogue"), ("อาซาเซล", "thought"),
        ]
        seen = self.run_chapter(
            self.engine(), self.chapter(rows, number=169.0), self.k, Translator()
        )
        self.assertEqual(seen[1][0], self.body(), "her speech should sound like the body")
        self.assertEqual(seen[2][0], self.mind(), "her thoughts should stay her own")
        self.assertEqual(seen[3][0], VOICES["เสียงบรรยายผู้ชาย"], "displaced mind has no known speaking body")
        self.assertEqual(seen[4][0], self.body(), "the original mind keeps its own thought voice")

    def test_a_character_outside_the_swap_is_untouched(self):
        self.k.add_character("คันน่า", "คันน่า")
        self.k.find_character("คันน่า").voice_description = "เสียงบรรยายผู้ชาย"
        seen = self.run_chapter(
            self.engine(),
            self.chapter([("คันน่า", "dialogue"), ("คันน่า", "thought")], number=169.0),
            self.k, Translator(),
        )
        self.assertEqual(seen[1][0], seen[2][0])

    def test_a_chapter_with_no_number_requires_voice_review(self):
        # Nothing places it relative to the swap, so guessing risks the wrong voice.
        seen = self.run_chapter(
            self.engine(), self.chapter(self.rows(), number=None), self.k, Translator()
        )
        self.assertEqual(seen[1][0], VOICES["เสียงบรรยายผู้ชาย"])
        self.assertEqual(seen[2][0], self.mind())

    def test_a_start_chapter_is_required(self):
        with self.assertRaises(ValueError):
            self.k.set_inhabiting("เยเรเมีย", "อาซาเซล", None)

    def test_each_voice_is_derived_once_and_cached_on_its_own_profile(self):
        translator = Translator()
        self.run_chapter(
            self.engine(), self.chapter(self.rows() * 3, number=169.0), self.k, translator
        )
        self.assertEqual(
            self.k.find_character("อาซาเซล").voice_control_prompt, self.body()
        )
        self.assertEqual(
            self.k.find_character("เยเรเมีย").voice_control_prompt, self.mind()
        )
        # Narrator, body, occupant -- one call each, not one per paragraph.
        self.assertEqual(len(translator.calls), 3, translator.calls)

    def test_tts_switches_at_paragraph_boundaries_and_back(self):
        self.k.set_inhabiting("เยเรเมีย", None, None)
        self.k.add_occupancy("เยเรเมีย", "อาซาเซล", StoryPosition(chapter_number=169, paragraph_index=2),
                             StoryPosition(chapter_number=169, paragraph_index=4))
        chap = self.chapter([("เยเรเมีย", "dialogue")] * 4, number=169)
        seen = self.run_chapter(self.engine(), chap, self.k, Translator())
        self.assertEqual([seen[i][0] for i in range(1, 5)],
                         [self.mind(), self.body(), self.body(), self.mind()])

    def test_voice_override_is_used_for_an_exceptional_line(self):
        chap = self.chapter([("เยเรเมีย", "dialogue")], number=169)
        chap.paragraphs[0].voice_override = "เยเรเมีย"
        seen = self.run_chapter(self.engine(), chap, self.k, Translator())
        self.assertEqual(seen[1][0], self.mind())

    def test_a_body_the_registry_does_not_know_is_refused(self):
        # The mechanism is a redirect to that character's voice, so a name
        # nothing resolves to would silently do nothing at synthesis.
        with self.assertRaises(ValueError):
            self.k.set_inhabiting("เยเรเมีย", "ไม่มีใคร", 100.0)

    def test_a_character_cannot_inhabit_themselves(self):
        with self.assertRaises(ValueError):
            self.k.set_inhabiting("เยเรเมีย", "เยเรเมีย", 100.0)

    def test_clearing_occupancy_restores_the_body(self):
        self.k.set_inhabiting("เยเรเมีย", None, None)
        seen = self.run_chapter(
            self.engine(), self.chapter(self.rows(), number=169.0), self.k, Translator()
        )
        self.assertEqual(seen[1][0], self.mind())
        self.assertEqual(seen[2][0], self.mind())


class TestTheGlossaryEditsTheSwap(unittest.TestCase):
    """Setting it is the only way the plot reaches synthesis, so the endpoint
    has to refuse the shapes that would silently re-voice a whole series."""

    def setUp(self):
        from fastapi.testclient import TestClient
        from vox_novel.web import app as web

        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        roots = [web.storage, web.knowledge_mgr, web.pipeline.storage, web.pipeline.knowledge]
        originals = [m.base_dir for m in roots]
        for m in roots:
            m.base_dir = self.tmp
        self.addCleanup(
            lambda: [setattr(m, "base_dir", o) for m, o in zip(roots, originals)]
        )

        self.web = web
        self.client = TestClient(web.web_app)
        k = web.knowledge_mgr.load_or_init("b", target_lang="th")
        k.add_character("อาซาเซล", "อาซาเซล")
        k.add_character("เยเรเมีย", "เยเรเมีย")
        k.find_character("อาซาเซล").voice_description = "ชายหนุ่ม"
        k.find_character("เยเรเมีย").voice_description = "หญิงสาว"
        web.knowledge_mgr.save(k)

    def save(self, **kw):
        body = {
            "series_id": "b", "target_lang": "th", "character_name": "เยเรเมีย",
            "inhabiting": "อาซาเซล", "from_chapter": 100,
        }
        body.update(kw)
        return self.client.post("/api/voices/inhabiting", json=body)

    def reload(self):
        return self.web.knowledge_mgr.load_or_init("b", target_lang="th").find_character("เยเรเมีย")

    def test_it_is_stored_without_touching_the_characters_own_voice(self):
        self.assertEqual(self.save().status_code, 200)
        char = self.reload()
        self.assertEqual(char.inhabiting, "อาซาเซล")
        self.assertEqual(char.inhabiting_from_chapter, 100.0)
        self.assertEqual(char.voice_description, "หญิงสาว", "her own voice was overwritten")

    def test_a_swap_without_a_start_chapter_is_refused(self):
        # It would apply to every chapter, including those set before the swap.
        self.assertEqual(self.save(from_chapter=None).status_code, 400)
        self.assertIsNone(self.reload().inhabiting)

    def test_clearing_it_does_not_need_a_start_chapter(self):
        self.save()
        self.assertEqual(self.save(inhabiting="", from_chapter=None).status_code, 200)
        self.assertIsNone(self.reload().inhabiting)
        self.assertIsNone(self.reload().inhabiting_from_chapter)

    def test_a_non_numeric_start_chapter_is_refused(self):
        self.assertEqual(self.save(from_chapter="soon").status_code, 400)

    def test_an_unknown_body_is_refused(self):
        self.assertEqual(self.save(inhabiting="ไม่มีใคร").status_code, 400)

    def test_a_character_cannot_inhabit_themselves(self):
        self.assertEqual(self.save(inhabiting="เยเรเมีย").status_code, 400)

    def test_an_unknown_character_is_refused(self):
        self.assertEqual(self.save(character_name="ไม่มีใคร").status_code, 404)

    def test_a_traversing_series_id_is_refused(self):
        self.assertEqual(self.save(series_id="../evil").status_code, 400)

    def test_the_glossary_page_shows_it(self):
        self.save()
        res = self.client.get("/series/b/glossary")
        self.assertEqual(res.status_code, 200, res.text)
        self.assertIn("Body occupancy timeline", res.text)
        self.assertIn("เยเรเมีย", res.text)


if __name__ == "__main__":
    unittest.main()
