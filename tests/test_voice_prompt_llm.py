import asyncio
import unittest

from vox_novel.tts.voxcpm import VoxCPM2TTS


class FakeTranslator:
    """Stands in for an LLM-backed translator."""

    def __init__(self, reply=None, error=None, delay=0.0):
        self.reply = reply
        self.error = error
        self.delay = delay
        self.calls = []

    async def complete(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.reply


def derive(description, translator=None, emotion=None):
    return asyncio.run(
        VoxCPM2TTS.derive_control_prompt(description, emotion=emotion, translator=translator)
    )


class TestLLMDerivedControlPrompt(unittest.TestCase):
    # A description the keyword table cannot express at all.
    UNTABLED = "เสียงแหบเล็กน้อยแบบคนเพิ่งตื่นนอน"

    def test_table_alone_cannot_express_this_description(self):
        # Establishes why the LLM path exists.
        self.assertEqual(VoxCPM2TTS.build_control_prompt(self.UNTABLED), "voice")

    def test_llm_result_is_used(self):
        t = FakeTranslator(reply="middle-aged voice, slightly hoarse, just woken up")
        self.assertEqual(derive(self.UNTABLED, t), "middle-aged voice, slightly hoarse, just woken up")
        self.assertEqual(len(t.calls), 1)

    def test_description_is_what_gets_sent(self):
        t = FakeTranslator(reply="calm male voice")
        derive("ชายเสียงนิ่ง", t)
        _, user_prompt = t.calls[0]
        self.assertIn("ชายเสียงนิ่ง", user_prompt)

    def test_no_translator_falls_back_to_table(self):
        self.assertEqual(derive("เสียงหวานใส ร่าเริง"), VoxCPM2TTS.build_control_prompt("เสียงหวานใส ร่าเริง"))

    def test_empty_description_never_calls_the_llm(self):
        t = FakeTranslator(reply="should not be used")
        derive("", t)
        derive(None, t)
        self.assertEqual(t.calls, [])


class TestLLMFallbacks(unittest.TestCase):
    """A bad completion must never be worse than no completion."""

    DESC = "เสียงหวานใส ร่าเริง"

    def expect_fallback(self, translator):
        self.assertEqual(derive(self.DESC, translator), VoxCPM2TTS.build_control_prompt(self.DESC))

    def test_non_ascii_reply_is_rejected(self):
        # The whole point: non-English control prompts get spoken aloud.
        self.expect_fallback(FakeTranslator(reply="เสียงหวานใส ร่าเริง"))

    def test_empty_reply_is_rejected(self):
        self.expect_fallback(FakeTranslator(reply="   "))

    def test_overlong_reply_is_rejected(self):
        self.expect_fallback(FakeTranslator(reply="word, " * 200))

    def test_refusal_is_rejected(self):
        self.expect_fallback(FakeTranslator(reply="Sorry, I cannot help with that request."))

    def test_reply_with_parentheses_is_rejected(self):
        # Parens would nest inside format_designed_text's own wrapper.
        self.expect_fallback(FakeTranslator(reply="female voice (young)"))

    def test_translator_error_is_survivable(self):
        self.expect_fallback(FakeTranslator(error=RuntimeError("502 upstream")))

    def test_unsupported_backend_is_survivable(self):
        self.expect_fallback(FakeTranslator(error=NotImplementedError("dummy")))

    def test_slow_llm_does_not_stall_synthesis(self):
        original = VoxCPM2TTS.VOICE_PROMPT_TIMEOUT_SECONDS
        VoxCPM2TTS.VOICE_PROMPT_TIMEOUT_SECONDS = 0.05
        try:
            self.expect_fallback(FakeTranslator(reply="too late", delay=0.5))
        finally:
            VoxCPM2TTS.VOICE_PROMPT_TIMEOUT_SECONDS = original


class TestReplyNormalization(unittest.TestCase):
    def test_wrapping_is_stripped(self):
        for raw, want in [
            ('"warm female voice"', "warm female voice"),
            ("(warm female voice)", "warm female voice"),
            ("`warm female voice`", "warm female voice"),
            ("  warm   female   voice  ", "warm female voice"),
        ]:
            self.assertEqual(VoxCPM2TTS._sanitize_control_prompt(raw), want, raw)

    def test_result_is_always_ascii_or_rejected(self):
        for raw in ["เสียงหวาน", "warm 声音", "warm female voice"]:
            out = VoxCPM2TTS._sanitize_control_prompt(raw)
            if out is not None:
                self.assertTrue(out.isascii(), out)


class TestCachedControlResolution(unittest.TestCase):
    def test_cached_value_skips_the_llm(self):
        t = FakeTranslator(reply="fresh result")
        got = asyncio.run(
            VoxCPM2TTS._resolve_cached_control(
                description="anything", cached="stored control", translator=t
            )
        )
        self.assertEqual(got, "stored control")
        self.assertEqual(t.calls, [], "a cached prompt must not cost an LLM call")

    def test_cache_miss_derives(self):
        t = FakeTranslator(reply="derived control")
        got = asyncio.run(
            VoxCPM2TTS._resolve_cached_control(description="x", cached=None, translator=t)
        )
        self.assertEqual(got, "derived control")
        self.assertEqual(len(t.calls), 1)


class TestKnowledgeCacheInvalidation(unittest.TestCase):
    def test_changing_the_description_clears_the_cached_control(self):
        from vox_novel.models.series_knowledge import SeriesKnowledge

        k = SeriesKnowledge(series_id="s", target_language="th")
        k.update_narrator_voice(voice_description="old", voice_control_prompt="old control")
        self.assertEqual(k.narrator_voice_control_prompt, "old control")

        k.update_narrator_voice(voice_description="new")
        self.assertIsNone(
            k.narrator_voice_control_prompt,
            "a stale control prompt would keep describing the previous voice",
        )

    def test_character_cache_is_invalidated_too(self):
        from vox_novel.models.series_knowledge import SeriesKnowledge

        k = SeriesKnowledge(series_id="s", target_language="th")
        k.add_character("Arthur", "อาเธอร์")
        k.update_character_voice("Arthur", voice_description="old", voice_control_prompt="old control")
        self.assertEqual(k.find_character("Arthur").voice_control_prompt, "old control")

        k.update_character_voice("Arthur", voice_description="new")
        self.assertIsNone(k.find_character("Arthur").voice_control_prompt)


if __name__ == "__main__":
    unittest.main()
