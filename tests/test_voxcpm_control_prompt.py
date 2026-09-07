import unittest
from vox_novel.tts.voxcpm import VoxCPM2TTS


class TestVoxCPMControlPrompt(unittest.TestCase):

    def test_user_female_narrator_prompt(self):
        prompt = "เสียงบรรยายผู้หญิง อายุย่างเข้า 30 ปี ให้ความรู้สึกเย็น ลึกลับ น่าหลงใหล นุ่มลึก ชัดถ้อยชัดคำ มีความเป็นคุณหนู กลุสตรี"
        result = VoxCPM2TTS.build_control_prompt(prompt)
        # Should detect 30-year-old, female, narrator
        self.assertIn("30-year-old", result)
        self.assertIn("female", result)
        self.assertIn("narrator", result)
        self.assertNotIn("male", result.replace("female", ""))
        # Should include relevant traits
        self.assertIn("noble lady", result)
        self.assertIn("gentlewoman", result)
        self.assertIn("mysterious", result)
        self.assertIn("soft and deep", result)

    def test_teen_female_character_prompt(self):
        prompt = "หญิงสาววัยรุ่น เสียงหวานใส ร่าเริง อ่อนหวาน น่าฟัง"
        result = VoxCPM2TTS.build_control_prompt(prompt)
        self.assertIn("female voice", result)
        self.assertIn("teenager", result)
        self.assertIn("sweet melodic", result)

    def test_young_male_character_prompt(self):
        prompt = "ชายหนุ่มวัย 20 เสียงห้าว มั่นใจ ชัดเจน เป็นมิตร"
        result = VoxCPM2TTS.build_control_prompt(prompt)
        self.assertIn("20-year-old male voice", result)
        self.assertIn("confident", result)
        self.assertIn("friendly", result)
        self.assertNotIn("female", result)
        # Verify no substring confusion between 'หนุ่ม' and 'นุ่ม'
        self.assertNotIn("soft, gentle", result)

    def test_neutral_narrator_prompt(self):
        prompt = "เสียงผู้บรรยายภาษาไทย ชัดเจน เป็นธรรมชาติ"
        result = VoxCPM2TTS.build_control_prompt(prompt)
        self.assertIn("narrator", result)
        self.assertIn("articulate, clear", result)
        self.assertIn("natural", result)
        # Verify 'บรรยาย' does not trigger 'ยาย' (female)
        self.assertNotIn("female", result)

    def test_english_pass_through(self):
        prompt = "calm, deep male voice, 40-year-old, authoritative narrator"
        result = VoxCPM2TTS.build_control_prompt(prompt)
        self.assertIn("40-year-old male narrator", result)
        self.assertIn("authoritative", result)
        self.assertIn("calm", result)

    def test_format_designed_text(self):
        formatted = VoxCPM2TTS.format_designed_text(
            "สวัสดีครับ", "30-year-old female narrator"
        )
        self.assertEqual(formatted, "(30-year-old female narrator)สวัสดีครับ")

        # No double wrapping when the same control is already applied
        formatted_again = VoxCPM2TTS.format_designed_text(
            "(30-year-old female narrator)สวัสดีครับ", "30-year-old female narrator"
        )
        self.assertEqual(formatted_again, "(30-year-old female narrator)สวัสดีครับ")

    def test_leading_parenthetical_prose_still_gets_its_control(self):
        # Prose can open with a parenthetical; that must not be mistaken for a
        # control prompt, or the line is synthesized with no voice design at all.
        formatted = VoxCPM2TTS.format_designed_text("(เสียงกระซิบ) เขาพูด", "female voice")
        self.assertTrue(formatted.startswith("(female voice)"), formatted)
        self.assertIn("(เสียงกระซิบ) เขาพูด", formatted)

        # Empty control
        plain = VoxCPM2TTS.format_designed_text("สวัสดีครับ", None)
        self.assertEqual(plain, "สวัสดีครับ")

    def test_prepare_text_for_tts(self):
        # 1. Plain unpunctuated Thai sentence gets a period and the trailing space
        res1 = VoxCPM2TTS.prepare_text_for_tts("สมุดบันทึกที่หาไม่เจอเมื่อวานอาจจะโผล่มาก็ได้")
        self.assertEqual(res1, "สมุดบันทึกที่หาไม่เจอเมื่อวานอาจจะโผล่มาก็ได้. ")

        # 2. Quoted dialogue without internal punctuation gets period before quote and trailing space
        res2 = VoxCPM2TTS.prepare_text_for_tts('"แล้วไง สิ่งนั้นมันทำอะไรบ้างล่ะ"')
        self.assertEqual(res2, '"แล้วไง สิ่งนั้นมันทำอะไรบ้างล่ะ." ')

        # 3. Quoted dialogue with internal punctuation gets trailing space
        res3 = VoxCPM2TTS.prepare_text_for_tts('"วงเวทอัญเชิญ?"')
        self.assertEqual(res3, '"วงเวทอัญเชิญ?" ')

        # 4. Quoted dialogue ending with ellipsis gets trailing space
        res4 = VoxCPM2TTS.prepare_text_for_tts('"...มันหยิบชุดของคุณหนูมาใส่แล้วออกไปเดิน..."')
        self.assertEqual(res4, '"...มันหยิบชุดของคุณหนูมาใส่แล้วออกไปเดิน..." ')

        # 5. Sentence with exclamation mark
        res5 = VoxCPM2TTS.prepare_text_for_tts("อ๊ะ! อยู่นี่ไง!")
        self.assertEqual(res5, "อ๊ะ! อยู่นี่ไง! ")

    def test_apply_tail_fadeout(self):
        import numpy as np
        # Create steady tone with high amplitude at the end
        sr = 48000
        audio = np.ones(sr, dtype=np.float32)
        self.assertEqual(audio[-1], 1.0)

        faded = VoxCPM2TTS._apply_tail_fadeout(audio, sample_rate=sr, fade_ms=20.0)
        # End should be 0.0 or near 0.0
        self.assertAlmostEqual(faded[-1], 0.0, places=5)
        # Beginning should be unaffected
        self.assertEqual(faded[0], 1.0)


class TestPrepareTextInvariants(unittest.TestCase):
    """The trailing space is part of the cutoff fix, so it must hold on every branch."""

    CASES = [
        "สมุดบันทึกที่หาไม่เจอเมื่อวานอาจจะโผล่มาก็ได้",
        "จบแล้ว.",
        "จริงหรือ?",
        "ไปเลย!",
        "เขาพูดว่า \u201cไปกันเถอะ\u201d",
        "รอสักครู่…",
        "plain english sentence",
        "  padded  ",
    ]

    def test_always_ends_with_space(self):
        for text in self.CASES:
            out = VoxCPM2TTS.prepare_text_for_tts(text)
            self.assertTrue(out.endswith(" "), f"{text!r} -> {out!r}")

    def test_always_has_terminal_punctuation_before_the_space(self):
        for text in self.CASES:
            out = VoxCPM2TTS.prepare_text_for_tts(text)
            stripped = out.rstrip()
            self.assertTrue(
                stripped.endswith((".", "!", "?", "\u2026", "\u2014", ":", ";", '"', "\u201d", "'", "\u2019")),
                f"{text!r} -> {out!r}",
            )

    def test_empty_input_is_left_alone(self):
        self.assertEqual(VoxCPM2TTS.prepare_text_for_tts(""), "")
        self.assertEqual(VoxCPM2TTS.prepare_text_for_tts("   "), "")


class TestControlPromptRobustness(unittest.TestCase):
    def test_proper_nouns_do_not_leak_into_the_prompt(self):
        result = VoxCPM2TTS.build_control_prompt(
            "เสียงนุ่ม from the ReadToon novel about Bangkok"
        )
        for noise in ("readtoon", "novel", "bangkok", "chapter"):
            self.assertNotIn(noise, result)
        self.assertIn("soft, gentle", result)

    def test_recognised_english_descriptors_still_pass_through(self):
        result = VoxCPM2TTS.build_control_prompt("a husky, authoritative male voice")
        self.assertIn("husky", result)
        self.assertIn("authoritative", result)

    def test_mixed_gender_description_resolves_to_one(self):
        result = VoxCPM2TTS.build_control_prompt("ชายหนุ่มคุยกับหญิงสาว")
        self.assertTrue(
            result.startswith("male") or result.startswith("female"), result
        )

    def test_min_len_scales_with_text_and_has_a_floor(self):
        self.assertEqual(VoxCPM2TTS._min_len_for(""), VoxCPM2TTS.MIN_LEN_FLOOR)
        self.assertGreater(VoxCPM2TTS._min_len_for("x" * 500), VoxCPM2TTS._min_len_for("x" * 50))


class TestControlPromptIsAsciiOnly(unittest.TestCase):
    """VoxCPM2 only interprets English control prompts.

    Verified by A/B: an English control steers the voice as asked, while a Thai
    control is ignored as an instruction and spoken aloud before the content --
    roughly 2.5x the audio length. So any non-Latin text reaching a control prompt
    becomes audible garbage at the start of every paragraph.
    """

    DESCRIPTIONS = [
        "เสียงบรรยายผู้หญิง อายุ 30 ปี เย็น ลึกลับ น่าหลงใหล",
        "ชายชราอายุ 70 ปี เสียงทุ้มต่ำมาก แหบ ห้าว",
        "เสียงแหบเล็กน้อยแบบคนเพิ่งตื่นนอน",   # no table entry matches
        "ตัวละครชื่อ อาเธอร์ พูดจาสุภาพ",       # contains a proper noun
        "混合中文描述",                          # a different non-Latin script
        "a husky, authoritative male voice",
        "",
        None,
    ]

    def test_never_emits_non_ascii(self):
        for desc in self.DESCRIPTIONS:
            out = VoxCPM2TTS.build_control_prompt(desc)
            self.assertTrue(out.isascii(), f"{desc!r} -> {out!r}")

    def test_never_emits_non_ascii_with_emotion(self):
        for desc in self.DESCRIPTIONS:
            out = VoxCPM2TTS.build_control_prompt(desc, emotion="โกรธมาก")
            self.assertTrue(out.isascii(), f"{desc!r} -> {out!r}")

    def test_always_returns_something_usable(self):
        for desc in self.DESCRIPTIONS:
            out = VoxCPM2TTS.build_control_prompt(desc)
            self.assertTrue(out.strip(), f"{desc!r} produced an empty control")

    def test_designed_text_keeps_thai_content_but_ascii_control(self):
        control = VoxCPM2TTS.build_control_prompt("ชายชราอายุ 70 ปี เสียงทุ้ม")
        designed = VoxCPM2TTS.format_designed_text("สวัสดีค่ะ", control)
        head = designed[: designed.index(")") + 1]
        self.assertTrue(head.isascii(), head)
        self.assertIn("สวัสดีค่ะ", designed)


class TestVoxCPMPatchGuard(unittest.TestCase):
    def test_patch_targets_a_pinned_version(self):
        # The patch is a copy of upstream's _inference; it must not be applied to a
        # version it was not derived from.
        self.assertRegex(VoxCPM2TTS.PATCHED_VOXCPM_VERSION, r"^\d+\.\d+\.\d+$")
        self.assertEqual(VoxCPM2TTS._EXPECTED_INFERENCE_PARAMS[0], "self")
        self.assertIn("min_len", VoxCPM2TTS._EXPECTED_INFERENCE_PARAMS)

    def test_patch_runs_under_inference_mode(self):
        import inspect

        src = inspect.getsource(VoxCPM2TTS._patch_voxcpm_inference)
        self.assertIn("@torch.inference_mode()", src)


if __name__ == "__main__":
    unittest.main()
