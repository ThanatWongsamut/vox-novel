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

        # No double wrapping
        formatted_again = VoxCPM2TTS.format_designed_text(
            "(already wrapped)สวัสดีครับ", "30-year-old female narrator"
        )
        self.assertEqual(formatted_again, "(already wrapped)สวัสดีครับ")

        # Empty control
        plain = VoxCPM2TTS.format_designed_text("สวัสดีครับ", None)
        self.assertEqual(plain, "สวัสดีครับ")

    def test_prepare_text_for_tts(self):
        # 1. Plain unpunctuated Thai sentence gets period
        res1 = VoxCPM2TTS.prepare_text_for_tts("สมุดบันทึกที่หาไม่เจอเมื่อวานอาจจะโผล่มาก็ได้")
        self.assertEqual(res1, "สมุดบันทึกที่หาไม่เจอเมื่อวานอาจจะโผล่มาก็ได้.")

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


if __name__ == "__main__":
    unittest.main()
