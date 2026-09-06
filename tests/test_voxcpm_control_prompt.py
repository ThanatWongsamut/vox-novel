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


if __name__ == "__main__":
    unittest.main()
