import unittest
from vox_novel.scrapers.readtoon import (
    ReadtoonScraper,
    classify_speech_type,
    split_html_paragraphs,
)
from vox_novel.scrapers.registry import registry


class TestReadtoonScraper(unittest.TestCase):

    def test_can_handle(self):
        self.assertTrue(ReadtoonScraper.can_handle("https://readtoon.com/content/novel-mypossessionbecameaghoststory"))
        self.assertTrue(ReadtoonScraper.can_handle("https://www.readtoon.com/content/novel-mypossessionbecameaghoststory/170"))
        self.assertFalse(ReadtoonScraper.can_handle("https://www.webnovel.com/book/some-book_1234567890"))

    def test_parse_slug(self):
        slug1 = ReadtoonScraper._parse_slug("https://readtoon.com/content/novel-mypossessionbecameaghoststory")
        self.assertEqual(slug1, "novel-mypossessionbecameaghoststory")

        slug2 = ReadtoonScraper._parse_slug("https://www.readtoon.com/content/novel-mypossessionbecameaghoststory/170")
        self.assertEqual(slug2, "novel-mypossessionbecameaghoststory")

        slug3 = ReadtoonScraper._parse_slug("https://readtoon.com/content/novel-mypossessionbecameaghoststory/170?tab=comments")
        self.assertEqual(slug3, "novel-mypossessionbecameaghoststory")

    def test_parse_chapter_no(self):
        no1 = ReadtoonScraper._parse_chapter_no("https://readtoon.com/content/novel-mypossessionbecameaghoststory/170")
        self.assertEqual(no1, 170)

        no2 = ReadtoonScraper._parse_chapter_no("https://readtoon.com/content/novel-mypossessionbecameaghoststory")
        self.assertIsNone(no2)

        no3 = ReadtoonScraper._parse_chapter_no("https://www.readtoon.com/content/novel-mypossessionbecameaghoststory/1?foo=bar")
        self.assertEqual(no3, 1)

    def test_registry_integration(self):
        scraper = registry.get_scraper_for_url("https://readtoon.com/content/novel-test")
        self.assertIsInstance(scraper, ReadtoonScraper)


class TestParagraphExtraction(unittest.TestCase):
    """The scraper and the Chrome extension must produce identical paragraphs."""

    def test_splits_on_br_and_paragraph_tags(self):
        html = "<p>one</p><p>two<br>three</p><p>four</p>"
        self.assertEqual(split_html_paragraphs(html), ["one", "two", "three", "four"])

    def test_drops_blank_segments(self):
        html = "<p>one</p><p>   </p><p></p><p>two</p>"
        self.assertEqual(split_html_paragraphs(html), ["one", "two"])

    def test_strips_inline_markup_and_unescapes_entities(self):
        html = "<p>a <em>bold</em> &amp; brave &quot;line&quot;</p>"
        self.assertEqual(split_html_paragraphs(html), ['a bold & brave "line"'])

    def test_handles_thai_content(self):
        html = "<p>บรรทัดหนึ่ง<br>บรรทัดสอง</p>"
        self.assertEqual(split_html_paragraphs(html), ["บรรทัดหนึ่ง", "บรรทัดสอง"])


class TestSpeechClassification(unittest.TestCase):
    def test_recognizes_all_dialogue_openers(self):
        for opener in ['"hi"', "\u201chi\u201d", "\u300chi\u300d", "\u300ehi\u300f"]:
            self.assertEqual(classify_speech_type(opener), "dialogue", opener)

    def test_plain_text_is_narration(self):
        self.assertEqual(classify_speech_type("เขาเดินไป"), "narration")
        self.assertEqual(classify_speech_type(""), "narration")


if __name__ == "__main__":
    unittest.main()
