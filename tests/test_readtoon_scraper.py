import unittest
from vox_novel.scrapers.readtoon import ReadtoonScraper
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


if __name__ == "__main__":
    unittest.main()
