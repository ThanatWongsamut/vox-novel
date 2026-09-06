import re
from typing import List, Optional
from urllib.parse import urljoin
import httpx
from bs4 import BeautifulSoup

from vox_novel.models.domain import Chapter, ChapterSummary, Novel, Paragraph
from vox_novel.scrapers.base import BaseScraper


def clean_webnovel_title(title: str) -> str:
    """Clean Webnovel titles from messy catalog artifacts."""
    # Remove leading repeated numbers like '2Chapter 2...' -> 'Chapter 2...'
    cleaned = re.sub(r"^\d+(?=Chapter\b|\bch\b)", "", title, flags=re.IGNORECASE)
    # Remove redundant prefix 'Chapter 100: Chapter 100.'
    cleaned = re.sub(r"^Chapter\s*\d+\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    # Remove trailing time tags like '*2 months ago', '2 months ago', etc.
    cleaned = re.sub(r"\s*\*?\s*\d+\s*(?:months?|days?|hours?|years?|weeks?)\s*ago\s*$", "", cleaned, flags=re.IGNORECASE)
    # Remove trailing asterisks
    cleaned = re.sub(r"\s*\*+\s*$", "", cleaned).strip()
    return cleaned


class WebnovelScraper(BaseScraper):
    """Scraper implementation for webnovel.com."""

    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    def __init__(self, headers: Optional[dict] = None, timeout: float = 20.0):
        self.headers = headers or self.DEFAULT_HEADERS
        self.timeout = timeout

    @classmethod
    def can_handle(cls, url: str) -> bool:
        return "webnovel.com" in url.lower()

    def _extract_book_id(self, url: str) -> str:
        match = re.search(r"_(\d{10,})", url)
        return match.group(1) if match else ""

    def _extract_chapter_id(self, url: str) -> str:
        match = re.search(r"_(\d{10,})$", url.rstrip("/"))
        return match.group(1) if match else ""

    async def _fetch_html(self, url: str) -> str:
        """Fetch URL using curl_cffi with Chrome TLS impersonation to bypass Cloudflare protection."""
        try:
            from curl_cffi import requests as curl_requests
            # Run in threadpool to keep async clean
            import asyncio
            def _get():
                return curl_requests.get(url, impersonate="chrome124", timeout=self.timeout)
            resp = await asyncio.to_thread(_get)
            if resp.status_code == 200:
                return resp.text
        except Exception:
            pass

        # Fallback to standard httpx
        async with httpx.AsyncClient(headers=self.headers, follow_redirects=True, timeout=self.timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text

    async def get_novel_info(self, url: str) -> Novel:
        book_id = self._extract_book_id(url)
        book_base_url = re.sub(r"/catalog/?$", "", url.split("?")[0])
        catalog_url = f"{book_base_url}/catalog"

        html_main = await self._fetch_html(book_base_url)
        soup_main = BeautifulSoup(html_main, "html.parser")

        title_el = soup_main.select_one("h1")
        title = title_el.get_text(strip=True) if title_el else "Unknown Novel"

        synopsis_el = soup_main.select_one(".j_synopsis")
        synopsis = synopsis_el.get_text("\n", strip=True) if synopsis_el else None

        author_el = soup_main.select_one("h2 span, h2 strong, .author-name, .c_sub_title")
        author = author_el.get_text(strip=True) if author_el else None

        cover_el = soup_main.select_one(".g_thumb img, .det-hd-pic img")
        cover_url = None
        if cover_el:
            cover_url = cover_el.get("src")
            if cover_url and cover_url.startswith("//"):
                cover_url = f"https:{cover_url}"

        chapters: List[ChapterSummary] = []
        try:
            html_cat = await self._fetch_html(catalog_url)
            soup_cat = BeautifulSoup(html_cat, "html.parser")
            seen_urls = set()
            for a in soup_cat.select("a[href*='/chapter-']"):
                raw_href = a.get("href", "")
                clean_href = urljoin("https://www.webnovel.com", raw_href.split("?")[0])
                if clean_href in seen_urls:
                    continue
                seen_urls.add(clean_href)

                raw_title = a.get_text(strip=True)
                clean_title = clean_webnovel_title(raw_title)

                cid = self._extract_chapter_id(clean_href)
                num_match = re.search(r"(?:chapter|\bch\.?)\s*(\d+(?:\.\d+)?)", clean_title, re.IGNORECASE)
                chap_num = float(num_match.group(1)) if num_match else None

                chapters.append(
                    ChapterSummary(
                        id=cid or clean_href,
                        book_id=book_id,
                        title=clean_title,
                        url=clean_href,
                        chapter_number=chap_num,
                        is_locked=False,
                    )
                )

            # Sort chapters by chapter_number ascending
            chapters.sort(key=lambda x: (x.chapter_number is None, x.chapter_number if x.chapter_number is not None else 0))
        except Exception:
            pass

        return Novel(
            id=book_id or book_base_url,
            title=title,
            url=book_base_url,
            author=author,
            cover_url=cover_url,
            synopsis=synopsis,
            source="webnovel",
            chapters=chapters,
        )

    async def get_chapter(self, url: str) -> Chapter:
        book_id = self._extract_book_id(url)
        chapter_id = self._extract_chapter_id(url)

        html = await self._fetch_html(url)
        soup = BeautifulSoup(html, "html.parser")

        title_el = soup.select_one(".cha-tit, h1, .chapter-title")
        raw_title = title_el.get_text(strip=True) if title_el else f"Chapter {chapter_id}"
        clean_title = clean_webnovel_title(raw_title)

        num_match = re.search(r"(?:chapter|\bch\.?)\s*(\d+(?:\.\d+)?)", clean_title, re.IGNORECASE)
        chap_num = float(num_match.group(1)) if num_match else None

        paragraphs: List[Paragraph] = []
        para_elements = soup.select("div.cha-paragraph")
        if not para_elements:
            para_elements = soup.select(".chapter_content p, .cha-content p")

        idx = 0
        for el in para_elements:
            p_tag = el.find("p") if el.name != "p" else el
            if p_tag:
                text = p_tag.get_text(strip=True)
                if text:
                    idx += 1
                    paragraphs.append(Paragraph(index=idx, text=text))

        return Chapter(
            id=chapter_id or url,
            book_id=book_id,
            title=clean_title,
            url=url,
            chapter_number=chap_num,
            paragraphs=paragraphs,
            source_language="en",
            metadata={"total_paragraphs": len(paragraphs)},
        )


async def search_webnovel(query: str, timeout: float = 15.0) -> List[dict]:
    """Search webnovel.com by keyword."""
    search_url = f"https://www.webnovel.com/search?keywords={httpx.URL(query).raw_path.decode() if hasattr(httpx.URL(query), 'raw_path') else query.replace(' ', '+')}"
    # Use clean query string
    from urllib.parse import quote_plus
    url = f"https://www.webnovel.com/search?keywords={quote_plus(query)}"

    async with httpx.AsyncClient(headers=WebnovelScraper.DEFAULT_HEADERS, timeout=timeout, follow_redirects=True) as client:
        resp = await client.get(url)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for li in soup.select("li"):
            h3 = li.select_one("h3")
            a = li.select_one("a[href*='/book/']")
            if h3 and a:
                title = h3.get_text(strip=True)
                raw_href = a.get("href", "")
                book_url = urljoin("https://www.webnovel.com", raw_href.split("?")[0])
                
                img = li.select_one("img")
                cover = None
                if img:
                    src = img.get("src") or img.get("data-src")
                    if src and src.startswith("//"):
                        cover = f"https:{src}"
                    elif src:
                        cover = src

                desc_el = li.select_one("p")
                desc = desc_el.get_text(" ", strip=True) if desc_el else ""

                m = re.search(r"_(\d{10,})", book_url)
                book_id = m.group(1) if m else ""

                results.append({
                    "id": book_id,
                    "title": title,
                    "url": book_url,
                    "cover_url": cover,
                    "synopsis": desc,
                })
        return results


async def browse_webnovel(category: str = "novel", timeout: float = 15.0) -> List[dict]:
    """Browse popular web novels."""
    url = f"https://www.webnovel.com/stories/{category}"
    async with httpx.AsyncClient(headers=WebnovelScraper.DEFAULT_HEADERS, timeout=timeout, follow_redirects=True) as client:
        resp = await client.get(url)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        seen = set()
        for li in soup.select("li"):
            h3 = li.select_one("h3")
            a = li.select_one("a[href*='/book/']")
            if h3 and a:
                title = h3.get_text(strip=True)
                raw_href = a.get("href", "")
                book_url = urljoin("https://www.webnovel.com", raw_href.split("?")[0])
                if book_url in seen:
                    continue
                seen.add(book_url)

                img = li.select_one("img")
                cover = None
                if img:
                    src = img.get("src") or img.get("data-src")
                    if src and src.startswith("//"):
                        cover = f"https:{src}"
                    elif src:
                        cover = src

                desc_el = li.select_one("p")
                desc = desc_el.get_text(" ", strip=True) if desc_el else ""

                m = re.search(r"_(\d{10,})", book_url)
                book_id = m.group(1) if m else ""

                results.append({
                    "id": book_id,
                    "title": title,
                    "url": book_url,
                    "cover_url": cover,
                    "synopsis": desc,
                })
        return results
