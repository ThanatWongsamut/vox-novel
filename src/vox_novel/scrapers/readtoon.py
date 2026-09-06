import asyncio
import json
import os
import re
from typing import List, Optional
import httpx
from bs4 import BeautifulSoup

from vox_novel.models.domain import Chapter, ChapterSummary, Novel, Paragraph
from vox_novel.scrapers.base import BaseScraper


class ReadtoonScraper(BaseScraper):
    """Scraper implementation for readtoon.com novels (already translated to Thai)."""

    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/145.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "th,en-US;q=0.9,en;q=0.8",
        "Referer": "https://readtoon.com/",
    }

    def __init__(self, auth_token: Optional[str] = None, timeout: float = 25.0):
        self.auth_token = auth_token or os.getenv("READTOON_AUTH_TOKEN", "")
        self.timeout = timeout

    @classmethod
    def can_handle(cls, url: str) -> bool:
        return "readtoon.com" in url.lower()

    @staticmethod
    def _parse_slug(url: str) -> str:
        """Extract content slug from Readtoon URL."""
        clean = url.split("?")[0].rstrip("/")
        # Match /content/{slug} or /content/{slug}/{chapter_no}
        m = re.search(r"/content/([^/]+?)(?:/\d+)?$", clean)
        if m:
            return m.group(1)
        # Fallback to last path component
        parts = [p for p in clean.split("/") if p]
        return parts[-1] if parts else ""

    @staticmethod
    def _parse_chapter_no(url: str) -> Optional[int]:
        """Extract chapter number from Readtoon chapter URL."""
        clean = url.split("?")[0].rstrip("/")
        m = re.search(r"/content/[^/]+/(\d+)$", clean)
        return int(m.group(1)) if m else None

    @staticmethod
    def _extract_flight_data(html: str) -> str:
        """Extract and concatenate Next.js flight data pushes."""
        pushes = re.findall(r"self\.__next_f\.push\(\[(\d+),\s*\"(.*?)\"\]\)", html, re.DOTALL)
        flight_chunks = []
        for _, raw_chunk in pushes:
            try:
                decoded = json.loads('"' + raw_chunk + '"')
                flight_chunks.append(decoded)
            except Exception:
                flight_chunks.append(raw_chunk)
        return "".join(flight_chunks)

    async def _fetch_html(self, url: str) -> str:
        """Fetch HTML page with curl_cffi or httpx."""
        try:
            from curl_cffi import requests as curl_requests
            def _get():
                return curl_requests.get(url, headers=self.DEFAULT_HEADERS, impersonate="chrome124", timeout=self.timeout)
            resp = await asyncio.to_thread(_get)
            if resp.status_code == 200:
                return resp.text
        except Exception:
            pass

        async with httpx.AsyncClient(headers=self.DEFAULT_HEADERS, follow_redirects=True, timeout=self.timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text

    async def get_novel_info(self, url: str) -> Novel:
        slug = self._parse_slug(url)
        novel_url = f"https://readtoon.com/content/{slug}"

        html = await self._fetch_html(novel_url)
        flight_data = self._extract_flight_data(html)

        # 1. Title
        title_m = re.search(r'\"title\":\"([^\"]+)\"', flight_data)
        if title_m:
            title = title_m.group(1)
        else:
            soup = BeautifulSoup(html, "html.parser")
            h1 = soup.select_one("h1")
            title = h1.get_text(strip=True) if h1 else f"Readtoon Novel ({slug})"

        # 2. Synopsis
        syn_m = re.search(r'\"shortDescription\":\"([^\"]+)\"', flight_data)
        if not syn_m:
            syn_m = re.search(r'\"description\":\"([^\"]+)\"', flight_data)
        synopsis = syn_m.group(1).replace("\\n", "\n") if syn_m else None

        # 3. Author / Uploader
        author_m = re.search(r'\"nickname\":\"([^\"]+)\"', flight_data)
        if not author_m:
            author_m = re.search(r'\"username\":\"([^\"]+)\"', flight_data)
        author = author_m.group(1) if author_m else None

        # 4. Cover URL
        thumb_m = re.search(r'\"thumbnailUrl\":\"([^\"]+)\"', flight_data)
        cover_url = None
        if thumb_m:
            raw_thumb = thumb_m.group(1)
            if raw_thumb.startswith("http"):
                cover_url = raw_thumb
            else:
                cover_url = f"https://w.nobuild.pro/{raw_thumb.lstrip('/')}"

        # 5. Chapters
        chapters_raw = []
        for m in re.finditer(r'\{\"no\":(\d+),\"name\":\"([^\"]+)\"(?:,\"isPaid\":(true|false))?', flight_data):
            no = int(m.group(1))
            name = m.group(2)
            is_paid = (m.group(3) == "true") if m.group(3) else False
            chap_title = name if (name.startswith(f"ตอนที่ {no}") or name.startswith("ตอนที่")) else f"ตอนที่ {no}: {name}"
            chapters_raw.append({
                "no": no,
                "name": chap_title,
                "is_paid": is_paid,
                "url": f"https://readtoon.com/content/{slug}/{no}",
                "id": str(no),
            })

        seen_no = set()
        chapter_summaries: List[ChapterSummary] = []
        # Deduplicate keeping earliest occurrence
        for ch in sorted(chapters_raw, key=lambda x: x["no"]):
            if ch["no"] in seen_no:
                continue
            seen_no.add(ch["no"])
            chapter_summaries.append(
                ChapterSummary(
                    id=ch["id"],
                    book_id=slug,
                    title=ch["name"],
                    url=ch["url"],
                    chapter_number=float(ch["no"]),
                    is_locked=ch["is_paid"],
                )
            )

        return Novel(
            id=slug,
            title=title,
            url=novel_url,
            author=author,
            cover_url=cover_url,
            synopsis=synopsis,
            source="readtoon",
            chapters=chapter_summaries,
            metadata={"slug": slug, "total_chapters": len(chapter_summaries)},
        )

    async def get_chapter(self, url: str) -> Chapter:
        from playwright.async_api import async_playwright

        slug = self._parse_slug(url)
        chapter_no = self._parse_chapter_no(url)
        if chapter_no is None:
            raise ValueError(f"Could not extract chapter number from Readtoon URL: {url}")

        chapter_url = f"https://readtoon.com/content/{slug}/{chapter_no}"

        async with async_playwright() as p:
            # Prefer system Chrome if available, fallback to bundled chromium
            browser = None
            for launch_opts in [{"channel": "chrome", "headless": True}, {"headless": True}]:
                try:
                    browser = await p.chromium.launch(**launch_opts)
                    break
                except Exception:
                    continue

            if not browser:
                raise RuntimeError("Failed to launch Playwright browser (neither Google Chrome nor Chromium available).")

            try:
                context = await browser.new_context(
                    user_agent=self.DEFAULT_HEADERS["User-Agent"],
                    viewport={"width": 1280, "height": 800},
                )
                page = await context.new_page()

                # Set AuthToken/cookies in context and localStorage if available
                auth_token = (self.auth_token or os.getenv("READTOON_AUTH_TOKEN", "")).strip()
                device_token = os.getenv("READTOON_DEVICE_TOKEN", "").strip()
                cookie_str = os.getenv("READTOON_COOKIE", "").strip()

                cookies_to_add = []
                # If a full cookie string is provided (e.g. from curl)
                if ";" in auth_token or "=" in auth_token:
                    cookie_str = auth_token
                    auth_token = ""

                if cookie_str:
                    for part in cookie_str.split(";"):
                        part = part.strip()
                        if "=" in part:
                            c_name, c_val = part.split("=", 1)
                            c_name = c_name.strip()
                            c_val = c_val.strip().strip("'\"")
                            cookies_to_add.append({
                                "name": c_name,
                                "value": c_val,
                                "domain": ".readtoon.com",
                                "path": "/",
                            })
                            if c_name == "auth_token":
                                auth_token = c_val

                if auth_token:
                    cookies_to_add.extend([
                        {"name": "auth_token", "value": auth_token, "domain": ".readtoon.com", "path": "/"},
                        {"name": "auth-token", "value": auth_token, "domain": ".readtoon.com", "path": "/"},
                        {"name": "token", "value": auth_token, "domain": ".readtoon.com", "path": "/"},
                        {"name": "_$AuthToken", "value": auth_token, "domain": ".readtoon.com", "path": "/"},
                    ])

                if device_token:
                    cookies_to_add.append(
                        {"name": "device_token", "value": device_token, "domain": ".readtoon.com", "path": "/"}
                    )

                if cookies_to_add:
                    try:
                        # Deduplicate by cookie name
                        seen_c = set()
                        unique_cookies = []
                        for c in cookies_to_add:
                            if c["name"] not in seen_c:
                                seen_c.add(c["name"])
                                unique_cookies.append(c)
                        await context.add_cookies(unique_cookies)

                        if auth_token:
                            await page.goto("https://readtoon.com", wait_until="domcontentloaded", timeout=10000)
                            await page.evaluate(f"""
                                localStorage.setItem('_$AuthToken', '{auth_token}');
                                localStorage.setItem('_$NETHER_TOKEN', '{auth_token}');
                                localStorage.setItem('auth_token', '{auth_token}');
                            """)
                    except Exception:
                        pass



                # Navigate to chapter page
                await page.goto(chapter_url, wait_until="domcontentloaded", timeout=25000)

                # Wait for content container
                try:
                    await page.wait_for_selector("div.prose.mx-auto", timeout=10000)
                except Exception:
                    await page.wait_for_timeout(2000)

                content_el = await page.query_selector("div.prose.mx-auto")
                if not content_el:
                    body_text = await page.inner_text("body")
                    if any(k in body_text for k in ["ยืนยันการซื้อตอน", "เข้าสู่ระบบเพื่อซื้อ", "เหรียญไม่เพียงพอ"]):
                        raise PermissionError(
                            f"Chapter {chapter_no} is a locked/paid chapter on ReadToon. "
                            "Please provide a valid READTOON_AUTH_TOKEN in environment variables or configuration."
                        )
                    raise ValueError(f"Novel content container (div.prose.mx-auto) not found on {chapter_url}")

                html_content = await content_el.inner_html()
                # Convert <br> tags to newlines
                html_content = re.sub(r"<br\s*/?>", "\n", html_content)
                # Strip remaining HTML tags
                soup_text = re.sub(r"<[^>]+>", "", html_content)
                raw_paragraphs = [p.strip() for p in soup_text.split("\n") if p.strip()]

                if not raw_paragraphs:
                    raise ValueError(f"No text extracted from chapter {chapter_no} on {chapter_url}")

                # Extract title from page title
                page_title = await page.title()
                # Pattern: 'Novel Title - ตอนที่ X: Name - ReadToon'
                m_title = re.search(r"-\s*(ตอนที่\s*\d+[^-\n]*?)\s*-\s*ReadToon", page_title)
                if m_title:
                    chapter_title = m_title.group(1).strip()
                else:
                    chapter_title = f"ตอนที่ {chapter_no}"

                # Construct paragraphs: since readtoon is already Thai, both text and translated_text are Thai
                paragraphs: List[Paragraph] = []
                for i, p_text in enumerate(raw_paragraphs, 1):
                    paragraphs.append(
                        Paragraph(
                            index=i,
                            text=p_text,
                            translated_text=p_text,
                            speech_type="dialogue" if any(p_text.startswith(q) for q in ('"', "“", "「", "「")) else "narration",
                        )
                    )

                return Chapter(
                    id=str(chapter_no),
                    book_id=slug,
                    title=chapter_title,
                    url=chapter_url,
                    chapter_number=float(chapter_no),
                    paragraphs=paragraphs,
                    translated_title=chapter_title,
                    source_language="th",
                    target_language="th",
                    metadata={
                        "total_paragraphs": len(paragraphs),
                        "source": "readtoon",
                        "slug": slug,
                        "chapter_no": chapter_no,
                    },
                )
            finally:
                await browser.close()
