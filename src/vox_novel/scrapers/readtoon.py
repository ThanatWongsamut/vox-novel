import asyncio
import json
import os
import re
from pathlib import Path
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

    @staticmethod
    def get_profile_dir() -> Path:
        """Directory where persistent Playwright browser profile is stored."""
        custom = os.getenv("READTOON_USER_DATA_DIR", "").strip()
        if custom:
            return Path(custom).expanduser()
        return Path.home() / ".vox_novel" / "readtoon_profile"

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

        profile_dir = self.get_profile_dir()
        use_persistent = profile_dir.exists() and any(profile_dir.iterdir())

        browser = None
        context = None

        async with async_playwright() as p:
            try:
                if use_persistent:
                    for opts in [{"channel": "chrome"}, {}]:
                        try:
                            context = await p.chromium.launch_persistent_context(
                                user_data_dir=str(profile_dir),
                                headless=True,
                                args=["--disable-blink-features=AutomationControlled"],
                                ignore_default_args=["--enable-automation"],
                                user_agent=self.DEFAULT_HEADERS["User-Agent"],
                                viewport={"width": 1280, "height": 800},
                                **opts,
                            )
                            break
                        except Exception:
                            continue
                else:
                    for launch_opts in [{"channel": "chrome", "headless": True}, {"headless": True}]:
                        try:
                            browser = await p.chromium.launch(**launch_opts)
                            break
                        except Exception:
                            continue
                    if not browser:
                        raise RuntimeError("Failed to launch Playwright browser (neither Google Chrome nor Chromium available).")

                    context = await browser.new_context(
                        user_agent=self.DEFAULT_HEADERS["User-Agent"],
                        viewport={"width": 1280, "height": 800},
                    )

                if not context:
                    raise RuntimeError("Failed to create browser context.")

                page = context.pages[0] if context.pages else await context.new_page()

                st_auth = ""
                st_dev = ""
                storage_file = profile_dir / "storage_state.json"
                if storage_file.exists():
                    try:
                        st = json.loads(storage_file.read_text(encoding="utf-8"))
                        if "cookies" in st and st["cookies"]:
                            await context.add_cookies(st["cookies"])
                            for c in st["cookies"]:
                                if c.get("name") == "auth_token":
                                    st_auth = c.get("value", "")
                                elif c.get("name") == "device_token":
                                    st_dev = c.get("value", "")
                    except Exception:
                        pass

                auth_token = (self.auth_token or st_auth or os.getenv("READTOON_AUTH_TOKEN", "")).strip()
                device_token = (st_dev or os.getenv("READTOON_DEVICE_TOKEN", "")).strip()
                cookie_str = os.getenv("READTOON_COOKIE", "").strip()

                cookies_to_add = []
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
                            cookies_to_add.extend([
                                {"name": c_name, "value": c_val, "domain": ".readtoon.com", "path": "/"},
                                {"name": c_name, "value": c_val, "domain": ".www.readtoon.com", "path": "/"},
                            ])
                            if c_name == "auth_token":
                                auth_token = c_val

                if auth_token:
                    cookies_to_add.extend([
                        {"name": "auth_token", "value": auth_token, "domain": ".readtoon.com", "path": "/"},
                        {"name": "token", "value": auth_token, "domain": ".readtoon.com", "path": "/"},
                        {"name": "auth_token", "value": auth_token, "domain": ".www.readtoon.com", "path": "/"},
                    ])

                if device_token:
                    cookies_to_add.extend([
                        {"name": "device_token", "value": device_token, "domain": ".readtoon.com", "path": "/"},
                        {"name": "device_token", "value": device_token, "domain": ".www.readtoon.com", "path": "/"},
                    ])

                if cookies_to_add:
                    try:
                        seen_c = set()
                        unique_cookies = []
                        for c in cookies_to_add:
                            key_c = (c["name"], c["domain"])
                            if key_c not in seen_c:
                                seen_c.add(key_c)
                                unique_cookies.append(c)
                        await context.add_cookies(unique_cookies)
                    except Exception:
                        pass

                # Pre-set localStorage tokens as proven in novel-narrator
                if auth_token:
                    try:
                        await page.goto("https://www.readtoon.com", wait_until="domcontentloaded", timeout=12000)
                        await page.evaluate(f"""() => {{
                            localStorage.setItem('_$AuthToken', '{auth_token}');
                            localStorage.setItem('_', '{auth_token}');
                        }}""")
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
                    is_login_required = (
                        any(k in body_text for k in ["เข้าสู่ระบบเพื่อซื้อ", "กรุณาเข้าสู่ระบบ"])
                        or (any(k in body_text for k in ["เข้าสู่ระบบ", "Sign In"]) and any(k in body_text for k in ["ยืนยันการซื้อ", "เหรียญ"]))
                    )
                    is_insufficient_coins = "เหรียญไม่เพียงพอ" in body_text
                    is_confirm_purchase = any(k in body_text for k in ["ยืนยันการซื้อ", "ยืนยันการซื้อตอน", "ซื้อตอนนี้"])

                    auto_purchase = os.getenv("READTOON_AUTO_PURCHASE", "false").lower() in ("1", "true", "yes")

                    # If user is logged in and purchase confirmation modal is open:
                    if is_confirm_purchase and not is_login_required and not is_insufficient_coins:
                        confirm_btn = await page.query_selector(
                            "button:has-text('ยืนยันการซื้อ'), button:has-text('ยืนยันการซื้อตอน'), button:has-text('ซื้อตอนนี้'), button:has-text('ยืนยัน')"
                        )
                        if confirm_btn and auto_purchase:
                            await confirm_btn.click()
                            try:
                                await page.wait_for_selector("div.prose.mx-auto", timeout=12000)
                                content_el = await page.query_selector("div.prose.mx-auto")
                            except Exception:
                                pass

                    if not content_el:
                        if is_login_required:
                            raise PermissionError(
                                f"Chapter {chapter_no} is a locked/paid chapter on ReadToon, but your session is not authenticated (ReadToon returned 'กรุณาเข้าสู่ระบบ'). "
                                "Please log into ReadToon using 'vox-novel login readtoon' in your terminal, or click 'Log In to ReadToon' in Settings."
                            )
                        elif is_insufficient_coins:
                            raise PermissionError(
                                f"Chapter {chapter_no} is a paid chapter on ReadToon, but your coin balance is insufficient ('เหรียญไม่เพียงพอ'). "
                                "Please top up coins on https://readtoon.com/user/wallet."
                            )
                        elif is_confirm_purchase:
                            raise PermissionError(
                                f"Chapter {chapter_no} is a paid chapter and has not been purchased yet on your ReadToon account. "
                                "To automatically unlock it with your coins, enable READTOON_AUTO_PURCHASE=true in settings, "
                                f"or unlock Chapter {chapter_no} on https://readtoon.com first."
                            )
                        raise ValueError(f"Novel content container (div.prose.mx-auto) not found on {chapter_url}")

                html_content = await content_el.inner_html()
                # Clean prose HTML using novel-narrator line break handling
                html_content = html_content.replace(
                    '<br style="color: rgb(255, 255, 255) !important; user-select: none !important;">', '\n'
                )
                html_content = re.sub(r"<br\s*/?>", "\n", html_content)
                soup_text = re.sub(r"<[^>]+>", "", html_content)
                soup_text = re.sub(r"\n{3,}", "\n\n", soup_text)
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
                if context:
                    await context.close()
                if browser:
                    await browser.close()
