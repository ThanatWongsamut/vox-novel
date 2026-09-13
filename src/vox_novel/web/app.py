import asyncio
import json
import os
import re
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from vox_novel.models.domain import Novel, Chapter, ChapterSummary, Paragraph
from vox_novel.pipeline.manager import NovelPipeline
from vox_novel.scrapers.readtoon import classify_speech_type
from vox_novel.speaker.detector import score_attribution
from vox_novel.storage.file import (
    StorageManager,
    UnsafePathSegment,
    character_voice_key,
    validate_path_segment,
)
from vox_novel.storage.knowledge import KnowledgeManager

web_app = FastAPI(title="VoxNovel Web UI")

# The importer runs as a content script, so its fetches carry the *page* origin
# (readtoon.com), not the extension origin. Allow only those specific origins --
# never "*", and never with credentials, or any site the user visits could read
# responses from this local server.
DEFAULT_ALLOWED_ORIGIN_REGEX = (
    r"^(?:"
    r"chrome-extension://[a-p]{32}"
    r"|moz-extension://[0-9a-f-]{36}"
    r"|https?://(?:localhost|127\.0\.0\.1)(?::\d+)?"
    r"|https://(?:[a-z0-9-]+\.)?readtoon\.com"
    r")$"
)
ALLOWED_ORIGIN_REGEX = os.getenv("VOXNOVEL_ALLOWED_ORIGIN_REGEX", DEFAULT_ALLOWED_ORIGIN_REGEX)

web_app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

pipeline = NovelPipeline()
storage = StorageManager()
knowledge_mgr = KnowledgeManager()

# In-memory progress tracking for active translation jobs
JOBS: Dict[str, dict] = {}
# In-memory pending reviews before user confirmation
PENDING_REVIEWS: Dict[str, dict] = {}
# asyncio only holds weak references to running tasks; keep strong ones here so
# long jobs are not garbage collected mid-flight.
BACKGROUND_TASKS: set = set()


def spawn_background(coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    BACKGROUND_TASKS.add(task)
    task.add_done_callback(BACKGROUND_TASKS.discard)
    return task


# A voice reference only needs a few seconds of audio; cap uploads well below that.
MAX_VOICE_UPLOAD_BYTES = 25 * 1024 * 1024
FFMPEG_TIMEOUT_SECONDS = 60


def safe_id(value: Optional[str], field: str = "identifier") -> str:
    """Validate an untrusted id before it is joined onto a storage path."""
    try:
        return validate_path_segment(value, field)
    except UnsafePathSegment as e:
        raise HTTPException(status_code=400, detail=str(e))


def _spool_upload(file: UploadFile, ext: str) -> Path:
    """Write an upload to a temp file, refusing anything over the size cap."""
    written = 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp_path = Path(tmp.name)
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_VOICE_UPLOAD_BYTES:
                tmp.close()
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"Audio file too large (max {MAX_VOICE_UPLOAD_BYTES // (1024 * 1024)}MB)",
                )
            tmp.write(chunk)
    return tmp_path


def _transcode_reference_audio(tmp_path: Path, out_file: Path) -> None:
    """Normalize reference audio to 48kHz mono WAV, trimmed to 15s."""
    cmd = [
        "ffmpeg", "-y", "-i", str(tmp_path),
        "-t", "15",
        "-ar", "48000",
        "-ac", "1",
        str(out_file),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT_SECONDS)
        ok = res.returncode == 0 and out_file.exists()
    except (OSError, subprocess.TimeoutExpired):
        ok = False

    if not ok:
        # Fallback for hosts without ffmpeg installed.
        data, sr = sf.read(str(tmp_path))
        if data.ndim > 1:
            data = data.mean(axis=1)
        sf.write(str(out_file), data.astype(np.float32), sr)


@web_app.get("/api/cover/{series_id}")
async def get_cover(series_id: str):
    series_id = safe_id(series_id, "series_id")
    cover_file = storage.get_local_cover_file(series_id)
    if cover_file and cover_file.exists():
        return FileResponse(cover_file, media_type="image/jpeg")

    novel = storage.get_novel(series_id)
    if novel and novel.cover_url:
        cached = storage.download_and_cache_cover(series_id, novel.cover_url)
        cover_file = storage.get_local_cover_file(series_id)
        if cover_file and cover_file.exists():
            return FileResponse(cover_file, media_type="image/jpeg")

    raise HTTPException(status_code=404, detail="Cover not found")


@web_app.api_route("/api/audio/{series_id}/{chapter_id}", methods=["GET", "HEAD"])
async def get_chapter_audio(series_id: str, chapter_id: str):
    series_id = safe_id(series_id, "series_id")
    chapter_id = safe_id(chapter_id, "chapter_id")
    audio_file = storage.get_chapter_audio_file(series_id, chapter_id)
    if not audio_file or not audio_file.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    media_type = "audio/mpeg" if audio_file.suffix.lower() == ".mp3" else "audio/wav"
    return FileResponse(
        audio_file,
        media_type=media_type,
        headers={"Accept-Ranges": "bytes"}
    )


@web_app.api_route("/api/audio/{series_id}/{chapter_id}/para/{para_idx}", methods=["GET", "HEAD"])
async def get_paragraph_audio(series_id: str, chapter_id: str, para_idx: int):
    series_id = safe_id(series_id, "series_id")
    chapter_id = safe_id(chapter_id, "chapter_id")
    if para_idx < 0:
        raise HTTPException(status_code=400, detail="Invalid paragraph index")
    chapters_dir = storage.base_dir / series_id / "chapters"
    # para_idx is Paragraph.index (1-based), matching the chunk names written by the TTS engines.
    p_file = chapters_dir / f"para_{chapter_id}_{para_idx}.wav"
    if not p_file.exists():
        raise HTTPException(status_code=404, detail="Paragraph audio not found")
    return FileResponse(p_file, media_type="audio/wav", headers={"Accept-Ranges": "bytes"})


@web_app.api_route("/api/test-voice/{num}", methods=["GET", "HEAD"])
async def get_test_voice(num: int):
    p = Path("output") / f"test_voice{num}.wav"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Test voice not found")
    return FileResponse(p, media_type="audio/wav", headers={"Accept-Ranges": "bytes"})


@web_app.api_route("/api/voices/{series_id}/narrator", methods=["GET", "HEAD"])
async def get_narrator_voice(series_id: str):
    """Serve the master narrator reference audio anchor."""
    series_id = safe_id(series_id, "series_id")
    p = storage.get_narrator_voice_file(series_id)
    if not p or not p.exists():
        raise HTTPException(status_code=404, detail="Narrator reference voice not found")
    media_type = "audio/mpeg" if p.suffix.lower() == ".mp3" else "audio/wav"
    return FileResponse(p, media_type=media_type, headers={"Accept-Ranges": "bytes"})


@web_app.api_route("/api/voices/{series_id}/character/{character_name}", methods=["GET", "HEAD"])
async def get_character_voice(series_id: str, character_name: str):
    """Serve character-specific reference voice anchor."""
    series_id = safe_id(series_id, "series_id")
    p = storage.get_character_voice_file(series_id, character_name)
    if not p or not p.exists():
        raise HTTPException(status_code=404, detail=f"Voice not found for character '{character_name}'")
    media_type = "audio/mpeg" if p.suffix.lower() == ".mp3" else "audio/wav"
    return FileResponse(p, media_type=media_type, headers={"Accept-Ranges": "bytes"})


@web_app.get("/explore", response_class=HTMLResponse)
async def explore_novels(request: Request, q: Optional[str] = None):
    from vox_novel.scrapers.webnovel import search_webnovel, browse_webnovel
    if q and q.strip():
        results = await search_webnovel(q.strip())
    else:
        results = await browse_webnovel("novel")

    return templates.TemplateResponse(
        request=request,
        name="explore.html",
        context={"results": results, "query": q or ""},
    )


@web_app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    novels = storage.list_saved_series()
    card_items = []
    for n in novels:
        translated_map = storage.get_translated_chapter_ids(n.id, target_lang="th")
        total = len(n.chapters)
        done = len(translated_map)
        pct = (done / total * 100) if total > 0 else 0
        card_items.append({
            "novel": n,
            "total_chapters": total,
            "translated_count": done,
            "progress_pct": pct,
        })

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"novels": card_items},
    )


@web_app.post("/api/import-novel")
async def import_novel(url: str = Form(...)):
    novel = await pipeline.scrape_novel(url.strip())
    return RedirectResponse(url=f"/series/{novel.id}", status_code=303)


class ExtensionImportRequest(BaseModel):
    url: str
    series_id: Optional[str] = None
    series_title: Optional[str] = None
    chapter_no: Optional[float] = None
    chapter_title: Optional[str] = None
    content: Optional[str] = ""
    paragraphs: Optional[List[str]] = None
    cover_url: Optional[str] = None
    author: Optional[str] = None
    source: str = "readtoon"
    source_language: str = "th"


@web_app.get("/api/extension/health")
async def extension_health():
    """Connectivity verification endpoint for the VoxNovel Chrome Extension."""
    return {"status": "ok", "app": "vox-novel", "version": "1.0.0"}


@web_app.post("/api/extension/import")
async def extension_import(req: ExtensionImportRequest):
    """Direct 1-click chapter ingestion endpoint for the Chrome Extension."""
    url = req.url.strip()
    series_id = (req.series_id or "").strip()
    chapter_no = req.chapter_no

    # Extract series_id and chapter_no from url if missing
    if not series_id:
        m = re.search(r"/content/([^/?#]+)(?:/(\d+(?:\.\d+)?))?", url)
        if m:
            series_id = m.group(1)
            if chapter_no is None and m.group(2):
                chapter_no = float(m.group(2))
        else:
            series_id = "imported-novel"

    # series_id becomes a directory name under the storage root -- never trust it raw.
    series_id = safe_id(series_id, "series_id")

    # Extract chapter_no from chapter_title if still None
    if chapter_no is None and req.chapter_title:
        m_no = re.search(r"(?:ตอนที่|chapter|ch\.?)\s*(\d+(?:\.\d+)?)", req.chapter_title, re.IGNORECASE)
        if m_no:
            chapter_no = float(m_no.group(1))

    # Parse and clean paragraphs
    raw_paras = []
    if req.paragraphs and len(req.paragraphs) > 0:
        raw_paras = req.paragraphs
    elif req.content:
        clean_text = re.sub(r"<\s*br\s*/?>", "\n", req.content, flags=re.IGNORECASE)
        clean_text = re.sub(r"</p>", "\n\n", clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r"<[^>]+>", "", clean_text)
        raw_paras = clean_text.split("\n")

    cleaned_paragraphs = [p.strip() for p in raw_paras if p.strip()]
    if not cleaned_paragraphs:
        raise HTTPException(status_code=400, detail="No paragraph content found to import.")

    # Chapter ID
    if chapter_no is not None:
        chap_id = str(int(chapter_no)) if chapter_no.is_integer() else str(chapter_no)
    else:
        chap_id = str(uuid.uuid4())[:8]

    novel = storage.get_novel(series_id)

    # Series Title
    novel_title = (req.series_title or "").strip()
    if not novel_title:
        novel_title = novel.title if novel and novel.title else series_id.replace("-", " ").title()

    # Resolve chapter title canonically matching catalog
    chap_title = (req.chapter_title or "").strip()
    existing_ch = next(
        (
            c
            for c in (novel.chapters if novel else [])
            if str(c.id) == str(chap_id)
            or (chapter_no is not None and c.chapter_number == chapter_no)
        ),
        None,
    )

    if existing_ch and existing_ch.title and existing_ch.title != novel_title:
        # Use canonical catalog title (e.g. "ตอนที่ 170: ฉันถูกเข้าใจผิดว่าเป็นผี0170")
        chap_title = existing_ch.title
    else:
        # Fallback or normalize format: replace "ตอนที่ X - " with "ตอนที่ X: "
        if not chap_title or chap_title.lower() == novel_title.lower():
            chap_title = (
                f"ตอนที่ {int(chapter_no)}"
                if chapter_no is not None and chapter_no.is_integer()
                else (f"Chapter {chapter_no}" if chapter_no is not None else "Imported Chapter")
            )
        elif chapter_no is not None:
            c_int = int(chapter_no) if chapter_no.is_integer() else chapter_no
            chap_title = re.sub(rf"^ตอนที่\s*{c_int}\s*[-–—]\s*", f"ตอนที่ {c_int}: ", chap_title)

    is_th = req.source_language == "th"

    para_objs = [
        Paragraph(
            id=str(uuid.uuid4())[:8],
            index=i,
            text=p_text,
            translated_text=p_text if is_th else None,
            speech_type=classify_speech_type(p_text),
        )
        for i, p_text in enumerate(cleaned_paragraphs, 1)
    ]

    chapter = Chapter(
        id=chap_id,
        book_id=series_id,
        title=chap_title,
        url=url,
        chapter_number=chapter_no,
        paragraphs=para_objs,
        translated_title=chap_title if is_th else None,
        source_language=req.source_language,
        target_language="th" if is_th else None,
        metadata={"source": req.source},
    )

    await asyncio.to_thread(storage.save_chapter, chapter, True)

    # Update or create Novel
    novel = storage.get_novel(series_id)
    summary_item = ChapterSummary(
        id=chapter.id,
        book_id=series_id,
        title=chapter.title,
        url=chapter.url,
        chapter_number=chapter.chapter_number,
        is_locked=False,
    )

    if novel:
        existing_idx = next((i for i, c in enumerate(novel.chapters) if c.id == chapter.id), -1)
        if existing_idx >= 0:
            if (
                novel.chapters[existing_idx].title
                and "ตอนที่" in novel.chapters[existing_idx].title
                and len(novel.chapters[existing_idx].title) > len(chapter.title)
            ):
                summary_item.title = novel.chapters[existing_idx].title
            novel.chapters[existing_idx] = summary_item
        else:
            novel.chapters.append(summary_item)

        novel.chapters.sort(key=lambda c: c.chapter_number if c.chapter_number is not None else 999999)

        if req.cover_url and not novel.cover_url:
            novel.cover_url = req.cover_url
        if req.series_title and (novel.title == series_id.replace("-", " ").title() or not novel.title):
            novel.title = req.series_title

        await asyncio.to_thread(storage.save_novel_metadata, novel)
    else:
        new_novel = Novel(
            id=series_id,
            title=novel_title,
            url=f"https://readtoon.com/content/{series_id}" if "readtoon" in req.source else url,
            author=req.author,
            cover_url=req.cover_url,
            source=req.source,
            chapters=[summary_item],
            metadata={"imported_via": "chrome-extension"},
        )
        await asyncio.to_thread(storage.save_novel_metadata, new_novel)

    return {
        "status": "success",
        "message": f"Successfully imported {chap_title}",
        "series_id": series_id,
        "chapter_id": chapter.id,
        "read_url": f"/series/{series_id}/read/{chapter.id}",
        "series_url": f"/series/{series_id}",
        "paragraph_count": len(para_objs),
    }



@web_app.get("/series/{series_id}", response_class=HTMLResponse)
async def series_detail(request: Request, series_id: str):
    series_id = safe_id(series_id, "series_id")
    novel = storage.get_novel(series_id)
    if not novel:
        raise HTTPException(status_code=404, detail="Series not found")

    translated_map = storage.get_translated_chapter_ids(series_id, target_lang="th")

    next_untranslated = None
    for ch in novel.chapters:
        if ch.id not in translated_map:
            next_untranslated = ch
            break

    return templates.TemplateResponse(
        request=request,
        name="series.html",
        context={
            "novel": novel,
            "chapters": novel.chapters,
            "translated_map": translated_map,
            "translated_count": len(translated_map),
            "next_untranslated": next_untranslated,
        },
    )


@web_app.post("/api/translate-chapter")
async def translate_chapter(
    request: Request,
    series_id: str = Form(...),
    chapter_url: str = Form(...),
    model: Optional[str] = Form(None),
    target_lang: str = Form("th"),
    agentic: str = Form("true"),
):
    series_id = safe_id(series_id, "series_id")
    agentic_mode = agentic.lower() in ("true", "1", "yes", "on")
    chosen_model = model or os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
    translator_name = "openrouter" if os.getenv("OPENROUTER_API_KEY") else "dummy"

    # Start background job immediately so user gets live SSE progress from second 0
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status": "starting",
        "progress": 5,
        "message": "Fetching chapter paragraphs...",
        "redirect_url": None,
        "review_url": None,
        "error": None,
    }

    async def _run_job():
        try:
            async def progress_cb(pct: int, msg: str):
                if job_id in JOBS:
                    JOBS[job_id]["progress"] = pct
                    JOBS[job_id]["message"] = msg

            await progress_cb(5, "Scraping raw chapter text...")
            chapter = await pipeline.get_chapter_raw(chapter_url.strip())

            # If chapter is already in target language (e.g. Readtoon is already Thai):
            if chapter.source_language == target_lang or chapter.source_language == "th":
                await progress_cb(80, "Content already in Thai. Ingesting chapter...")
                chap = await pipeline.scrape_and_translate_chapter(
                    url=chapter_url.strip(),
                    target_lang=target_lang,
                    chapter_obj=chapter,
                    progress_callback=progress_cb,
                )
                JOBS[job_id]["progress"] = 100
                JOBS[job_id]["status"] = "completed"
                JOBS[job_id]["message"] = "Chapter ingested! Opening reader..."
                JOBS[job_id]["redirect_url"] = f"/series/{series_id}/read/{chap.id}"
                return

            await progress_cb(10, "Pre-scanning chapter for new characters & entities...")
            new_terms, new_chars, _ = await pipeline.pre_scan_chapter_entities(
                chapter=chapter,
                target_lang=target_lang,
                translator_name=translator_name,
                model=chosen_model,
                progress_callback=progress_cb,
            )


            if new_terms or new_chars:
                review_id = str(uuid.uuid4())
                PENDING_REVIEWS[review_id] = {
                    "series_id": series_id,
                    "chapter_url": chapter_url,
                    "chapter_title": chapter.title,
                    "new_terms": new_terms,
                    "new_characters": new_chars,
                    "model": chosen_model,
                    "target_lang": target_lang,
                    "agentic_mode": agentic_mode,
                    "review_id": review_id,
                }
                JOBS[job_id]["progress"] = 15
                JOBS[job_id]["status"] = "needs_review"
                JOBS[job_id]["message"] = f"Found {len(new_chars)} character(s) and {len(new_terms)} term(s). Opening review..."
                JOBS[job_id]["review_url"] = f"/series/{series_id}/review/{review_id}"
                return

            # No new terms found: proceed directly with translation
            await progress_cb(15, "Glossary up to date. Starting translation...")
            kwargs = {"model": chosen_model} if translator_name == "openrouter" else {}
            chap = await pipeline.scrape_and_translate_chapter(
                url=chapter_url.strip(),
                target_lang=target_lang,
                translator_name=translator_name,
                translator_kwargs=kwargs,
                chapter_obj=chapter,
                auto_learn=True,
                progress_callback=progress_cb,
                agentic_mode=agentic_mode,
            )
            JOBS[job_id]["progress"] = 100
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["message"] = "Translation finished! Opening reader..."
            JOBS[job_id]["redirect_url"] = f"/series/{series_id}/read/{chap.id}"
        except Exception as e:
            if job_id in JOBS:
                JOBS[job_id]["status"] = "failed"
                JOBS[job_id]["error"] = str(e)

    spawn_background(_run_job())
    return {"job_id": job_id, "status": "started"}


@web_app.get("/series/{series_id}/review/{review_id}", response_class=HTMLResponse)
async def review_chapter_page(request: Request, series_id: str, review_id: str):
    series_id = safe_id(series_id, "series_id")
    data = PENDING_REVIEWS.get(review_id)
    if not data or data.get("series_id") != series_id:
        return RedirectResponse(f"/series/{series_id}", status_code=302)

    return templates.TemplateResponse(
        request=request,
        name="review.html",
        context={
            "request": request,
            **data,
        },
    )


@web_app.post("/api/confirm-translation")
async def confirm_translation(request: Request):
    form_data = await request.form()
    series_id = form_data.get("series_id")
    chapter_url = form_data.get("chapter_url")
    model = form_data.get("model") or os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
    target_lang = form_data.get("target_lang", "th")
    agentic_val = form_data.get("agentic", "true")
    agentic_mode = str(agentic_val).lower() in ("true", "1", "yes", "on")

    review_id = form_data.get("review_id")
    if review_id and review_id in PENDING_REVIEWS:
        PENDING_REVIEWS.pop(review_id, None)

    knowledge = knowledge_mgr.load_or_init(series_id, target_lang=target_lang)

    idx = 0
    while f"char_name_en_{idx}" in form_data:
        if form_data.get(f"char_include_{idx}") == "1":
            name_en = form_data.get(f"char_name_en_{idx}")
            target_name = form_data.get(f"char_target_{idx}")
            gender = form_data.get(f"char_gender_{idx}")
            role = form_data.get(f"char_role_{idx}")
            raw_aliases = form_data.get(f"char_aliases_{idx}", "")
            alias_list = [a.strip() for a in raw_aliases.split(",") if a.strip()]
            if name_en and target_name:
                knowledge.add_character(
                    name_en, target_name, gender=gender or None, role=role, aliases=alias_list
                )
        idx += 1

    idx = 0
    while f"term_source_{idx}" in form_data:
        if form_data.get(f"term_include_{idx}") == "1":
            source = form_data.get(f"term_source_{idx}")
            target_term = form_data.get(f"term_target_{idx}")
            cat = form_data.get(f"term_category_{idx}", "general")
            notes = form_data.get(f"term_notes_{idx}")
            raw_aliases = form_data.get(f"term_aliases_{idx}", "")
            alias_list = [a.strip() for a in raw_aliases.split(",") if a.strip()]
            if source and target_term:
                knowledge.add_term(source, target_term, category=cat, aliases=alias_list, notes=notes)
        idx += 1

    knowledge_mgr.save(knowledge)

    # Start live progress job after approval
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status": "starting",
        "progress": 5,
        "message": "Applying approved terms to chapter...",
        "redirect_url": None,
        "error": None,
    }

    async def _run_job():
        try:
            async def progress_cb(pct: int, msg: str):
                if job_id in JOBS:
                    JOBS[job_id]["progress"] = pct
                    JOBS[job_id]["message"] = msg

            translator_name = "openrouter" if os.getenv("OPENROUTER_API_KEY") else "dummy"
            kwargs = {"model": model} if translator_name == "openrouter" else {}

            chap = await pipeline.scrape_and_translate_chapter(
                url=chapter_url,
                target_lang=target_lang,
                translator_name=translator_name,
                translator_kwargs=kwargs,
                auto_learn=False,
                progress_callback=progress_cb,
                agentic_mode=agentic_mode,
            )
            JOBS[job_id]["progress"] = 100
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["message"] = "Translation finished! Opening reader..."
            JOBS[job_id]["redirect_url"] = f"/series/{series_id}/read/{chap.id}"
        except Exception as e:
            if job_id in JOBS:
                JOBS[job_id]["status"] = "failed"
                JOBS[job_id]["error"] = str(e)

    spawn_background(_run_job())
    return {"job_id": job_id, "status": "started"}


@web_app.get("/api/progress/{job_id}")
async def sse_job_progress(job_id: str):
    """Server-Sent Events stream for real-time progress updates."""
    async def event_generator():
        while True:
            if job_id not in JOBS:
                yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                break

            job = JOBS[job_id]
            data_json = json.dumps(job)
            yield f"data: {data_json}\n\n"

            if job.get("status") in ("completed", "failed"):
                # Clean up after completion
                await asyncio.sleep(1.0)
                JOBS.pop(job_id, None)
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@web_app.get("/series/{series_id}/read/{chapter_id}", response_class=HTMLResponse)
async def read_chapter(request: Request, series_id: str, chapter_id: str):
    series_id = safe_id(series_id, "series_id")
    chapter_id = safe_id(chapter_id, "chapter_id")
    novel = storage.get_novel(series_id)
    chapters_dir = storage.base_dir / series_id / "chapters"

    target_file = None
    for jf in chapters_dir.glob("*.json"):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("id") == chapter_id:
                    target_file = jf
                    break
        except Exception:
            pass

    if not target_file or not target_file.exists():
        raise HTTPException(status_code=404, detail="Chapter not found")

    with open(target_file, "r", encoding="utf-8") as f:
        chapter = Chapter.model_validate(json.load(f))

    prev_chap_id = None
    next_chap_id = None
    if novel:
        trans_map = storage.get_translated_chapter_ids(series_id, target_lang="th")
        trans_ids = [c.id for c in novel.chapters if c.id in trans_map]
        if chapter_id in trans_ids:
            idx = trans_ids.index(chapter_id)
            if idx > 0:
                prev_chap_id = trans_ids[idx - 1]
            if idx < len(trans_ids) - 1:
                next_chap_id = trans_ids[idx + 1]

    audio_file = storage.get_chapter_audio_file(series_id, chapter_id)
    audio_url = f"/api/audio/{series_id}/{chapter_id}" if (audio_file and audio_file.exists()) else None

    return templates.TemplateResponse(
        request=request,
        name="reader.html",
        context={
            "chapter": chapter,
            "prev_chap_id": prev_chap_id,
            "next_chap_id": next_chap_id,
            "audio_url": audio_url,
        },
    )


@web_app.post("/api/synthesize-chapter")
async def synthesize_chapter_endpoint(
    series_id: str = Form(...),
    chapter_id: str = Form(...),
    engine: str = Form("voxcpm2"),
    voice: Optional[str] = Form(None),
):
    series_id = safe_id(series_id, "series_id")
    chapter_id = safe_id(chapter_id, "chapter_id")
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status": "starting",
        "progress": 5,
        "message": "Initializing VoxCPM2 TTS...",
        "redirect_url": None,
        "audio_url": None,
        "error": None,
    }

    async def _run_job():
        try:
            async def progress_cb(pct: int, msg: str):
                if job_id in JOBS:
                    JOBS[job_id]["progress"] = pct
                    JOBS[job_id]["message"] = msg

            await progress_cb(10, "Preparing chapter text for VoxCPM2...")
            audio_path = await pipeline.synthesize_chapter_audio(
                series_id=series_id,
                chapter_id=chapter_id,
                engine_name=engine,
                voice_description=voice,
                progress_callback=progress_cb,
            )
            JOBS[job_id]["progress"] = 100
            JOBS[job_id]["status"] = "completed"
            # Leave the engine's own closing message in place: it reports when the
            # audio is only placeholder tones rather than real speech.
            JOBS[job_id]["audio_url"] = f"/api/audio/{series_id}/{chapter_id}"
        except Exception as e:
            if job_id in JOBS:
                JOBS[job_id]["status"] = "failed"
                JOBS[job_id]["error"] = str(e)

    spawn_background(_run_job())
    return {"job_id": job_id, "status": "started"}


@web_app.get("/series/{series_id}/speakers/{chapter_id}", response_class=HTMLResponse)
async def speaker_review(request: Request, series_id: str, chapter_id: str, lang: str = "th"):
    """Review and correct who speaks each line of a chapter.

    Attribution is not reliable enough to run unattended, so this is where a
    human fixes it. Correcting a line here is also how a gold set gets built:
    the corrections are the labels.
    """
    series_id = safe_id(series_id, "series_id")
    chapter_id = safe_id(chapter_id, "chapter_id")

    chapter = storage.get_chapter(series_id, chapter_id)
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")

    knowledge = knowledge_mgr.load_or_init(series_id, target_lang=lang)
    characters = sorted(
        knowledge.characters.values(), key=lambda c: (not c.is_narrator, c.name_en.lower())
    )
    attributed = sum(1 for p in chapter.paragraphs if p.speaker)
    spoken = sum(1 for p in chapter.paragraphs if p.speech_type in ("dialogue", "thought"))
    verified = sum(1 for p in chapter.paragraphs if p.speaker_verified)
    unreviewed = sum(
        1 for p in chapter.paragraphs
        if p.speech_type in ("dialogue", "thought") and not p.speaker_verified
    )
    score = score_attribution(chapter)

    return templates.TemplateResponse(
        request=request,
        name="speakers.html",
        context={
            "chapter": chapter,
            "characters": characters,
            "narration_note": knowledge.narration_note or "",
            "near_duplicates": knowledge.near_duplicate_characters(),
            "attributed": attributed,
            "spoken": spoken,
            "verified": verified,
            "unreviewed": unreviewed,
            "score": score,
            "lang": lang,
        },
    )


@web_app.post("/api/speakers/update")
async def update_speaker(request: Request):
    """Correct one paragraph's speaker or type."""
    body = await request.json()
    series_id = safe_id(body.get("series_id"), "series_id")
    chapter_id = safe_id(body.get("chapter_id"), "chapter_id")
    try:
        index = int(body.get("paragraph"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="paragraph must be an integer")

    speech_type = body.get("speech_type")
    if speech_type not in ("dialogue", "thought", "narration"):
        raise HTTPException(
            status_code=400, detail="speech_type must be dialogue, thought or narration"
        )
    speaker = (body.get("speaker") or "").strip() or None

    chapter = storage.get_chapter(series_id, chapter_id)
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")

    target = next((p for p in chapter.paragraphs if p.index == index), None)
    if target is None:
        raise HTTPException(status_code=404, detail=f"Paragraph {index} not found")

    # Narration has no speaker; storing one would leave the two disagreeing.
    target.speech_type = speech_type
    target.speaker = None if speech_type == "narration" else speaker
    # A human corrected this line, so it is ground truth rather than a guess.
    target.speaker_verified = True

    await asyncio.to_thread(storage.save_chapter, chapter, True)
    # The review page saves without reloading, so it needs the running score
    # back or its accuracy readout goes stale the moment a correction lands.
    score = score_attribution(chapter)
    return {
        "status": "ok",
        "paragraph": index,
        "speaker": target.speaker,
        "detected": target.speaker_detected,
        "score": {
            "scored": score["scored"],
            "correct": score["correct"],
            "wrong": score["wrong"],
            "accuracy": score["accuracy"],
        },
    }


@web_app.post("/api/speakers/verify-rest")
async def verify_remaining_speakers(request: Request):
    """Mark every unreviewed spoken line in a chapter as correct as detected.

    A read-through only produces labels for the lines a reviewer *changed* --
    the ones detection got right are never touched, so a chapter read end to end
    scores on a handful of paragraphs, all of them errors. This is the other
    half of the verdict: everything left is correct.

    Only offered as a bulk action after reading the chapter, and it never
    overwrites an existing verdict.
    """
    body = await request.json()
    series_id = safe_id(body.get("series_id"), "series_id")
    chapter_id = safe_id(body.get("chapter_id"), "chapter_id")

    chapter = storage.get_chapter(series_id, chapter_id)
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")

    confirmed = []
    for p in chapter.paragraphs:
        if p.speaker_verified or p.speech_type not in ("dialogue", "thought"):
            continue
        p.speaker_verified = True
        confirmed.append(p.index)

    await asyncio.to_thread(storage.save_chapter, chapter, True)
    score = score_attribution(chapter)
    return {
        "status": "ok",
        "confirmed": len(confirmed),
        "score": {
            "scored": score["scored"],
            "correct": score["correct"],
            "wrong": score["wrong"],
            "accuracy": score["accuracy"],
        },
    }


@web_app.get("/series/{series_id}/glossary", response_class=HTMLResponse)
async def series_glossary(request: Request, series_id: str, lang: str = "th"):
    series_id = safe_id(series_id, "series_id")
    knowledge = knowledge_mgr.load_or_init(series_id, target_lang=lang)
    narrator_voice_file = storage.get_narrator_voice_file(series_id)
    has_narrator_ref = narrator_voice_file is not None and narrator_voice_file.exists()

    # Map character voice status for template rendering
    char_voice_status = {}
    for key, char in knowledge.characters.items():
        c_file = storage.get_character_voice_file(series_id, char.name_en)
        char_voice_status[key] = {
            "has_ref": c_file is not None and c_file.exists(),
            "has_desc": bool(char.voice_description),
            "ref_path": str(c_file) if c_file else None,
        }

    return templates.TemplateResponse(
        request=request,
        name="glossary.html",
        context={
            "knowledge": knowledge,
            "has_narrator_ref": has_narrator_ref,
            "char_voice_status": char_voice_status,
        },
    )


@web_app.post("/api/add-term")
async def add_term(
    series_id: str = Form(...),
    target_lang: str = Form("th"),
    source: str = Form(...),
    target: str = Form(...),
    category: str = Form("general"),
    gender: str = Form(""),
    aliases: str = Form(""),
):
    series_id = safe_id(series_id, "series_id")
    knowledge = knowledge_mgr.load_or_init(series_id, target_lang=target_lang)
    alias_list = [a.strip() for a in aliases.split(",") if a.strip()]
    if category == "character":
        knowledge.add_character(source, target, gender=gender or None, aliases=alias_list)
    else:
        knowledge.add_term(source, target, category=category, aliases=alias_list)
    knowledge_mgr.save(knowledge)
    return RedirectResponse(url=f"/series/{series_id}/glossary", status_code=303)


@web_app.post("/api/glossary/update")
async def update_glossary_item(request: Request):
    """Update an existing term or character in the series knowledge."""
    body = await request.json()
    series_id = body.get("series_id")
    target_lang = body.get("target_lang", "th")
    item_type = body.get("type")  # 'character' or 'term'
    old_key = body.get("old_key")  # original source/name_en
    new_name = body.get("name")
    new_target = body.get("target")
    gender = body.get("gender")
    role_or_cat = body.get("role_or_category", "")
    notes = body.get("notes")
    aliases = body.get("aliases", [])
    voice_description = body.get("voice_description")

    if not series_id or not old_key or not new_name or not new_target:
        raise HTTPException(status_code=400, detail="Missing required fields")

    series_id = safe_id(series_id, "series_id")
    knowledge = knowledge_mgr.load_or_init(series_id, target_lang=target_lang)

    if item_type == "character":
        existing_char = knowledge.find_character(old_key)
        v_ref = existing_char.voice_ref_audio if existing_char else None
        v_desc = voice_description if voice_description is not None else (existing_char.voice_description if existing_char else None)

        # Remove old if name changed
        if old_key.strip().lower() != new_name.strip().lower():
            knowledge.remove_character(old_key)
        knowledge.add_character(
            name_en=new_name,
            name_target=new_target,
            gender=gender or None,
            role=role_or_cat or None,
            aliases=aliases,
            notes=notes or None,
            voice_description=v_desc,
            voice_ref_audio=v_ref,
        )
    else:
        # Term
        if old_key.strip().lower() != new_name.strip().lower():
            knowledge.remove_term(old_key)
        knowledge.add_term(
            source=new_name,
            target=new_target,
            category=role_or_cat or "general",
            aliases=aliases,
            notes=notes or None,
        )

    knowledge_mgr.save(knowledge)
    return {"status": "ok", "message": "Updated successfully"}


@web_app.post("/api/glossary/delete")
async def delete_glossary_item(request: Request):
    """Delete a term or character from the series knowledge."""
    body = await request.json()
    series_id = body.get("series_id")
    target_lang = body.get("target_lang", "th")
    item_type = body.get("type")  # 'character' or 'term'
    key = body.get("key")

    if not series_id or not key:
        raise HTTPException(status_code=400, detail="Missing required fields")

    series_id = safe_id(series_id, "series_id")
    knowledge = knowledge_mgr.load_or_init(series_id, target_lang=target_lang)

    if item_type == "character":
        knowledge.remove_character(key)
    else:
        knowledge.remove_term(key)

    knowledge_mgr.save(knowledge)
    return {"status": "ok", "message": "Deleted successfully"}


@web_app.post("/api/voices/update-prompt")
async def update_voice_prompt(request: Request):
    """Update Voice Design prompt description for narrator or character."""
    body = await request.json()
    series_id = body.get("series_id")
    target_lang = body.get("target_lang", "th")
    voice_type = body.get("voice_type")  # "narrator" or "character"
    character_name = body.get("character_name")
    voice_description = body.get("voice_description", "").strip()
    # "own", or "body" for the voice a character's spoken lines take while they
    # are in someone else's body. Their thoughts keep their own voice.
    voice_slot = body.get("voice_slot", "own")
    from_chapter = body.get("body_voice_from_chapter")

    if voice_slot not in ("own", "body"):
        raise HTTPException(status_code=400, detail="voice_slot must be 'own' or 'body'")
    if voice_slot == "body" and voice_type != "character":
        raise HTTPException(status_code=400, detail="only a character has a body voice")
    if from_chapter is not None:
        try:
            from_chapter = float(from_chapter)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=400, detail="body_voice_from_chapter must be a number"
            )
    # A body voice with no start chapter would apply to the whole series,
    # re-voicing chapters set before the swap.
    if voice_slot == "body" and voice_description and from_chapter is None:
        raise HTTPException(
            status_code=400,
            detail="a body voice needs the chapter number the swap starts at",
        )

    if not series_id or not voice_type:
        raise HTTPException(status_code=400, detail="Missing required parameters")
    if voice_type not in ("narrator", "character"):
        raise HTTPException(
            status_code=400, detail="voice_type must be 'narrator' or 'character'"
        )

    series_id = safe_id(series_id, "series_id")
    knowledge = knowledge_mgr.load_or_init(series_id, target_lang=target_lang)

    char = None
    if voice_type == "character":
        if not character_name:
            raise HTTPException(status_code=400, detail="character_name is required for character voice")
        # character_name may be an alias -- store against the canonical name.
        char = knowledge.find_character(character_name)
        if not char:
            raise HTTPException(status_code=404, detail=f"Character '{character_name}' not found")

    # Derive the English control prompt now, while the user is waiting on a single
    # save, rather than during synthesis where it would delay the whole chapter.
    # Resolve it before writing, so the description and its control land together.
    from vox_novel.tts.voxcpm import VoxCPM2TTS

    control = await VoxCPM2TTS.derive_control_prompt(
        voice_description, translator=pipeline.voice_prompt_translator()
    )
    if voice_type == "narrator":
        knowledge.update_narrator_voice(
            voice_description=voice_description, voice_control_prompt=control
        )
    else:
        knowledge.update_character_voice(
            char.name_en,
            voice_description=voice_description,
            voice_control_prompt=control,
            slot=voice_slot,
            from_chapter=from_chapter,
        )

    knowledge_mgr.save(knowledge)
    return {
        "status": "ok",
        "message": "Voice prompt updated successfully",
        "control_prompt": control,
    }


@web_app.post("/api/voices/generate")
async def generate_voice_sample(request: Request):
    """Generate reference voice sample from prompt using VoxCPM2."""
    from vox_novel.tts.voxcpm import VoxCPM2TTS

    body = await request.json()
    series_id = body.get("series_id")
    target_lang = body.get("target_lang", "th")
    voice_type = body.get("voice_type")  # "narrator" or "character"
    character_name = body.get("character_name")
    voice_description = body.get("voice_description", "").strip()
    sample_text = body.get("sample_text", "").strip()

    if not series_id or not voice_type:
        raise HTTPException(status_code=400, detail="Missing series_id or voice_type")

    series_id = safe_id(series_id, "series_id")
    knowledge = knowledge_mgr.load_or_init(series_id, target_lang=target_lang)
    voices_dir = storage.get_voices_dir(series_id)

    tts = VoxCPM2TTS()

    if voice_type == "narrator":
        effective_desc = (
            voice_description
            or knowledge.narrator_voice_description
            or "เสียงบรรยายผู้ชาย นุ่มลึก มีชีวิตชีวา ชัดถ้อยชัดคำ เหมาะกับการเล่านิยายแฟนตาซี"
        )
        text = sample_text or "ยินดีต้อนรับสู่โลกแห่งนิยาย นี่คือเสียงตัวอย่างสำหรับผู้บรรยาย"
        out_file = voices_dir / "narrator_ref.wav"
        control = await tts.derive_control_prompt(
            effective_desc, translator=pipeline.voice_prompt_translator()
        )
        await tts.synthesize(
            text=text,
            output_file=out_file,
            voice_description=effective_desc,
            reference_audio=None,
            control_prompt=control,
        )
        knowledge.update_narrator_voice(
            voice_description=effective_desc,
            voice_ref_audio=str(out_file),
            voice_control_prompt=control,
        )
        knowledge_mgr.save(knowledge)
        return {
            "status": "ok",
            "audio_url": f"/api/voices/{series_id}/narrator",
            "voice_description": effective_desc,
            "control_prompt": control,
        }
    else:
        if not character_name:
            raise HTTPException(status_code=400, detail="character_name is required")
        char = knowledge.find_character(character_name)
        if not char:
            raise HTTPException(status_code=404, detail=f"Character '{character_name}' not found")

        is_female = (char.gender or "").lower() in ("female", "f", "หญิง")
        default_char_desc = (
            "หญิงสาววัยรุ่น เสียงหวานใส ร่าเริง อ่อนหวาน น่าฟัง"
            if is_female
            else "ชายหนุ่มวัย 20 เสียงห้าว มั่นใจ ชัดเจน เป็นมิตร"
        )
        effective_desc = voice_description or char.voice_description or default_char_desc
        text = sample_text or f"สวัสดี ข้าชื่อ{char.name_target} ยินดีที่ได้รู้จัก"
        out_file = voices_dir / f"{character_voice_key(char.name_en)}_ref.wav"
        control = await tts.derive_control_prompt(
            effective_desc, translator=pipeline.voice_prompt_translator()
        )
        await tts.synthesize(
            text=text,
            output_file=out_file,
            voice_description=effective_desc,
            reference_audio=None,
            control_prompt=control,
        )
        knowledge.update_character_voice(
            char.name_en,
            voice_description=effective_desc,
            voice_ref_audio=str(out_file),
            voice_control_prompt=control,
        )
        knowledge_mgr.save(knowledge)
        return {
            "status": "ok",
            "audio_url": f"/api/voices/{series_id}/character/{quote(char.name_en, safe='')}",
            "voice_description": effective_desc,
            "control_prompt": control,
        }


@web_app.post("/api/voices/upload")
async def upload_voice_sample(
    file: UploadFile = File(...),
    series_id: str = Form(...),
    target_lang: str = Form("th"),
    voice_type: str = Form(...),
    character_name: Optional[str] = Form(None),
):
    """Upload custom reference audio file (.wav, .mp3, etc.) for narrator or character."""
    series_id = safe_id(series_id, "series_id")
    voices_dir = storage.get_voices_dir(series_id)
    knowledge = knowledge_mgr.load_or_init(series_id, target_lang=target_lang)

    ext = Path(file.filename or "").suffix.lower()
    if ext not in [".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac"]:
        raise HTTPException(status_code=400, detail="Supported audio formats: .wav, .mp3, .m4a, .ogg, .flac")

    tmp_path = await asyncio.to_thread(_spool_upload, file, ext)

    if voice_type == "narrator":
        out_file = voices_dir / "narrator_ref.wav"
        audio_url = f"/api/voices/{series_id}/narrator"
    else:
        if not character_name:
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="character_name is required")
        char = knowledge.find_character(character_name)
        if not char:
            tmp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=404, detail=f"Character '{character_name}' not found")
        out_file = voices_dir / f"{character_voice_key(char.name_en)}_ref.wav"
        audio_url = f"/api/voices/{series_id}/character/{quote(char.name_en, safe='')}"

    try:
        await asyncio.to_thread(_transcode_reference_audio, tmp_path, out_file)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to process audio file: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)

    if voice_type == "narrator":
        knowledge.update_narrator_voice(voice_ref_audio=str(out_file))
    else:
        knowledge.update_character_voice(char.name_en, voice_ref_audio=str(out_file))
    knowledge_mgr.save(knowledge)

    return {"status": "ok", "audio_url": audio_url}


@web_app.post("/api/voices/delete")
async def delete_voice_sample(request: Request):
    """Delete reference audio anchor, reverting to prompt-to-voice or narrator fallback."""
    body = await request.json()
    series_id = body.get("series_id")
    target_lang = body.get("target_lang", "th")
    voice_type = body.get("voice_type")
    character_name = body.get("character_name")

    if not series_id or not voice_type:
        raise HTTPException(status_code=400, detail="Missing required parameters")

    series_id = safe_id(series_id, "series_id")
    storage.delete_voice_file(series_id, voice_type, character_name)
    knowledge = knowledge_mgr.load_or_init(series_id, target_lang=target_lang)
    if voice_type == "narrator":
        knowledge.update_narrator_voice(voice_ref_audio="")
    elif character_name:
        char = knowledge.find_character(character_name)
        knowledge.update_character_voice(
            char.name_en if char else character_name, voice_ref_audio=""
        )
    knowledge_mgr.save(knowledge)
    return {"status": "ok"}


@web_app.get("/api/settings")
async def get_settings():
    """Retrieve current API configuration and models."""
    from vox_novel.settings import get_app_settings
    return get_app_settings()


@web_app.post("/api/settings")
async def update_settings(request: Request):
    """Save updated API keys and model preferences."""
    from vox_novel.settings import save_app_settings
    body = await request.json()
    return save_app_settings(body)


@web_app.post("/api/settings/test-key")
async def test_api_key(request: Request):
    """Verify validity of OpenRouter API key."""
    from vox_novel.settings import verify_openrouter_api_key
    body = await request.json()
    api_key = body.get("api_key")
    return await verify_openrouter_api_key(api_key)

