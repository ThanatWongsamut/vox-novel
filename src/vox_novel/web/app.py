import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Dict, Optional
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from vox_novel.models.domain import Novel, Chapter
from vox_novel.pipeline.manager import NovelPipeline
from vox_novel.storage.file import StorageManager
from vox_novel.storage.knowledge import KnowledgeManager

web_app = FastAPI(title="VoxNovel Web UI")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

pipeline = NovelPipeline()
storage = StorageManager()
knowledge_mgr = KnowledgeManager()

# In-memory progress tracking for active translation jobs
JOBS: Dict[str, dict] = {}
# In-memory pending reviews before user confirmation
PENDING_REVIEWS: Dict[str, dict] = {}


@web_app.get("/api/cover/{series_id}")
async def get_cover(series_id: str):
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
    chapters_dir = storage.base_dir / series_id / "chapters"
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


@web_app.get("/series/{series_id}", response_class=HTMLResponse)
async def series_detail(request: Request, series_id: str):
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
    agentic_mode = agentic.lower() in ("true", "1", "yes", "on")
    chosen_model = model or os.getenv("OPENROUTER_MODEL", "minimax/minimax-m3:free")
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

    asyncio.create_task(_run_job())
    return {"job_id": job_id, "status": "started"}


@web_app.get("/series/{series_id}/review/{review_id}", response_class=HTMLResponse)
async def review_chapter_page(request: Request, series_id: str, review_id: str):
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
    model = form_data.get("model") or os.getenv("OPENROUTER_MODEL", "minimax/minimax-m3:free")
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

    asyncio.create_task(_run_job())
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
            JOBS[job_id]["message"] = "Audiobook generation complete!"
            JOBS[job_id]["audio_url"] = f"/api/audio/{series_id}/{chapter_id}"
        except Exception as e:
            if job_id in JOBS:
                JOBS[job_id]["status"] = "failed"
                JOBS[job_id]["error"] = str(e)

    asyncio.create_task(_run_job())
    return {"job_id": job_id, "status": "started"}


@web_app.get("/series/{series_id}/glossary", response_class=HTMLResponse)
async def series_glossary(request: Request, series_id: str, lang: str = "th"):
    knowledge = knowledge_mgr.load_or_init(series_id, target_lang=lang)
    return templates.TemplateResponse(
        request=request,
        name="glossary.html",
        context={"knowledge": knowledge},
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

    if not series_id or not old_key or not new_name or not new_target:
        raise HTTPException(status_code=400, detail="Missing required fields")

    knowledge = knowledge_mgr.load_or_init(series_id, target_lang=target_lang)

    if item_type == "character":
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

    knowledge = knowledge_mgr.load_or_init(series_id, target_lang=target_lang)

    if item_type == "character":
        knowledge.remove_character(key)
    else:
        knowledge.remove_term(key)

    knowledge_mgr.save(knowledge)
    return {"status": "ok", "message": "Deleted successfully"}


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

