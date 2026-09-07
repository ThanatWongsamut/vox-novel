import asyncio
import os
import re
import sys
from typing import List, Optional
import typer
import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from vox_novel.models.domain import ChapterSummary, Novel
from vox_novel.models.series_knowledge import SeriesKnowledge
from vox_novel.pipeline.manager import NovelPipeline
from vox_novel.storage.file import StorageManager
from vox_novel.storage.knowledge import KnowledgeManager

app = typer.Typer(help="VoxNovel: Novel Scraping, Translation, and TTS Pipeline")
console = Console()


def pick_chapter_interactive(chapters: List[ChapterSummary], translated_map: dict) -> Optional[ChapterSummary]:
    """Provides a fast, user-friendly way to find and select a chapter."""
    while True:
        method = questionary.select(
            "How would you like to select a chapter?",
            choices=[
                "🔍 Search by keyword or chapter title",
                "🔢 Enter Chapter Number directly (e.g. 100)",
                "⏳ Show only Untranslated chapters",
                "📄 Browse paginated chapters (1-50, 51-100...)",
                "⬅️ Back",
            ],
        ).ask()

        if not method or method == "⬅️ Back":
            return None

        if method == "🔢 Enter Chapter Number directly (e.g. 100)":
            num_str = questionary.text("Enter chapter number:").ask()
            if not num_str:
                continue
            try:
                target_num = float(num_str.strip())
                # Exact match first
                matched = [c for c in chapters if c.chapter_number == target_num]
                if not matched:
                    # Partial match on title
                    matched = [c for c in chapters if f"chapter {int(target_num)}" in c.title.lower()]
                
                if matched:
                    return matched[0]
                else:
                    console.print(f"[yellow]No chapter found matching number {num_str}.[/yellow]")
            except ValueError:
                console.print("[red]Invalid number format.[/red]")

        elif method == "🔍 Search by keyword or chapter title":
            kw = questionary.text("Enter keyword to search (e.g. 'malice', 'hunt', '100'):").ask()
            if not kw:
                continue
            matches = [
                c for c in chapters if kw.lower() in c.title.lower() or kw in str(c.chapter_number or "")
            ]
            if not matches:
                console.print(f"[yellow]No chapters found matching '{kw}'.[/yellow]")
                continue

            choices = []
            for ch in matches[:40]:
                icon = "✅" if ch.id in translated_map else "⏳"
                num_label = f"Ch.{int(ch.chapter_number)}" if ch.chapter_number is not None else ""
                choices.append(f"{icon} {num_label} {ch.title} | {ch.id}")
            choices.append("⬅️ Back")

            chosen = questionary.select(f"Found {len(matches)} matches. Pick one:", choices=choices).ask()
            if not chosen or chosen == "⬅️ Back":
                continue
            cid = chosen.split("|")[-1].strip()
            return next((c for c in chapters if c.id == cid), None)

        elif method == "⏳ Show only Untranslated chapters":
            untranslated = [c for c in chapters if c.id not in translated_map]
            if not untranslated:
                console.print("[green]All chapters are already translated![/green]")
                continue

            choices = []
            for ch in untranslated[:40]:
                num_label = f"Ch.{int(ch.chapter_number)}" if ch.chapter_number is not None else ""
                choices.append(f"⏳ {num_label} {ch.title} | {ch.id}")
            choices.append("⬅️ Back")

            chosen = questionary.select(f"Select from {len(untranslated)} untranslated:", choices=choices).ask()
            if not chosen or chosen == "⬅️ Back":
                continue
            cid = chosen.split("|")[-1].strip()
            return next((c for c in chapters if c.id == cid), None)

        elif method == "📄 Browse paginated chapters (1-50, 51-100...)":
            chunk_size = 30
            num_chunks = (len(chapters) + chunk_size - 1) // chunk_size
            page_choices = []
            for p in range(num_chunks):
                start = p * chunk_size + 1
                end = min((p + 1) * chunk_size, len(chapters))
                page_choices.append(f"Chapters {start} to {end}")
            page_choices.append("⬅️ Back")

            chosen_page = questionary.select("Select range:", choices=page_choices).ask()
            if not chosen_page or chosen_page == "⬅️ Back":
                continue

            p_idx = page_choices.index(chosen_page)
            sub_chapters = chapters[p_idx * chunk_size : (p_idx + 1) * chunk_size]

            chap_choices = []
            for ch in sub_chapters:
                icon = "✅" if ch.id in translated_map else "⏳"
                num_label = f"Ch.{int(ch.chapter_number)}" if ch.chapter_number is not None else ""
                chap_choices.append(f"{icon} {num_label} {ch.title} | {ch.id}")
            chap_choices.append("⬅️ Back")

            chosen_ch = questionary.select("Select chapter:", choices=chap_choices).ask()
            if not chosen_ch or chosen_ch == "⬅️ Back":
                continue
            cid = chosen_ch.split("|")[-1].strip()
            return next((c for c in chapters if c.id == cid), None)


@app.command()
def info(
    url: str = typer.Argument(..., help="URL of the novel on Webnovel or supported sites"),
):
    """Fetch and display metadata and chapter catalog of a novel."""
    pipeline = NovelPipeline()

    async def _run():
        with console.status("[bold green]Scraping novel metadata..."):
            novel = await pipeline.scrape_novel(url)

        console.print(
            Panel(
                f"[bold cyan]{novel.title}[/bold cyan]\n"
                f"[dim]ID:[/dim] {novel.id}\n"
                f"[dim]Source:[/dim] {novel.source}\n"
                f"[dim]Author:[/dim] {novel.author or 'Unknown'}\n\n"
                f"[bold]Synopsis:[/bold]\n{novel.synopsis or 'No synopsis available'}",
                title="Novel Information",
            )
        )

        storage = StorageManager()
        translated_map = storage.get_translated_chapter_ids(novel.id, target_lang="th")

        table = Table(title=f"Chapters Catalog ({len(novel.chapters)} total, {len(translated_map)} translated into Thai)")
        table.add_column("#", style="dim", width=6)
        table.add_column("Chapter Title", style="cyan")
        table.add_column("Status (Thai)", style="bold")
        table.add_column("Locked", style="yellow")
        table.add_column("URL", style="dim")

        for i, ch in enumerate(novel.chapters[:15], 1):
            is_trans = ch.id in translated_map
            status_str = f"[green]✅ Translated[/green]" if is_trans else "[dim]⏳ Not translated[/dim]"
            table.add_row(
                str(int(ch.chapter_number) if ch.chapter_number is not None else i),
                ch.title,
                status_str,
                "🔒 Yes" if ch.is_locked else "No",
                ch.url,
            )

        console.print(table)
        if len(novel.chapters) > 15:
            console.print(f"[dim]... and {len(novel.chapters) - 15} more chapters saved in catalog.[/dim]")

    asyncio.run(_run())


@app.command()
def chapter(
    url: str = typer.Argument(..., help="URL of the chapter to scrape"),
    translate_to: str = typer.Option(
        "th", "--translate-to", "-t", help="Target language code (default: th)"
    ),
    translator: str = typer.Option(
        "openrouter", "--translator", help="Translator backend: 'openrouter', 'gemini', or 'dummy'"
    ),
    model: str = typer.Option(
        "google/gemma-4-31b-it:free", "--model", "-m", help="Model name (e.g. google/gemma-4-31b-it:free, z-ai/glm-5.2:free)"
    ),
    auto_learn: bool = typer.Option(
        True, "--auto-learn/--no-auto-learn", help="Automatically learn new terms/characters per series"
    ),
    agentic: bool = typer.Option(
        True, "--agentic/--no-agentic", help="Enable Agentic Editor pass for polished Thai prose and alias alignment"
    ),
):
    """Scrape a chapter, translate it with self-learning glossary, and save to disk."""
    pipeline = NovelPipeline()

    async def _run():
        mode_desc = "Agentic 2-Pass (Draft + Polish)" if agentic else "Standard Draft"
        with console.status(f"[bold green]Scraping & Translating into {translate_to} using {translator} ({model}) [{mode_desc}]..."):
            kwargs = {"model": model} if translator in ("openrouter", "gemini") else {}
            chap = await pipeline.scrape_and_translate_chapter(
                url=url,
                target_lang=translate_to,
                translator_name=translator,
                translator_kwargs=kwargs,
                auto_learn=auto_learn,
                agentic_mode=agentic,
            )

        title = chap.translated_title or chap.title
        console.print(
            Panel(
                f"[bold green]{title}[/bold green]\n"
                f"[dim]Book ID:[/dim] {chap.book_id} | [dim]Chapter ID:[/dim] {chap.id}\n"
                f"[dim]Total Paragraphs:[/dim] {len(chap.paragraphs)}\n"
                f"[dim]Language:[/dim] {chap.source_language} -> {chap.target_language}\n\n"
                f"[bold]First 3 Paragraphs Preview:[/bold]\n"
                + "\n\n".join(
                    f"[{p.index}] {p.translated_text or p.text}" for p in chap.paragraphs[:3]
                ),
                title="Chapter Download & Translation Complete",
            )
        )
        console.print(f"[bold cyan]Saved to:[/bold cyan] output/{chap.book_id}/chapters/chapter_{chap.id}.[json/md]")

    asyncio.run(_run())


@app.command()
def list_series(
    target_lang: str = typer.Option("th", "--lang", "-l", help="Language code (default: th)"),
):
    """List all tracked novel series and their translation progress."""
    storage = StorageManager()
    novels = storage.list_saved_series()
    if not novels:
        console.print("[yellow]No series tracked yet. Use 'vox-novel info <url>' to scrape a novel first![/yellow]")
        return

    table = Table(title=f"Saved Series & Translation Progress ({target_lang.upper()})")
    table.add_column("Series ID", style="dim")
    table.add_column("Title", style="bold cyan")
    table.add_column("Total Chapters", justify="right")
    table.add_column("Translated", justify="right", style="green")
    table.add_column("Pending", justify="right", style="yellow")
    table.add_column("Progress", justify="right", style="magenta")

    for n in novels:
        trans_map = storage.get_translated_chapter_ids(n.id, target_lang=target_lang)
        total = len(n.chapters)
        done = len(trans_map)
        pending = max(0, total - done)
        pct = (done / total * 100) if total > 0 else 0
        table.add_row(
            n.id,
            n.title,
            str(total),
            str(done),
            str(pending),
            f"{pct:.1f}%",
        )

    console.print(table)


@app.command("glossary")
def show_glossary(
    series_id: str = typer.Argument(..., help="Series Book ID (e.g. 36119734008764305)"),
    target_lang: str = typer.Option("th", "--lang", "-l", help="Language code (default: th)"),
):
    """View learned glossary terms and characters for a series."""
    km = KnowledgeManager()
    k = km.load_or_init(series_id, target_lang=target_lang)

    table_terms = Table(title=f"Series {series_id} Terms ({len(k.terms)} entries)")
    table_terms.add_column("Original (EN)", style="cyan")
    table_terms.add_column("Target Translation", style="green")
    table_terms.add_column("Category", style="yellow")
    table_terms.add_column("Notes", style="dim")

    for t in sorted(k.terms.values(), key=lambda x: x.source.lower()):
        table_terms.add_row(t.source, t.target, t.category, t.notes or "")

    table_chars = Table(title=f"Series {series_id} Characters ({len(k.characters)} entries)")
    table_chars.add_column("Name (EN)", style="cyan")
    table_chars.add_column("Name (Target)", style="green")
    table_chars.add_column("Role / Notes", style="magenta")

    for c in sorted(k.characters.values(), key=lambda x: x.name_en.lower()):
        table_chars.add_row(c.name_en, c.name_target, f"{c.role or ''} {c.notes or ''}".strip())

    console.print(table_terms)
    console.print(table_chars)


@app.command()
def ui():
    """Interactive CLI interface to select series, browse chapters, and translate."""
    storage = StorageManager()
    pipeline = NovelPipeline()

    console.print(Panel("[bold cyan]VoxNovel Interactive Hub 🎙️📖[/bold cyan]\nTranslate and track novel series with persistent lore memory.", expand=False))

    novels = storage.list_saved_series()
    series_choices = []
    for n in novels:
        done = len(storage.get_translated_chapter_ids(n.id, "th"))
        series_choices.append(f"{n.title} (ID: {n.id}) - {done}/{len(n.chapters)} translated")

    series_choices.append("➕ Add/Scrape new novel URL")
    series_choices.append("❌ Exit")

    selected_series_str = questionary.select(
        "Select a series to manage:",
        choices=series_choices,
    ).ask()

    if not selected_series_str or selected_series_str == "❌ Exit":
        return

    if selected_series_str == "➕ Add/Scrape new novel URL":
        novel_url = questionary.text("Enter novel URL (e.g. Webnovel):").ask()
        if not novel_url:
            return
        with console.status("[bold green]Scraping novel metadata and catalog..."):
            novel = asyncio.run(pipeline.scrape_novel(novel_url))
        console.print(f"[bold green]Successfully imported:[/bold green] {novel.title} ({len(novel.chapters)} chapters)")
    else:
        m = re.search(r"\(ID:\s*(\d+)\)", selected_series_str)
        if not m:
            return
        series_id = m.group(1)
        novel = storage.get_novel(series_id)
        if not novel:
            console.print("[red]Could not load series metadata.[/red]")
            return

    target_lang = "th"
    translated_map = storage.get_translated_chapter_ids(novel.id, target_lang=target_lang)

    while True:
        action = questionary.select(
            f"Series: {novel.title} [{len(translated_map)}/{len(novel.chapters)} translated]",
            choices=[
                "📖 Select and translate chapter (Search / Number / Untranslated)",
                "⚡ Translate next untranslated chapter",
                "📚 View chapter translation status list",
                "🧠 View/manage learned glossary",
                "⬅️ Back to Main Menu",
            ],
        ).ask()

        if action == "⬅️ Back to Main Menu" or not action:
            break

        if action == "📚 View chapter translation status list":
            table = Table(title=f"{novel.title} - Chapter Status ({target_lang.upper()})")
            table.add_column("#", width=6)
            table.add_column("Chapter Title", style="cyan")
            table.add_column("Status", style="bold")
            table.add_column("Translated Title", style="green")

            for i, ch in enumerate(novel.chapters, 1):
                is_done = ch.id in translated_map
                status = "[green]✅ Done[/green]" if is_done else "[dim]⏳ Pending[/dim]"
                trans_title = translated_map[ch.id]["title"] if is_done else "-"
                table.add_row(
                    str(int(ch.chapter_number) if ch.chapter_number is not None else i),
                    ch.title,
                    status,
                    trans_title,
                )
            console.print(table)

        elif action == "🧠 View/manage learned glossary":
            show_glossary(novel.id, target_lang=target_lang)

        elif action == "⚡ Translate next untranslated chapter":
            selected_chapter = None
            for ch in novel.chapters:
                if ch.id not in translated_map:
                    selected_chapter = ch
                    break
            if not selected_chapter:
                console.print("[green]All cataloged chapters for this series are already translated![/green]")
                continue

            _execute_translation(pipeline, storage, novel, selected_chapter, target_lang)
            translated_map = storage.get_translated_chapter_ids(novel.id, target_lang=target_lang)

        elif action.startswith("📖 Select and translate chapter"):
            selected_chapter = pick_chapter_interactive(novel.chapters, translated_map)
            if not selected_chapter:
                continue

            _execute_translation(pipeline, storage, novel, selected_chapter, target_lang)
            translated_map = storage.get_translated_chapter_ids(novel.id, target_lang=target_lang)


def _execute_translation(pipeline, storage, novel, selected_chapter, target_lang):
    translator_choice = questionary.select(
        f"Translate '{selected_chapter.title}' using:",
        choices=[
            "OpenRouter (google/gemma-4-31b-it:free) [Recommended]",
            "OpenRouter (z-ai/glm-5.2:free)",
            "OpenRouter (google/gemma-4-31b-it:free)",
            "Google Gemini (gemini-2.5-flash)",
            "Dummy / Mock (Instant test)",
        ],
    ).ask()

    if not translator_choice:
        return

    translator_name = "openrouter"
    model_name = "google/gemma-4-31b-it:free"

    if "Dummy" in translator_choice:
        translator_name = "dummy"
    elif "Gemini" in translator_choice:
        translator_name = "gemini"
        model_name = "gemini-2.5-flash"
    elif "glm-5.2" in translator_choice:
        model_name = "z-ai/glm-5.2:free"
    if "gemma-4-31b" in translator_choice:
        model_name = "google/gemma-4-31b-it:free"

    agentic_mode = True
    if translator_name != "dummy":
        agentic_mode = questionary.confirm(
            "Enable Agentic Editor pass? (Polishes Thai prose, eliminates stiff passive phrasing & strictly enforces aliases)",
            default=True,
        ).ask()
        if agentic_mode is None:
            return

    mode_label = "Agentic Polish" if agentic_mode else "Single Pass"
    with console.status(f"[bold green]Scraping & Translating into Thai ({selected_chapter.title}) [{mode_label}]..."):
        kwargs = {"model": model_name} if translator_name in ("openrouter", "gemini") else {}
        chap = asyncio.run(
            pipeline.scrape_and_translate_chapter(
                url=selected_chapter.url,
                target_lang=target_lang,
                translator_name=translator_name,
                translator_kwargs=kwargs,
                auto_learn=True,
                agentic_mode=agentic_mode,
            )
        )

    console.print(
        Panel(
            f"[bold green]{chap.translated_title or chap.title}[/bold green]\n"
            f"[dim]Chapter ID:[/dim] {chap.id}\n"
            f"[dim]Total Paragraphs:[/dim] {len(chap.paragraphs)}\n"
            f"[dim]Saved to:[/dim] output/{chap.book_id}/chapters/chapter_{chap.id}.md\n\n"
            f"[bold]First 3 Paragraphs:[/bold]\n"
            + "\n\n".join(f"[{p.index}] {p.translated_text or p.text}" for p in chap.paragraphs[:3]),
            title="Translation Finished",
        )
    )


if __name__ == "__main__":
    app()


@app.command("web")
def run_web(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host address to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port number"),
    reload: bool = typer.Option(True, "--reload/--no-reload", help="Enable auto-reloading"),
):
    """Launch the VoxNovel Web UI Dashboard and Reader."""
    import uvicorn
    console.print(Panel(f"[bold cyan]VoxNovel Web UI[/bold cyan]\nRunning on: [bold green]http://{host}:{port}[/bold green]", expand=False))
    uvicorn.run("vox_novel.web.app:web_app", host=host, port=port, reload=reload)


def _review_entities_cli(new_terms: list, new_chars: list, knowledge: SeriesKnowledge) -> bool:
    """CLI prompt to review new entities interactively before translating."""
    console.print(
        Panel(
            "[bold yellow]New Entities Detected in this Chapter![/bold yellow]\n"
            "Reviewing these terms ensures smooth and consistent translation across the series.",
            title="Pre-Translation Review",
        )
    )

    if new_chars:
        table_c = Table(title=f"New Characters ({len(new_chars)})")
        table_c.add_column("Character (EN)", style="cyan")
        table_c.add_column("Suggested Translation", style="green")
        table_c.add_column("Role", style="magenta")
        for c in new_chars:
            table_c.add_row(c.get("name_en", ""), c.get("suggested_target", ""), c.get("role", "") or "-")
        console.print(table_c)

    if new_terms:
        table_t = Table(title=f"New Terms / Skills ({len(new_terms)})")
        table_t.add_column("Term (EN)", style="cyan")
        table_t.add_column("Suggested Translation", style="green")
        table_t.add_column("Category", style="yellow")
        for t in new_terms:
            table_t.add_row(t.get("source", ""), t.get("suggested_target", ""), t.get("category", "general"))
        console.print(table_t)

    action = questionary.select(
        "How would you like to proceed?",
        choices=[
            "✅ Accept all suggested terms & translate",
            "✏️ Edit or add specific terms first",
            "⏭️ Skip review & translate directly",
            "❌ Cancel translation",
        ],
    ).ask()

    if action == "❌ Cancel translation" or not action:
        return False

    if action == "✅ Accept all suggested terms & translate":
        for c in new_chars:
            knowledge.add_character(c["name_en"], c["suggested_target"], role=c.get("role"))
        for t in new_terms:
            knowledge.add_term(t["source"], t["suggested_target"], category=t.get("category", "general"))

    elif action == "✏️ Edit or add specific terms first":
        for c in new_chars:
            custom_tgt = questionary.text(
                f"Translation for character '{c['name_en']}':", default=c.get("suggested_target", "")
            ).ask()
            if custom_tgt:
                knowledge.add_character(c["name_en"], custom_tgt, role=c.get("role"))

        for t in new_terms:
            custom_tgt = questionary.text(
                f"Translation for term '{t['source']}':", default=t.get("suggested_target", "")
            ).ask()
            if custom_tgt:
                knowledge.add_term(t["source"], custom_tgt, category=t.get("category", "general"))

    return True


@app.command(name="tts")
def tts_chapter_cmd(
    series_id: str = typer.Argument(..., help="Series ID (e.g. 36119734008764305)"),
    chapter_id: str = typer.Argument(..., help="Chapter ID"),
    engine: str = typer.Option("voxcpm2", "--engine", "-e", help="TTS engine (voxcpm2, dummy)"),
    voice: Optional[str] = typer.Option(None, "--voice", "-v", help="Voice design prompt description"),
    api_url: Optional[str] = typer.Option(None, "--api-url", help="Remote VoxCPM2 / vLLM-Omni API URL"),
    device: Optional[str] = typer.Option(None, "--device", help="Device (auto, mps, cuda, cpu)"),
):
    """Synthesize chapter into Thai audiobook using VoxCPM2 TTS."""
    pipeline = NovelPipeline()

    kwargs = {}
    if api_url:
        kwargs["api_url"] = api_url
    if device:
        kwargs["device"] = device

    async def _run():
        console.print(Panel(
            f"[bold magenta]🎙️ VoxNovel Audio Synthesis[/bold magenta]\n"
            f"[bold]Series ID:[/bold] {series_id}\n"
            f"[bold]Chapter ID:[/bold] {chapter_id}\n"
            f"[bold]Engine:[/bold] {engine}\n"
            f"[bold]Voice Design:[/bold] {voice or 'Default Narrator Voice'}",
            title="VoxCPM2 Audiobook Generator",
            border_style="magenta",
        ))

        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating audiobook...", total=100)

            async def progress_cb(pct: int, msg: str):
                progress.update(task, completed=pct, description=f"[cyan]{msg}[/cyan]")

            try:
                audio_file = await pipeline.synthesize_chapter_audio(
                    series_id=series_id,
                    chapter_id=chapter_id,
                    engine_name=engine,
                    voice_description=voice,
                    progress_callback=progress_cb,
                    engine_kwargs=kwargs,
                )
                progress.update(task, completed=100, description="[bold green]Complete![/bold green]")
                console.print(f"\n[bold green]✅ Audio saved to:[/bold green] {audio_file}")
            except Exception as e:
                console.print(f"\n[bold red]❌ Synthesis failed:[/bold red] {e}")

    asyncio.run(_run())

