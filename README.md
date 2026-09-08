# VoxNovel 🎙️📖

A modular, extensible pipeline to **scrape**, **translate** (with self-improving memory/glossaries), and synthesize web novels into **audiobooks (TTS)**.

---

## 🌟 Features

1. **Default Thai Translation**: Fine-tuned for fantasy/gaming light novels with strict preservation of gaming terminology, Korean/foreign name transliterations, and light novel dialogue nuance.
2. **Interactive CLI (`vox-novel ui`)**:
   - Browse tracked series and chapter translation status (`✅ Translated` vs `⏳ Pending`).
   - Select and translate individual chapters interactively.
   - One-click **"Translate next untranslated chapter"** to read through series sequentially.
3. **OpenRouter Support**: Works out of the box on a free model (`google/gemma-4-31b-it:free`). Free tiers are capped at 50 requests/day and cannot translate a whole novel — see [docs/model-selection.md](docs/model-selection.md) for measured costs and model choices.
4. **Self-Improving Series Glossary**: Auto-detects and records newly introduced characters, skills, item names, and locations into `output/<series_id>/knowledge_th.json`.
5. **Clean Dual Storage**:
   - `output/<series_id>/chapters/chapter_<id>.json` (Full metadata + 1:1 paragraph indices for future TTS)
   - `output/<series_id>/chapters/chapter_<id>.md` (Human-readable Thai markdown)

---

## 🚀 Usage

### 1. Interactive Hub (Recommended)
Launch the interactive terminal UI:
```bash
uv run vox-novel ui
```
You can:
- Select any tracked novel or paste a new novel URL
- See how many chapters are translated vs pending
- Pick any chapter to translate or let it auto-translate the next pending chapter
- Inspect and manage the series' learned glossary

### 2. Series & Status Overview
```bash
uv run vox-novel list-series
```

### 3. Check Chapters Catalog & Status for a Novel
```bash
uv run vox-novel info "https://www.webnovel.com/book/illusion-hunter-from-another-world_36119734008764305"
```

### 4. Direct Chapter Translation
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
uv run vox-novel chapter "https://www.webnovel.com/book/illusion-hunter-from-another-world_36119734008764305/chapter-100.-unavoidable-malice_97077461090620248"
```
*(Default target language is **Thai (`th`)** and default translator is **OpenRouter** with `google/gemma-4-31b-it:free`).*

### 5. View Series Glossary / Lore Memory
```bash
uv run vox-novel glossary 36119734008764305
```

---

## 🌐 Modern Web UI Dashboard & Reader

VoxNovel comes with a built-in modern dark-mode Web UI built on **FastAPI + TailwindCSS + HTMX**.

### Launch Web UI:
```bash
uv run vox-novel web
```
Then open: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

### What you can do in the Web UI:
- 📊 **Dashboard (`/`)**: Visual grid of all tracked series, cover art, chapter counts, and translation % progress bars. One-click "Import Novel" modal for new Webnovel URLs.
- 📑 **Series Management (`/series/<id>`)**:
  - Live chapter catalog table with instant search/filter bar.
  - See status badges (`✅ Translated` vs `⏳ Pending`).
  - One-click **"Translate Next"** button to advance your reading sequentially.
  - Direct links to **"Read"** translated chapters or trigger re-translations.
- 📖 **Immersive Reader (`/series/<id>/read/<chapter_id>`)**:
  - Clean, distraction-free reading mode optimized for Thai typography (`Sarabun` / system font stack).
  - Previous / Next chapter navigation buttons.
- 🧠 **Series Glossary & Lore Hub (`/series/<id>/glossary`)**:
  - View all registered and self-learned character names and terminology.
  - Interactive form to add custom terms, character names, or corrections on the fly.

---

## 🧐 Pre-Translation Entity & Vocabulary Review

To ensure translation quality remains high and never breaks immersion:

1. **Pre-Scan**: Before translating any chapter, VoxNovel scans the source English text against your existing series glossary.
2. **Detection**: Any new character names, gaming terms, skills, monsters, or locations that have never appeared before are flagged with AI-suggested Thai translations.
3. **Interactive Review**:
   - **In Web UI**: Automatically routes to a `/review` screen where you can approve, customize the Thai spelling, or untick terms before translation begins.
   - **In Terminal**: Prompts you with a table of detected entities to accept or customize inline.
4. **Permanent Memory**: Once approved, the terms are permanently saved into `output/<series_id>/knowledge_th.json` so every subsequent chapter is guaranteed to use the exact same spelling and terminology!

---

## 🧭 In-App Search & Novel Discovery

Instead of manually browsing Webnovel and copying URLs, you can search and browse directly within VoxNovel:

- **Web Search**: Navigate to `/explore` or click **"Explore & Search"** in the top navigation.
- **Search by keyword**: E.g. `Release that Witch`, `Shadow Slave`, `Hunter`, etc.
- **Browse Trending**: Browse popular trending novels directly from `https://www.webnovel.com/stories/novel`.
- **1-Click Import & Track**: Click **"Import & Track"** on any novel card to instantly fetch metadata, catalog all chapters, and start translating with your series glossary.
