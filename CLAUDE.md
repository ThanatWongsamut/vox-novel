# VoxNovel

Scrapes web novels, translates them to Thai with a self-improving glossary, and
synthesizes audiobooks with VoxCPM2.

## Before working on models, translation, or TTS voice design

Read **[docs/model-selection.md](docs/model-selection.md)** first. It records
measured findings — real billed costs, an A/B on voice control prompts, a survey
of free models — that are expensive to re-derive and counterintuitive in places.
The four that trip people up:

1. **VoxCPM2 only acts on English control prompts.** A Thai voice description is
   not interpreted as an instruction; it is spoken aloud before the content
   (~2.5x the audio length). Voice descriptions must be translated to English
   before synthesis. The model card's "30 languages" claim is about the content
   being synthesized, not the control prompt.
2. **Thai costs ~4x the tokens of English** (1.04 vs 0.19 tok/char). Cost
   estimates built on English intuition are wrong by an order of magnitude.
3. **Put the stronger model on the polish pass, not the draft.** Polish sees the
   English original alongside the Thai draft, so it can repair meaning; it also
   batches 25 paragraphs to the draft's 15, so it is cheaper to upgrade.
   `OPENROUTER_MODEL` / `OPENROUTER_POLISH_MODEL`.
4. **Free tiers cannot translate a novel** — 50 requests/day, and the upstream
   shared pool rejects translation-sized batches regardless of quota. They are
   fine for voice prompts, which are small and cached.

Never quote a model price from memory or from a list-price table. Reasoning
models burn output tokens invisibly; get real cost from
`/api/v1/generation?id=<gen_id>`.

## Commands

```bash
uv run python -m unittest discover -s tests   # full suite, hermetic, <1s
uv run vox-novel web                          # web UI on :8000
uv run vox-novel ui                           # interactive CLI
uv run vox-novel tts <series_id> <chapter_id> # synthesize a chapter
```

`pytest` is not installed; the suite is `unittest`.

## Layout

| Path | Purpose |
| --- | --- |
| `scrapers/` | Per-site scrapers behind a registry (`readtoon`, `webnovel`) |
| `translators/` | LLM backends behind a registry; also the project's LLM gateway |
| `tts/` | TTS engines behind a registry (`voxcpm2`, `dummy`) |
| `pipeline/manager.py` | Orchestrates scrape → translate → polish → synthesize |
| `storage/` | Filesystem persistence under `output/<series_id>/` |
| `web/` | FastAPI app and Jinja templates |
| `extension/` | Manifest V3 Chrome importer for ReadToon |

## Conventions

- **Paragraph indices are 1-based** and stable across translation. TTS audio
  chunks are named `para_{chapter_id}_{paragraph.index}.wav` and the
  `/api/audio/.../para/{i}` endpoint resolves them by that index. Keep the two
  in agreement.
- **Untrusted ids are validated before touching the filesystem.** `series_id`
  and `chapter_id` arrive from request bodies and path parameters and become
  directory names — route them through `validate_path_segment`.
- **CORS is an explicit origin allowlist, never `*`.** The Chrome importer runs
  as a content script, so its requests carry the page origin. A wildcard with
  credentials would let any visited site read this server.
- **The narrator anchor is content-addressed** — `voices/narrator_ref.<sha8>.wav`,
  named for the control prompt that produced it and never rewritten. Every
  paragraph clones from it, so a mutable shared file meant a one-off `--voice` run
  or a second concurrent job could silently re-voice a series mid-book. A different
  voice is a different file; nothing needs to detect staleness.
- `voxcpm` is an optional `tts` extra and is **pinned**, because
  `VoxCPM2TTS._patch_voxcpm_inference` replaces an upstream method with a copy of
  that specific version's implementation. The patch verifies the installed
  version and refuses to apply otherwise.
- Commits and PR descriptions carry no AI attribution.

## Known gaps

- **Per-character voices are wired but inert.** `Paragraph.speaker` is never
  assigned, so `VoxCPM2TTS.synthesize_chapter`'s character branch, the
  per-speaker prompt cache and the emotion suffix cannot run. Everything
  downstream of speaker detection is built and tested; adding detection turns it
  on. Until then, every paragraph uses the narrator voice.
- **There is no CI.** The test suite only runs when someone runs it.

## Sources

ReadToon renders chapters client-side and gates paid ones behind Turnstile, so
the server cannot fetch them; the Chrome extension reads them from the browser
session instead. ReadToon content is already Thai, so the pipeline ingests it
directly and skips translation and entity pre-scan. Webnovel content is English
and goes through the full translate → polish path.
