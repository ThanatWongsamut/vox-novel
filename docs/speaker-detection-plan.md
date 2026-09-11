# Plan: speaker detection

Turns on per-character voices. Everything downstream already exists and is
tested — `control_by_speaker`, per-character reference clips, the emotion
suffix — and stays inert only because nothing assigns `Paragraph.speaker`.

Source: the `novel-to-script` prototype, which already solves this for Thai and
carries a benchmark on a real chapter.

## What the prototype establishes

Measured on chapter 164, 30 hand-labelled quote lines, RTX 3090:

| model | full pipeline | curated registry |
| --- | --- | --- |
| qwen3:14b | 87% | **100%** |
| typhoon2.5-qwen3-30b | 83% | 97% |
| qwen3:8b | 73% | 80% |

Three findings shape everything below.

**Registry quality is the bottleneck, not attribution.** Fixing one entry — the
first-person narrator — took qwen3:14b from 87% to 100%. Every residual error
without a curated registry was the same class: untagged quotes spoken by the
narrator.

**Confidence is not a usable signal.** Local models return 1.0 while wrong.
Review has to key on something else — untagged quotes, or disagreement between
two models.

**Three failures only appear over many chapters**, all already handled: the
registry freezing because models put unfamiliar names straight into `speaker`
without declaring them; name drift splitting one character across several
voices; and the registry inflating every prompt as the cast grows.

## Gap analysis

| prototype | VoxNovel | gap |
| --- | --- | --- |
| `Segment.paragraph` (0-based) | `Paragraph.index` (1-based) | off-by-one |
| `Segment.type` dialogue/thought/narration | `speech_type` dialogue/narration | no `thought` |
| `Segment.speaker` | `Paragraph.speaker` | the inert field |
| `Character.is_narrator` | — | **missing, and highest-value** |
| `Character.speech_style` | — | **missing** |
| `CharacterRegistry` | `SeriesKnowledge` | overlapping responsibilities |
| `registry.find()` | `find_character()` | VoxNovel's is weaker (see below) |
| `backend.structured()` | `translator.complete()` | no schema enforcement |

`_call_chat_completion` already accepts `response_format`, so structured output
is mostly present. Missing: the strict-schema conversion providers require
(`additionalProperties: false`, every property in `required`, no `default`),
`provider: {require_parameters: true}` so a fallback provider cannot silently
return prose, and fence-tolerant parsing.

## Decisions

### 1. One registry, not two — extend `SeriesKnowledge`

The alternative is a separate `characters.json` per series. Rejected: two
character lists that can disagree is exactly how the voice-prompt bugs in the
last PR happened, and `SeriesKnowledge` already carries aliases, gender, and the
per-character voice fields attribution needs to reach.

Add to `CharacterProfile`:

- `is_narrator: bool` — the single highest-value field per the benchmark
- `speech_style: str` — pronouns and particles, e.g. `ดิฉัน/ค่ะ, formal female`
- `line_count: int`, `last_seen_chapter: str` — needed for prompt scoping

Add to `SeriesKnowledge`: `narration_note: str`.

### 2. Fix `find_character` before relying on it

`_normalize_name_key` deletes separators rather than collapsing them, so
`An Na` == `Anna` and `Li Wei` == `Liwei`. Two distinct characters silently
share one voice — plausible in Chinese and Thai web novels, and attribution will
make it far more reachable than it is today.

Port the prototype's resolution order: exact, then alias, then *unambiguous*
prefix match, with its `norm_name` that strips stray CJK tokens some models emit
into Thai names.

### 3. Attribute on the text that will be spoken

ReadToon is already Thai, so there is no choice there. For webnovel there is:
attribute on the English source, or on the Thai translation.

Attribute on **whatever `synthesize_chapter` will read** — the translated text.
Speaker indices must line up with the paragraphs that become audio chunks, and a
translation can merge or split a paragraph. Attributing on English and carrying
the mapping across adds a failure mode that produces a wrong voice rather than a
visible error.

Worth measuring later; not worth the coupling now.

### 4. A separate pass, not folded into translation

Translation already makes 7–18 LLM calls per chapter across two passes. Speaker
detection is a third concern with a different chunk size (25 paragraphs vs 15),
different context requirements (4 paragraphs of read-only overlap), and a
different failure mode.

It runs after translation and before synthesis, over `translated_text`, and
writes `speaker` and `speech_type` back onto the existing `Paragraph` objects.
Re-runnable without re-translating.

## Shape

```
src/vox_novel/speaker/
  __init__.py
  models.py      Segment, ChunkAnnotation, CharacterDraft   (ported)
  chunking.py    make_chunks                                 (ported, 1-based)
  prompts.py     EXTRACT_SYSTEM, ANNOTATE_SYSTEM             (ported)
  detector.py    extract_registry(), annotate_chapter()      (adapted)
```

Dropped from the prototype: its CLI, its three backends, its own registry
persistence. The existing translator registry supplies the LLM; `SeriesKnowledge`
supplies the registry.

Added to `translators/openrouter.py`: a `structured()` method alongside
`complete()`, carrying the strict-schema conversion and provider routing.

## Steps

1. **Structured output.** `_strict_schema`, fence-tolerant parsing,
   `provider.require_parameters`, and `BaseTranslator.structured()` raising
   `NotImplementedError` so a backend without it degrades like `complete()` does.
   Test against a live model with a trivial schema.
2. **Registry fields.** `is_narrator`, `speech_style`, `line_count`,
   `last_seen_chapter`, `narration_note`. Existing `knowledge_th.json` files load
   unchanged — all default.
3. **Fix `find_character`.** Port `norm_name` and the exact/alias/prefix order.
   Test the `An Na` / `Anna` collision explicitly.
4. **Port the detector.** `models.py`, `chunking.py`, `prompts.py`, and the parts
   of `pipeline.py` that matter: chunking with overlap, the repair pass for
   skipped paragraphs, auto-registration of unseen speakers, canonicalisation,
   and `scoped()` prompt trimming.
5. **Wire into the pipeline.** `detect_speakers(series_id, chapter_id)` on
   `NovelPipeline`; a `POST /api/detect-speakers` job mirroring
   `/api/synthesize-chapter`; a CLI `vox-novel speakers`.
6. **Registry review UI.** The benchmark says this is where accuracy is won. The
   glossary page already edits characters — add `is_narrator` and `speech_style`,
   and surface `near_duplicates()`.
7. **Turn the voice path on.** Remove the "inert" markers, and fix what the last
   review found in that branch: `para_ref_audio` currently clones a character
   with a description but no clip from the *narrator* anchor, so their
   description never reaches the model.

Steps 1–4 are independently testable without touching synthesis. Step 7 is where
audio changes.

## Risks

**Attribution quality depends on a registry we do not curate.** VoxNovel builds
its glossary as a side effect of translation; the prototype builds it with a
dedicated extraction pass and assumes a human reviews it. Shipping step 5 without
step 6 lands nearer 73–87% than 100%, and the errors are silent — a line read in
the wrong voice.

**Cost.** Roughly 4 chunks per 100-paragraph chapter, plus a repair pass, plus
extraction once per series. Comparable to one translation pass. Should be
measured on a real chapter before it runs over a novel.

**`thought` has nowhere to go.** VoxNovel's `speech_type` is dialogue or
narration. Either add `thought` and decide how it is voiced — the prototype
treats it as attributed speech — or collapse it to `dialogue` on import and lose
the distinction.

**Paragraph indices must line up exactly.** Audio chunks are
`para_{chapter_id}_{index}.wav` and the reader resolves them by that index. An
off-by-one here misattributes every voice in the chapter. The prototype is
0-based; VoxNovel is 1-based.

## Open questions

1. Is a curated-registry workflow acceptable, or must this be fully automatic?
   The benchmark gap is 87% vs 100%.
2. `thought` — keep as a third type, or collapse to `dialogue`?
3. Run speaker detection automatically after translation, or on demand like
   synthesis?
4. Local model via `OPENROUTER_BASE_URL` pointing at Ollama, which the prototype
   benchmarked at 100% with a curated registry and no per-chapter cost?
