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
| qwen3:14b | 87% | 100% |
| typhoon2.5-qwen3-30b | 83% | 97% |
| qwen3:8b | 73% | 80% |

**Read the absolute numbers with care.** Those figures come from a single
unrepeated run of n=30 on the chapter the prompt was tuned against, and the
curated registry was written *after* observing the errors -- it tells the model
outright that one character is "asleep throughout this chapter, do not identify
her as the speaker of any dialogue" and that four others are "only mentioned".
Five of its ten entries carry an instruction of that shape. So 100% describes a
ceiling on a burned evaluation set, not what to expect on unseen text. Chapter
164 cannot be used to measure generalisation by either implementation.

What survives is the *gap*, which is implementation-independent. See the
measurements section below.

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

## Measured after steps 1-5

Same gold set, same model (qwen3:14b on a local Ollama), scored the same way:

| arm | prototype | this port |
| --- | --- | --- |
| auto-built registry | 87% | 70% |
| curated registry | 100% | 83% |
| **delta** | **+13** | **+13** |

The delta reproduces exactly, on a different implementation, and for the same
reason: auto-extraction flagged no narrator, and six of eight errors were the
narrator's lines going to the most prominent character present. That is the
durable finding and it is what justifies step 6.

The absolute numbers are not comparable. The prototype's curated run passed the
chapter-specific "does not speak here" hints to the model; this port's 83% run
did not pass character descriptions at all, so it is the cleaner measurement of
the two. Adding descriptions afterwards gave 80% -- within noise.

Two hypotheses were tested and rejected, both by measurement:

| change | result |
| --- | --- |
| echo each paragraph's text verbatim, as the prototype does | 77%, and 74% slower |
| send character descriptions in the registry block | 80% |

Three runs of one configuration gave 83 / 77 / 80, so roughly +/-3 points is
noise at n=30 and neither change is distinguishable from it. Dropping the text
echo was therefore right on both accuracy and its 41% output-cost saving.

Runtime: ~2.5 minutes per 232-paragraph chapter, full coverage, no gaps.

### Model choice, and what agreement-gating is actually for

Same gold set, curated registry:

| configuration | right voice | wrong voice | narrator | time |
| --- | --- | --- | --- | --- |
| qwen3:14b alone | 25 | 5 | 0 | 148s |
| qwen3:14b, confirmed by qwen3.8 | 24 | 0 | 6 | 434s |
| **qwen3.8 alone** | **30** | **0** | **0** | **221s** |

The gate converts wrong voices into narrator fallbacks, which is the right trade
when a model cannot be trusted -- a fallback is inaudible, a wrong character voice
is not. But it is not a substitute for a better model. Against qwen3:14b it looked
like a large win; against qwen3.8 it is strictly worse, costing six correct
attributions and twice the time to protect against errors the stronger model does
not make.

So: pick the best model available, and reach for --confirm-with only when the
primary is known to be weak, or on a series where a wrong voice is more costly
than a missed one. Do not read the 30/30 as an accuracy estimate -- it is the same
burned chapter, and a perfect score there says more about contamination than
about capability.

**No number here measures generalisation.** Both implementations' prompts derive
from one tuned on this chapter. A held-out set -- ideally a chapter of the
series actually being produced -- is needed before trusting an absolute figure.

## Held-out measurement

Chapter 68 of `novel-mypossessionbecameaghoststory`, a series in production and
not the chapter any prompt was tuned against. qwen3.8 on a local Ollama, single
model, no agreement gate. Every spoken line reviewed by a human through the
review page, so this is 45 labels rather than a spot check:

| | |
| --- | --- |
| spoken lines | 45 |
| right | 42 |
| wrong | 3 |
| **accuracy** | **93.3%** |

**What this does and does not measure.** The prompt is genuinely held out --
it derives from chapter 164 of a different series. The registry is not: it was
built by `--extract` on chapter 68 itself, and its `narration_note` spells out
that chapter's mid-chapter POV shift and names the first-person narrator
outright. So 93.3% is the accuracy of *prompt on unseen text, with a registry
extracted from that text*, which is the configuration the pipeline actually
runs. It is not a measurement of transfer to a chapter the registry has never
seen.

Chapter 68 is now burned for future runs -- detection has been re-run on it, and
its labels informed the analysis below.

### Does the registry transfer?

Chapter 169, a different arc of the same series, run with the registry built
from chapter 68 and no `--extract`. Reviewed the same way:

| | chapter 68 | chapter 169 |
| --- | --- | --- |
| spoken lines | 45 | 49 |
| right | 42 | 49 |
| wrong | 3 | 0 |
| accuracy | 93.3% | 100% |

Combined, 91 of 94 spoken lines across two arcs: 96.8%.

**One extraction per series is enough.** That was the open question, and the
answer decides real money -- per-chapter extraction would have roughly doubled
the LLM calls per chapter. A registry built from one chapter lost nothing on a
chapter it had never seen.

Read the 100% as "no worse than 93.3%", not as a ceiling. It is one chapter,
n=49, and no line was corrected, so nothing in the run distinguishes a perfect
model from a fast review. The transfer conclusion holds under either reading,
which is why it is the part worth acting on.

The narrator leak below did not recur here: `เอวานเจลีน` narrates chapter 169 and
none of her nine spoken lines went astray.

### The remaining errors are one class

All three misses are a two-line exchange where the narrator speaks first and
another character answers. The model gave the narrator's line to whichever
character was named in the adjacent narration:

| line | next line | name in nearby narration | attributed to |
| --- | --- | --- | --- |
| 2 | 3 | มาดามโทเทน | มาดามโทเทน |
| 7 | 8, Toten answering | เดซี่ | เดซี่ |
| 81 | 80 names Daisy | เดซี่ | เดซี่ |

The narrator is the one character never named in the prose, because she is
`ฉัน`, so proximity to a name pulls every unattributed line away from her. This
is the same failure as the 87% run on chapter 164 and the same as the errors the
curated registry fixed there -- registry curation raised it from 70% to 83% but
did not remove it, and here the registry is already correct: `เอวานเจลีน` is
flagged narrator and the narration note is explicit.

In all three cases the *reply* was attributed correctly. Turn-taking -- adjacent
dialogue lines alternate unless the text says otherwise -- would have resolved
every one.

Left unfixed for now. Chapter 169 showed the leak does not always happen, so
changing the prompt against three errors on one chapter risks fitting to them.
A third chapter exhibiting it is the trigger.

### Cost, measured

Chapter 169, per detection run:

| | chars |
| --- | --- |
| chapter text | 12,362 |
| registry block, resent per chunk | 5,532 |
| chunks | 4 |
| registry total | 22,128 |

The registry costs 1.79x what the content costs. Free on the local Ollama this
was measured on, and the first thing to fix if detection ever runs against a
paid endpoint. `_scoped_registry` already trims by chapters seen; the larger win
is that seven of the fourteen entries carry chapter-specific commentary from the
`--extract` run ("Not present in this chapter. Mentioned as..."), which is both
bloat and misleading input on every other chapter.

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

1. At ~80%, roughly one dialogue line in five gets the wrong voice -- which is
   worse for a listener than no per-character voices at all, since the narrator
   reading everything is at least consistent. Three ways to make that shippable,
   not mutually exclusive: review before synthesis (step 6); raise accuracy; or
   fail toward the narrator, so an uncertain line sounds exactly as it does
   today instead of wrong. Confidence cannot drive the third -- models report 1.0
   while wrong -- but agreement between two models can, and a second local model
   costs only time.
2. `thought` — keep as a third type, or collapse to `dialogue`?
3. Run speaker detection automatically after translation, or on demand like
   synthesis?
4. Local model via `OPENROUTER_BASE_URL` pointing at Ollama, which the prototype
   benchmarked at 100% with a curated registry and no per-chapter cost?
