# Model selection: measured findings

Research notes from choosing models for translation and TTS voice design.
Everything here was measured, not inferred from documentation. Re-verify before
trusting any number — providers change models and pricing without notice.

**Measured:** 2026-09-08/09
**Test material:** `Chapter 100. Unavoidable Malice` from series `36119734008764305`
(232 paragraphs, English source), paragraphs 46–60, with that series' real
42-term / 7-character glossary.

---

## 1. VoxCPM2 only acts on English control prompts

This is the single most important finding and the least obvious.

VoxCPM2 takes a voice description in parentheses ahead of the text —
`(warm female voice)สวัสดี` — but **interprets it only in English**. A Thai
control prompt is not treated as an instruction; it is spoken aloud before the
content.

Same Thai sentence, two runs per arm:

| control prompt | audio length | resulting voice |
| --- | --- | --- |
| none | 3.2s / 3.4s | female (the default for this text) |
| English | 3.5s / 4.3s | **male, as instructed** |
| Thai | 8.8s / 8.3s | female — unchanged, and ~5s longer |

The extra ~5s is the description being read out. Confirmed by ear, not just by
duration: only the English arm produced the requested elderly male voice.

**Consequence:** any non-Latin text reaching a control prompt becomes audible
garbage at the start of every paragraph. `VoxCPM2TTS.build_control_prompt` must
return ASCII; `tests/test_voxcpm_control_prompt.py` enforces this.

The model card's "30 languages, input text in any supported language directly"
refers to the **content** being synthesized, not the control prompt. Do not
re-derive this from the card — it reads as though Thai should work.

## 2. Thai costs ~4x the tokens of English

```
EN  88 chars ->  17 tokens  (0.19 tok/char)
TH  67 chars ->  70 tokens  (1.04 tok/char)
```

Measured with `cl100k_base`. Output is Thai and output is the expensive side, so
naive English-based cost estimates understate real cost by roughly an order of
magnitude. An early estimate here was wrong by 12–20x for exactly this reason.

## 3. Reasoning models waste most of their output budget

Output tokens per character of returned Thai (natural rate is ~1.0):

| model | tok/char | note |
| --- | --- | --- |
| `deepseek-v3.2` | 0.52 | emits **zero** reasoning tokens |
| `deepseek-v4-flash` | 2.67 | |
| `deepseek-v4-pro` | 3.01 | |
| `minimax-m3` | 4.05–7.17 | |
| `deepseek-v4-flash-0731` | **16.60** | 9,892 reasoning tokens for 616 chars |

Translation needs no reasoning, so this is pure cost. It also makes list prices
misleading: `v4-flash-0731` has the cheapest input rate of the flash tier and
still costs 2x `v4-flash` per result, takes 334s against 27s, and produces
visible artifacts (Latin full stops on 7/15 Thai paragraphs, corrupted text such
as `ที่รัวิ่งวิ่ง`).

**Billed cost varies run to run.** The same Pro draft pass on identical input
billed $0.0067 and $0.0109 in two configurations — reasoning models decide how
much to think each time. Budget with ~50% headroom.

## 4. Real per-chapter cost

Billed by OpenRouter's `/api/v1/generation` endpoint, not computed from list
prices. Scaled to a 232-paragraph chapter (16 draft batches of 15, 10 polish
batches of 25, plus pre-scan and title).

| draft | polish | per chapter | 150-chapter novel |
| --- | --- | --- | --- |
| flash | flash | $0.014 | $2.09 |
| flash | **pro** | $0.097 | **$14.62** |
| pro | flash | $0.117 | $17.52 |
| pro | pro | $0.206 | $30.85 |

**Put the stronger model on polish, not draft.** It is both cheaper and more
effective:

- Cheaper: polish batches 25 paragraphs to the draft's 15, so 10 calls per
  chapter against the draft's 16.
- More effective: polish receives the English original alongside the Thai draft
  (`Draft: … / Original: …`), so it can repair meaning rather than only smooth
  phrasing, and its prompt targets naturalness — which is what separates these
  models on Thai prose.

Configure with `OPENROUTER_MODEL` and `OPENROUTER_POLISH_MODEL`.

**Cost driver worth attacking:** the glossary is resent on every call — 2,170
tokens across 26 calls per chapter on the test series. Sending only entries
whose terms appear in each batch would cut input cost noticeably at any tier.

## 5. Free tiers cannot translate a novel

Two independent blockers, either one disqualifying.

**Quota.** An account that has never purchased credits gets 50 requests/day
across all `:free` models. A chapter costs 7–18 calls, so 2–7 chapters/day.
Purchasing $10 once raises this to 1,000/day permanently.

**Shared-pool congestion.** Independent of quota, and worse. A 15-paragraph
batch to `google/gemma-4-31b-it:free` failed through six rounds of the
pipeline's own backoff (4 retries at 8/16/24s each) while single-token requests
to the same model kept succeeding. The error is
`limit_source: upstream_provider_shared_pool`, not an account limit. No retry
logic fixes contention you do not control.

Free tiers **are** fine for TTS voice prompts: one small call per character,
cached, with a keyword-table fallback.

## 6. Free model survey

Of the 19 models with $0 pricing at time of writing, most were unusable for
translation or voice prompts:

| model | outcome |
| --- | --- |
| `thinkingmachines/inkling*` | HTTP 403 — agentic harnesses only |
| `nvidia/nemotron-3.5-lightning` | leaks chain-of-thought into the answer |
| `nvidia/nemotron-3-ultra-550b` | `"boy"` — drops nearly every attribute |
| `dots-studio/dots-3-note-preview` | `หัวหน้าจันทร์` ("moon chief") for "moonlit" |
| `openrouter/free` | a moderation router: `"User Safety: safe"` |
| `poolside/laguna-s-2.1` | coding agent, not a translator |
| **`google/gemma-4-31b-it:free`** | **handled both tasks** — current default |

`minimax/minimax-m3:free` returns HTTP 404: the free variant was withdrawn
though the paid slug and the model page both still exist. A resolving model page
does not mean a routable slug — check `/api/v1/models`.

That withdrawal is the second one this project has hit. Expect free variants to
disappear; a dead default breaks translation *and* voice prompts.

## 7. Local hosting (RTX 3090, 24GB)

Computed, not measured — no 3090 was available.

`gemma-4-31b` at Q4_K_M: 17.8GB weights, ~5.2GB left for KV cache (~20k context,
against a ~6k need per batch). ~34 tok/s → ~31 min/chapter, ~4 days for a
200-chapter novel. No quota, no shared pool, no per-chapter cost.

Two caveats:

- Q4 quantization degrades low-resource languages more than English. A
  Thai-specialised model at full precision (SCB10X **Typhoon**, 8B at FP16 fits
  in 16GB) may beat a general 31B at Q4. Worth testing first.
- `OpenRouterTranslator` accepts `base_url` as a constructor argument but never
  reads it from the environment, so there is currently no way to point the
  pipeline at a local vLLM/Ollama server without a code change.

## How to re-verify

```bash
# catalogue and pricing
curl -s https://openrouter.ai/api/v1/models | jq '.data[] | select(.id|test("deepseek")) | {id, pricing}'

# account tier and quota
curl -s https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY"

# what a specific call actually cost
curl -s "https://openrouter.ai/api/v1/generation?id=<gen_id>" -H "Authorization: Bearer $OPENROUTER_API_KEY"
```

Cost figures come from `total_cost` on the generation endpoint. Do not compute
from list prices — reasoning tokens make that unreliable.
