"""Speaker attribution over a chapter's paragraphs.

Adapted from the novel-to-script prototype, which established three failure
modes that only appear across many chapters. All three are handled here:

- The registry freezes. Models put an unfamiliar name straight into `speaker`
  and rarely declare it in `new_characters`, so the cast stops growing while
  off-registry speakers accumulate. Any speaker seen is registered.
- Name drift splits one character across several voices. Corrupted and alias
  spellings resolve through SeriesKnowledge.find_character.
- The registry inflates every prompt as the cast grows. Only a scoped view is
  shown to the model; the full registry is still used for resolution.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from vox_novel.models.domain import Chapter, Paragraph
from vox_novel.models.series_knowledge import SeriesKnowledge
from vox_novel.speaker.chunking import Chunk, format_chunk, make_chunks
from vox_novel.speaker.models import ChunkAnnotation, RegistryDraft, Segment
from vox_novel.speaker.prompts import ANNOTATE_SYSTEM, EXTRACT_SYSTEM, registry_block

logger = logging.getLogger(__name__)

# Read this far into a chapter when building the registry. Extraction only needs
# enough text to meet the cast.
EXTRACT_MAX_CHARS = 60_000

# Characters shown to the model per call. The full registry still resolves names;
# capping what the model sees keeps a large cast from crowding out the chapter.
MAX_REGISTRY_SHOWN = 40
RECENCY_WINDOW = 3

CHUNK_SIZE = 25
CONTEXT_SIZE = 4


def _paragraph_text(p: Paragraph, use_translated: bool) -> str:
    if use_translated:
        return (p.translated_text or p.text or "").strip()
    return (p.text or "").strip()


def _scoped_registry(knowledge: SeriesKnowledge, chapters_seen: int) -> List[Dict[str, Any]]:
    """The registry as the model should see it: narrator first, then recent and
    high-volume speakers."""
    characters = list(knowledge.characters.values())
    if len(characters) > MAX_REGISTRY_SHOWN:
        characters.sort(
            key=lambda c: (c.is_narrator, c.line_count),
            reverse=True,
        )
        characters = characters[:MAX_REGISTRY_SHOWN]
    return [
        {
            "name": c.name_target or c.name_en,
            "aliases": c.aliases,
            "gender": c.gender or "unknown",
            "speech_style": c.speech_style or "",
            # Who this character is. Dropping it measurably hurt attribution on
            # untagged reactive lines, where role is the only way to judge which
            # of two same-register speakers would plausibly say something.
            "description": c.notes or c.role or "",
            "is_narrator": c.is_narrator,
        }
        for c in characters
    ]


async def extract_registry(
    chapter: Chapter,
    knowledge: SeriesKnowledge,
    translator,
    use_translated: bool = True,
) -> List[str]:
    """Populate the registry from a chapter. Returns the names added.

    Registry quality is the bottleneck for attribution accuracy, not the
    attribution step itself -- correcting a single narrator entry moved the
    prototype's benchmark from 87% to 100%. This pass exists so a human has
    something to review before a long run.
    """
    text = "\n\n".join(
        _paragraph_text(p, use_translated) for p in chapter.paragraphs
    )[:EXTRACT_MAX_CHARS]

    draft: RegistryDraft = await translator.structured(EXTRACT_SYSTEM, text, RegistryDraft)

    added = []
    for c in draft.characters:
        if knowledge.find_character(c.name, allow_prefix=False) is not None:
            continue
        knowledge.add_character(
            c.name, c.name, gender=c.gender if c.gender != "unknown" else None,
            aliases=c.aliases, notes=c.description or None,
        )
        stored = knowledge.find_character(c.name)
        if stored is not None:
            stored.is_narrator = c.is_narrator
            stored.speech_style = c.speech_style or None
        added.append(c.name)

    if draft.narration_note:
        knowledge.narration_note = draft.narration_note
    return added


def _register_unseen(
    segments: List[Segment], knowledge: SeriesKnowledge, chapter_id: str
) -> List[str]:
    """Register speakers the model used without declaring.

    This is the safety net for the registry freezing: in practice models name a
    new speaker in `speaker` far more often than they fill `new_characters`.
    """
    added = []
    for seg in segments:
        name = (seg.speaker or "").strip()
        if not name or name.upper() == "UNKNOWN":
            continue
        if knowledge.find_character(name) is not None:
            continue
        knowledge.add_character(name, name, notes=f"auto-registered from chapter {chapter_id}")
        stored = knowledge.find_character(name)
        if stored is not None:
            stored.last_seen_chapter = chapter_id
        added.append(name)
    return added


def _canonicalize(segments: List[Segment], knowledge: SeriesKnowledge) -> None:
    """Fold alias and corrupted spellings onto one canonical name, so a single
    character does not end up split across several voices."""
    for seg in segments:
        if not seg.speaker or seg.speaker.upper() == "UNKNOWN":
            continue
        match = knowledge.find_character(seg.speaker)
        if match is not None:
            seg.speaker = match.name_en


def _apply(
    segments: List[Segment], by_index: Dict[int, Paragraph], knowledge: SeriesKnowledge,
    chapter_id: str,
) -> int:
    """Write attributions onto the paragraphs. Returns how many were applied."""
    applied = 0
    for seg in segments:
        paragraph = by_index.get(seg.paragraph)
        if paragraph is None:
            # A number the model invented, or one from the context block.
            continue
        paragraph.speech_type = seg.type
        paragraph.speaker = None if seg.type == "narration" else seg.speaker
        applied += 1

        if paragraph.speaker and paragraph.speaker.upper() != "UNKNOWN":
            character = knowledge.find_character(paragraph.speaker)
            if character is not None:
                character.line_count += 1
                character.last_seen_chapter = chapter_id
    return applied


def _agreement_gate(
    primary: List[Segment], secondary: List[Segment]
) -> Tuple[Dict[int, Segment], List[Dict[str, Any]]]:
    """Keep only attributions two models agree on; report the rest.

    A wrong character voice is worse for a listener than no character voice at
    all -- the narrator reading everything is at least consistent, while a voice
    that switches mid-conversation is obviously broken. So an uncertain line
    falls back to narration, which sounds exactly as it does today.

    Confidence cannot drive this: models report 1.0 while wrong. Agreement
    between two independent runs can.
    """
    other = {s.paragraph: s for s in secondary}
    kept: Dict[int, Segment] = {}
    disagreements: List[Dict[str, Any]] = []

    for seg in primary:
        rival = other.get(seg.paragraph)
        if rival is None:
            # The second model said nothing about this paragraph; treat silence
            # as disagreement rather than assent.
            disagreements.append(
                {"paragraph": seg.paragraph, "primary": seg.speaker, "secondary": None}
            )
            continue

        same_type = seg.type == rival.type
        same_speaker = (seg.speaker or "") == (rival.speaker or "")
        if same_type and same_speaker:
            kept[seg.paragraph] = seg
        else:
            disagreements.append(
                {
                    "paragraph": seg.paragraph,
                    "primary": f"{seg.type}/{seg.speaker}",
                    "secondary": f"{rival.type}/{rival.speaker}",
                }
            )
    return kept, disagreements


async def _annotate_pass(
    indexed: List[Tuple[int, str]],
    knowledge: SeriesKnowledge,
    translator,
    chapter_id: str,
    progress_callback=None,
    label: str = "",
) -> List[Segment]:
    """One full annotation pass over a chapter, including the repair pass."""
    chunks = make_chunks(indexed, chunk_size=CHUNK_SIZE, context_size=CONTEXT_SIZE)
    segments: List[Segment] = []

    for n, chunk in enumerate(chunks, start=1):
        if progress_callback:
            await _report(progress_callback, n, len(chunks), label)
        segments.extend(await _annotate_chunk(chunk, knowledge, translator, chapter_id))

    # A model that skipped paragraphs -- common near the end of a long chunk --
    # gets one more pass over just the gaps before they fall back to narration.
    seen = {s.paragraph for s in segments}
    missing = [pair for pair in indexed if pair[0] not in seen]
    if missing:
        logger.info(f"Repairing {len(missing)} unannotated paragraph(s)...")
        segments.extend(
            await _annotate_chunk(Chunk(context=[], target=missing), knowledge, translator, chapter_id)
        )
    return segments


async def annotate_chapter(
    chapter: Chapter,
    knowledge: SeriesKnowledge,
    translator,
    use_translated: bool = True,
    progress_callback=None,
    confirm_with=None,
) -> Dict[str, Any]:
    """Attribute every paragraph of a chapter in place.

    confirm_with runs a second, independent annotation pass and keeps only the
    attributions both agree on. The rest fall back to narration: at the accuracy
    these models reach, a wrong character voice is worse than none, and an
    unattributed line sounds exactly as it does today.
    """
    indexed: List[Tuple[int, str]] = [
        (p.index, _paragraph_text(p, use_translated))
        for p in chapter.paragraphs
        if _paragraph_text(p, use_translated)
    ]
    if not indexed:
        return {"annotated": 0, "missing": [], "low_confidence": [], "disagreements": []}

    by_index = {p.index: p for p in chapter.paragraphs}
    all_segments = await _annotate_pass(
        indexed, knowledge, translator, chapter.id, progress_callback,
        label="Attributing" if confirm_with else "",
    )

    disagreements: List[Dict[str, Any]] = []
    if confirm_with is not None:
        second = await _annotate_pass(
            indexed, knowledge, confirm_with, chapter.id, progress_callback,
            label="Confirming",
        )
        _canonicalize(second, knowledge)
        _canonicalize(all_segments, knowledge)
        kept, disagreements = _agreement_gate(all_segments, second)
        # A disagreement is not an error to drop silently -- the paragraph still
        # needs a type, it just does not get a character voice.
        for seg in all_segments:
            if seg.paragraph not in kept:
                seg.type = "narration"
                seg.speaker = None

    _canonicalize(all_segments, knowledge)
    applied = _apply(all_segments, by_index, knowledge, chapter.id)

    still_missing = [i for i, _ in indexed if i not in {s.paragraph for s in all_segments}]
    for index in still_missing:
        # Narration is the safe default: it uses the narrator voice, which is what
        # an unattributed paragraph would have got before this feature existed.
        paragraph = by_index.get(index)
        if paragraph is not None and not paragraph.speech_type:
            paragraph.speech_type = "narration"

    return {
        "annotated": applied,
        "missing": still_missing,
        "low_confidence": [
            {"paragraph": s.paragraph, "speaker": s.speaker, "confidence": s.confidence}
            for s in all_segments
            if s.type != "narration" and s.confidence < 0.7
        ],
        "disagreements": disagreements,
    }


async def _annotate_chunk(
    chunk: Chunk, knowledge: SeriesKnowledge, translator, chapter_id: str
) -> List[Segment]:
    block = registry_block(
        json.dumps(_scoped_registry(knowledge, 0), ensure_ascii=False, indent=1),
        knowledge.narration_note or "",
    )
    user = f"{block}\n\n{format_chunk(chunk)}"

    try:
        result: ChunkAnnotation = await translator.structured(
            ANNOTATE_SYSTEM, user, ChunkAnnotation
        )
    except Exception as e:
        logger.warning(f"Speaker annotation failed for a chunk: {e}")
        return []

    for draft in result.new_characters:
        if knowledge.find_character(draft.name, allow_prefix=False) is None:
            knowledge.add_character(
                draft.name, draft.name,
                gender=draft.gender if draft.gender != "unknown" else None,
                aliases=draft.aliases, notes=draft.description or None,
            )
            stored = knowledge.find_character(draft.name)
            if stored is not None:
                stored.speech_style = draft.speech_style or None
                stored.last_seen_chapter = chapter_id

    _register_unseen(result.segments, knowledge, chapter_id)

    # Only keep numbers this chunk actually asked about.
    wanted = set(chunk.indices)
    return [s for s in result.segments if s.paragraph in wanted]


async def _report(progress_callback, n: int, total: int, label: str = "") -> None:
    import inspect

    what = label or "Attributing speakers"
    result = progress_callback(int(n / max(total, 1) * 100), f"{what} ({n}/{total})...")
    if inspect.isawaitable(result):
        await result
