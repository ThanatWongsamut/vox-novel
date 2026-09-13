"""Splitting a chapter into annotation windows.

Paragraph indices here are the 1-based Paragraph.index used everywhere else in
the project -- audio chunks are named para_{chapter_id}_{index}.wav and the
reader resolves them by that number, so an off-by-one misvoices a whole chapter.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Chunk:
    """A window to annotate, plus preceding paragraphs shown for continuity.

    Context paragraphs let the model see who is already in a conversation when a
    chunk boundary lands mid-exchange. They are not annotated again.
    """

    context: List[Tuple[int, str]] = field(default_factory=list)
    target: List[Tuple[int, str]] = field(default_factory=list)

    @property
    def indices(self) -> List[int]:
        return [i for i, _ in self.target]


def make_chunks(
    indexed: List[Tuple[int, str]], chunk_size: int = 25, context_size: int = 4
) -> List[Chunk]:
    """Split (index, text) pairs into overlapping windows.

    Takes indices from the caller rather than enumerating, so a chapter with
    gaps -- a filtered-out empty paragraph -- keeps its real numbering.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    chunks = []
    for start in range(0, len(indexed), chunk_size):
        chunks.append(
            Chunk(
                context=indexed[max(0, start - context_size):start],
                target=indexed[start:start + chunk_size],
            )
        )
    return chunks


def format_chunk(chunk: Chunk) -> str:
    """Render a chunk for the prompt, marking which paragraphs to annotate."""
    lines = []
    if chunk.context:
        lines.append("# Context (already annotated, do not repeat)")
        lines += [f"[{i}] {text}" for i, text in chunk.context]
        lines.append("")
    lines.append("# Annotate every paragraph below")
    lines += [f"[{i}] {text}" for i, text in chunk.target]
    return "\n".join(lines)
