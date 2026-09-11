"""Speaker attribution: which character speaks each paragraph.

Turns on the per-character voice path in the TTS engine, which is built and
tested but inert while nothing assigns Paragraph.speaker.
"""

from vox_novel.speaker.models import CharacterDraft, ChunkAnnotation, Segment

__all__ = ["CharacterDraft", "ChunkAnnotation", "Segment"]
