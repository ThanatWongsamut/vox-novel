"""Speaker attribution: which character speaks each paragraph.

Assigns Paragraph.speaker, which is what the TTS engine's per-character voice
path reads. Accuracy depends on the character registry more than on the model,
so a chapter is reviewed before it is synthesized.
"""

from vox_novel.speaker.models import CharacterDraft, ChunkAnnotation, Segment

__all__ = ["CharacterDraft", "ChunkAnnotation", "Segment"]
