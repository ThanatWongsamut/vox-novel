from pathlib import Path
from typing import Any, Callable, Optional
import numpy as np
import soundfile as sf
from vox_novel.models.domain import Chapter
from vox_novel.tts.base import BaseTTS


class DummyTTS(BaseTTS):
    """Generates synthetic audio for testing the TTS pipeline without GPU."""

    @property
    def name(self) -> str:
        return "dummy"

    async def synthesize(
        self,
        text: str,
        output_file: Path,
        voice_description: Optional[str] = None,
        reference_audio: Optional[Path] = None,
        emotion: Optional[str] = None,
    ) -> Path:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        sample_rate = 24000
        duration = 1.5
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = (0.1 * np.sin(2 * np.pi * 440 * t) * np.linspace(1, 0.05, len(t))).astype(np.float32)
        sf.write(output_file, audio, sample_rate)
        return output_file

    async def synthesize_chapter(
        self,
        chapter: Chapter,
        output_dir: Path,
        use_translated: bool = True,
        voice_description: Optional[str] = None,
        reference_audio: Optional[Path] = None,
        progress_callback: Optional[Callable[[int, str], Any]] = None,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        final_file = output_dir / f"chapter_{chapter.id}.wav"
        
        valid_paras = [
            p for p in chapter.paragraphs 
            if (p.translated_text if use_translated else p.text) and (p.translated_text if use_translated else p.text).strip()
        ]
        
        total = max(len(valid_paras), 1)
        sample_rate = 24000
        audio_chunks = []
        pause = np.zeros(int(sample_rate * 0.3), dtype=np.float32)

        for idx, p in enumerate(valid_paras):
            if progress_callback:
                pct = int((idx / total) * 100)
                await progress_callback(pct, f"Synthesizing paragraph {idx + 1}/{total} (Dummy TTS)...")
            
            t = np.linspace(0, 0.8, int(sample_rate * 0.8), endpoint=False)
            tone = (0.05 * np.sin(2 * np.pi * 350 * t)).astype(np.float32)
            audio_chunks.append(tone)
            audio_chunks.append(pause)

        if not audio_chunks:
            audio_chunks.append(np.zeros(sample_rate, dtype=np.float32))

        combined = np.concatenate(audio_chunks)
        sf.write(final_file, combined, sample_rate)
        
        if progress_callback:
            await progress_callback(100, "Audio synthesis complete!")

        return final_file
