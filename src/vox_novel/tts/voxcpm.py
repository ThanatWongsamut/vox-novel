import base64
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, List, Optional
import httpx
import numpy as np
import soundfile as sf
from vox_novel.models.domain import Chapter, Paragraph
from vox_novel.tts.base import BaseTTS

logger = logging.getLogger(__name__)


class VoxCPM2TTS(BaseTTS):
    """
    VoxCPM2 Text-to-Speech adapter supporting:
    - Thai language synthesis (Tokenizer-Free 48kHz).
    - Prompt-to-Voice / Voice Design (natural language voice descriptions).
    - Voice Cloning & Controllable Cloning (character voice consistency via reference audio).
    - Dual runtime: Local inference (MPS/CUDA/CPU) or Remote API (vLLM-Omni / self-hosted VoxCPM).
    """

    def __init__(
        self,
        model_name: str = "openbmb/VoxCPM2",
        device: Optional[str] = None,
        api_url: Optional[str] = None,
        sample_rate: int = 48000,
    ):
        self.model_name = model_name
        self.device_name = device or os.getenv("VOXCPM_DEVICE", "auto")
        self.api_url = api_url or os.getenv("VOXCPM_API_URL")
        self.sample_rate = sample_rate
        self._local_model = None
        self._warned_local = False

    @property
    def name(self) -> str:
        return "voxcpm2"

    def _resolve_device(self) -> str:
        if self.device_name != "auto":
            return self.device_name
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _get_local_model(self):
        """Lazy load VoxCPM model locally."""
        if self._local_model is not None:
            return self._local_model

        try:
            from voxcpm import VoxCPM
            device = self._resolve_device()
            logger.info(f"Loading VoxCPM2 model ({self.model_name}) on device: {device}...")
            self._local_model = VoxCPM.from_pretrained(self.model_name)
            return self._local_model
        except ImportError as e:
            if not self._warned_local:
                logger.warning(
                    f"voxcpm package not installed locally ({e}). "
                    "Set VOXCPM_API_URL to use a remote GPU server, or install voxcpm."
                )
                self._warned_local = True
            return None
        except Exception as e:
            if not self._warned_local:
                logger.error(f"Failed to load VoxCPM2 model: {e}")
                self._warned_local = True
            return None

    async def _call_remote_api(
        self,
        text: str,
        voice_description: Optional[str] = None,
        reference_audio: Optional[Path] = None,
        emotion: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Call remote VoxCPM / vLLM-Omni HTTP API."""
        if not self.api_url:
            return None

        url = self.api_url.rstrip("/")
        headers = {}
        api_key = os.getenv("VOXCPM_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Combine voice description and emotion into control prompt
        control_prompt = voice_description or "เสียงบรรยายภาษาไทย นุ่มนวล ชัดเจน มีชีวิตชีวา"
        if emotion:
            control_prompt += f", อารมณ์: {emotion}"

        payload: dict[str, Any] = {
            "model": self.model_name,
            "input": text,
            "text": text,
            "voice": control_prompt,
            "control": control_prompt,
            "response_format": "wav",
        }

        if reference_audio and Path(reference_audio).exists():
            with open(reference_audio, "rb") as f:
                payload["reference_audio"] = base64.b64encode(f.read()).decode("utf-8")

        async with httpx.AsyncClient(timeout=120.0) as client:
            endpoints = [f"{url}/v1/audio/speech", f"{url}/synthesize"]
            for ep in endpoints:
                try:
                    resp = await client.post(ep, json=payload, headers=headers)
                    if resp.status_code == 200:
                        import io
                        audio_data, sr = sf.read(io.BytesIO(resp.content))
                        return audio_data.astype(np.float32)
                except Exception as e:
                    logger.debug(f"Endpoint {ep} failed: {e}")
                    continue

        return None

    def _generate_synthetic_placeholder(self, text: str) -> np.ndarray:
        """
        Generate a placeholder audio when neither local weights nor remote API are configured.
        Allows testing end-to-end audiobook reader and UI without stalling.
        """
        duration = min(max(len(text) * 0.05, 0.8), 3.0)
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        # Gentle multi-harmonic tone (warm chime)
        tone = (
            0.05 * np.sin(2 * np.pi * 320 * t)
            + 0.02 * np.sin(2 * np.pi * 480 * t)
        ) * np.linspace(0.8, 0.1, len(t))
        return tone.astype(np.float32)

    async def synthesize(
        self,
        text: str,
        output_file: Path,
        voice_description: Optional[str] = None,
        reference_audio: Optional[Path] = None,
        emotion: Optional[str] = None,
    ) -> Path:
        """Synthesize a single text into audio."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        audio_array = None

        # 1. Try remote API first if configured
        if self.api_url:
            try:
                audio_array = await self._call_remote_api(
                    text=text,
                    voice_description=voice_description,
                    reference_audio=reference_audio,
                    emotion=emotion,
                )
            except Exception as e:
                logger.warning(f"Remote VoxCPM API error: {e}")

        # 2. Try local model
        if audio_array is None:
            model = self._get_local_model()
            if model is not None:
                try:
                    control_prompt = voice_description or "เสียงผู้บรรยายภาษาไทย ชัดเจน เป็นธรรมชาติ"
                    if emotion:
                        control_prompt += f", อารมณ์ {emotion}"

                    kwargs: dict[str, Any] = {"text": text, "control": control_prompt}
                    if reference_audio and Path(reference_audio).exists():
                        kwargs["reference_audio"] = str(reference_audio)

                    audio_array = model.generate(**kwargs)
                except Exception as e:
                    logger.error(f"Local VoxCPM generation failed: {e}")

        # 3. Fallback to placeholder if unconfigured
        if audio_array is None:
            logger.info("Using synthetic placeholder for PoC demonstration.")
            audio_array = self._generate_synthetic_placeholder(text)

        sf.write(output_file, audio_array, self.sample_rate)
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
        """
        Synthesize entire chapter into a single master audio file.
        Stitches paragraph audio chunks with natural pauses.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        final_file = output_dir / f"chapter_{chapter.id}.wav"

        valid_paras: List[Paragraph] = [
            p for p in chapter.paragraphs
            if (p.translated_text if use_translated else p.text) and (p.translated_text if use_translated else p.text).strip()
        ]

        total = max(len(valid_paras), 1)
        audio_segments: List[np.ndarray] = []
        pause_samples = int(self.sample_rate * 0.35)  # 350ms pause between paragraphs
        pause = np.zeros(pause_samples, dtype=np.float32)

        default_desc = voice_description or "เสียงบรรยายผู้ชาย นุ่มลึก ชัดถ้อยชัดคำ เหมาะกับการเล่านิยายแฟนตาซี"

        for idx, p in enumerate(valid_paras):
            text_to_speak = (p.translated_text if use_translated else p.text).strip()
            
            # Progress reporting
            if progress_callback:
                pct = int((idx / total) * 95)
                speaker_tag = f"[{p.speaker}] " if p.speaker else ""
                short_text = (text_to_speak[:25] + "...") if len(text_to_speak) > 25 else text_to_speak
                await progress_callback(pct, f"Synthesizing {idx + 1}/{total}: {speaker_tag}{short_text}")

            # Determine voice & emotion for this paragraph (consistent character reference or narrator)
            para_voice_desc = default_desc
            para_ref_audio = reference_audio
            para_emotion = p.emotion

            # Clean and synthesize chunk
            chunk_file = output_dir / f"para_{chapter.id}_{idx}.wav"
            await self.synthesize(
                text=text_to_speak,
                output_file=chunk_file,
                voice_description=para_voice_desc,
                reference_audio=para_ref_audio,
                emotion=para_emotion,
            )

            # Read back array to concatenate
            try:
                data, sr = sf.read(chunk_file)
                if data.ndim > 1:
                    data = data.mean(axis=1)  # Convert stereo to mono
                audio_segments.append(data.astype(np.float32))
                audio_segments.append(pause)
                p.audio_path = str(chunk_file)
            except Exception as e:
                logger.warning(f"Failed to read paragraph audio chunk {chunk_file}: {e}")

        if progress_callback:
            await progress_callback(96, "Stitching chapter audio...")

        if not audio_segments:
            audio_segments.append(np.zeros(self.sample_rate, dtype=np.float32))

        combined = np.concatenate(audio_segments)
        sf.write(final_file, combined, self.sample_rate)
        chapter.audio_path = str(final_file)

        if progress_callback:
            await progress_callback(100, "Audiobook generation complete!")

        return final_file
