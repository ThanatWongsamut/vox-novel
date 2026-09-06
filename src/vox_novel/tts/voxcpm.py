import asyncio
import base64
import inspect
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
        self._ref_cache_key: Optional[tuple] = None
        self._ref_cache_value: Optional[str] = None
        # Set when output came from the placeholder tone rather than a real model.
        self.used_placeholder = False

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
            self._local_model = VoxCPM.from_pretrained(
                self.model_name,
                load_denoiser=False,
                device=device,
            )
            return self._local_model
        except ImportError as e:
            if not self._warned_local:
                logger.warning(
                    f"voxcpm package not installed ({e}). "
                    "Set VOXCPM_API_URL to use a remote GPU server, or install voxcpm."
                )
                self._warned_local = True
            return None
        except Exception as e:
            if not self._warned_local:
                logger.error(f"Failed to load VoxCPM2 model: {e}")
                self._warned_local = True
            return None

    @staticmethod
    def build_control_prompt(voice_description: Optional[str], emotion: Optional[str] = None) -> str:
        """
        Translate and normalize natural language voice descriptions (Thai/English) into
        concise, highly effective English control instructions for VoxCPM2 / MiniCPM-based voice design.
        """
        raw = f"{voice_description or ''} {emotion or ''}".strip()
        if not raw:
            return "natural articulate narrator"

        clean_text = raw

        # 1. Extract age
        age_str = None
        age_match = re.search(r"อายุ\s*(?:ย่างเข้า|ประมาณ|ราวๆ|ราว)?\s*(\d{1,2})\s*(?:ปี)?", clean_text)
        if not age_match:
            age_match = re.search(r"วัย\s*(\d{1,2})\s*(?:ปี)?", clean_text)
        if not age_match:
            age_match = re.search(r"\b(\d{1,2})\s*[- ]*(?:years?|yr)?[- ]*(?:old|yo)\b", clean_text, re.I)
        if not age_match:
            age_match = re.search(r"\b(\d{1,2})\s*ปี\b", clean_text)
        if age_match:
            age_val = int(age_match.group(1))
            if 5 <= age_val <= 100:
                age_str = f"{age_val}-year-old"

        # 2. Gender & Persona mapping
        text_no_narrate = clean_text.replace("บรรยาย", " ")
        female_keys = [
            "ผู้หญิง", "เพศหญิง", "หญิงสาว", "หญิง", "สาว", "กุลสตรี", "กลุสตรี",
            "คุณหนู", "เด็กสาว", "สาวน้อย", "สตรี", "นางสาว", "คุณยาย", "หญิงชรา",
            "คุณแม่", "มารดา", "แม่เฒ่า"
        ]
        male_keys = [
            "ผู้ชาย", "เพศชาย", "ชายหนุ่ม", "ชาย", "หนุ่ม", "สุภาพบุรุษ", "คุณชาย",
            "เด็กหนุ่ม", "หนุ่มน้อย", "บุรุษ", "คุณตา", "คุณปู่", "คุณลุง", "คุณพ่อ",
            "บิดา", "ชายชรา", "พ่อเฒ่า"
        ]

        has_female = any(k in text_no_narrate for k in female_keys) or bool(
            re.search(r"\b(?:female|woman|girl|lady)\b", clean_text, re.I)
        )
        has_male = any(k in text_no_narrate for k in male_keys) or bool(
            re.search(r"\b(?:male|man|boy|gentleman)\b", clean_text, re.I)
        )

        gender = None
        if has_female and not has_male:
            gender = "female"
        elif has_male and not has_female:
            gender = "male"

        is_narrator = any(k in clean_text for k in ["ผู้บรรยาย", "คนเล่า", "เล่าเรื่อง", "บรรยาย"]) or bool(
            re.search(r"\bnarrat(?:or|ion)\b", clean_text, re.I)
        )

        traits: List[str] = []

        # Personas / Archetypes
        if any(k in clean_text for k in ["คุณหนู"]):
            traits.append("noble lady")
        if any(k in clean_text for k in ["กุลสตรี", "กลุสตรี"]):
            traits.append("elegant gentlewoman")
        if any(k in clean_text for k in ["คุณชาย", "สุภาพบุรุษ"]):
            traits.append("noble gentleman")
        if any(k in clean_text for k in ["เด็กสาว", "สาวน้อย"]):
            traits.append("young girl")
        if any(k in clean_text for k in ["เด็กหนุ่ม", "หนุ่มน้อย"]):
            traits.append("young boy")
        if any(k in clean_text for k in ["เด็ก", "เด็กน้อย"]):
            traits.append("child")
        if any(k in clean_text for k in ["คนชรา", "คนแก่", "สูงวัย", "ชรา"]):
            traits.append("elderly")
        if any(k in clean_text for k in ["วัยรุ่น"]):
            traits.append("teenager")

        # Tone / Timbre / Emotion
        if any(k in clean_text for k in ["เย็น", "เยือกเย็น", "ยะเยือก"]):
            traits.append("cool, calm")
        if any(k in clean_text for k in ["ลึกลับ"]):
            traits.append("mysterious")
        if any(k in clean_text for k in ["น่าหลงใหล", "มีเสน่ห์", "ตรึงใจ", "เย้ายวน", "เซ็กซี่"]):
            traits.append("enchanting, charming")
        if "นุ่มลึก" in clean_text or "ทุ้มลึก" in clean_text:
            traits.append("soft and deep")
        elif re.search(r"(?<!ห)นุ่ม", clean_text) or "ละมุน" in clean_text or "นุ่มนวล" in clean_text:
            traits.append("soft, gentle")
        if any(k in clean_text for k in ["เสียงทุ้ม", "ทุ้ม"]) and "ทุ้มลึก" not in clean_text:
            traits.append("deep voice")
        if any(k in clean_text for k in ["อ่อนโยน"]):
            traits.append("gentle")
        if any(k in clean_text for k in ["ชัดถ้อยชัดคำ", "ชัดเจน", "ฉะฉาน"]):
            traits.append("articulate, clear")
        if any(k in clean_text for k in ["หวาน", "เสียงหวาน", "หวานใส"]):
            traits.append("sweet melodic")
        if any(k in clean_text for k in ["เสียงใส", "กังวาน"]):
            traits.append("crisp, clear")
        if any(k in clean_text for k in ["สดใส", "ร่าเริง"]):
            traits.append("bright, cheerful")
        if any(k in clean_text for k in ["มีชีวิตชีวา"]):
            traits.append("lively")
        if any(k in clean_text for k in ["ขี้เล่น", "ซุกซน"]):
            traits.append("playful")
        if any(k in clean_text for k in ["สง่างาม", "สุขุม"]):
            traits.append("dignified, elegant")
        if any(k in clean_text for k in ["อบอุ่น"]):
            traits.append("warm")
        if any(k in clean_text for k in ["มั่นใจ", "หนักแน่น"]):
            traits.append("confident")
        if any(k in clean_text for k in ["เข้มขรึม", "เคร่งขรึม"]):
            traits.append("serious, solemn")
        if any(k in clean_text for k in ["ห้าว", "ดุดัน", "แข็งกร้าว"]):
            traits.append("husky, fierce")
        if any(k in clean_text for k in ["ทรงพลัง", "มีอำนาจ", "น่าเกรงขาม"]):
            traits.append("authoritative, commanding")
        if any(k in clean_text for k in ["แฟนตาซี"]):
            traits.append("fantasy storytelling")
        if any(k in clean_text for k in ["เศร้า", "หม่นหมอง", "โศกเศร้า"]):
            traits.append("melancholic, sad")
        if any(k in clean_text for k in ["โกรธ", "ฉุนเฉียว", "ดุ"]):
            traits.append("angry")
        if any(k in clean_text for k in ["ตื่นเต้น", "ลุ้นระทึก"]):
            traits.append("suspenseful")
        if any(k in clean_text for k in ["ตื่นตระหนก", "กลัว"]):
            traits.append("fearful")
        if any(k in clean_text for k in ["กระซิบ"]):
            traits.append("whispering")
        if any(k in clean_text for k in ["น่ารัก"]):
            traits.append("cute")
        if any(k in clean_text for k in ["ใจดี"]):
            traits.append("kind")
        if any(k in clean_text for k in ["เป็นมิตร"]):
            traits.append("friendly")
        if any(k in clean_text for k in ["เป็นธรรมชาติ"]):
            traits.append("natural")

        # Include explicit English tokens (length >= 3)
        eng_tokens = re.findall(r"[a-zA-Z]{3,}", clean_text)
        ignored_eng = {
            "female", "male", "narrator", "voice", "years", "year",
            "old", "with", "and", "the", "for", "from", "that", "this"
        }
        for w in eng_tokens:
            lw = w.lower()
            if lw not in ignored_eng:
                traits.append(lw)

        parts = []
        if age_str:
            parts.append(age_str)
        if gender:
            parts.append(gender)
        if is_narrator:
            parts.append("narrator")
        else:
            parts.append("voice")

        core = " ".join(parts)

        unique_traits = []
        seen = set()
        for t in traits:
            for sub in t.split(","):
                sub = sub.strip()
                if sub and sub not in seen and sub not in core:
                    seen.add(sub)
                    unique_traits.append(sub)

        if unique_traits:
            return f"{core}, " + ", ".join(unique_traits)
        return core

    @staticmethod
    def format_designed_text(text: str, control: Optional[str]) -> str:
        control_clean = (control or "").strip()
        if not control_clean:
            return text
        if text.startswith("(") and ")" in text:
            return text
        return f"({control_clean}){text}"

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

        control_prompt = self.build_control_prompt(voice_description, emotion=emotion)
        designed_text = self.format_designed_text(text, control_prompt)

        payload: dict[str, Any] = {
            "model": self.model_name,
            "input": designed_text,
            "text": designed_text,
            "voice": control_prompt,
            "control": control_prompt,
            "cfg_value": 2.5 if not reference_audio else 2.0,
            "response_format": "wav",
        }

        if reference_audio and Path(reference_audio).exists():
            payload["reference_audio"] = self._encoded_reference(Path(reference_audio))

        async with httpx.AsyncClient(timeout=120.0) as client:
            endpoints = [f"{url}/v1/audio/speech", f"{url}/synthesize"]
            for ep in endpoints:
                try:
                    resp = await client.post(ep, json=payload, headers=headers)
                    if resp.status_code == 200:
                        import io
                        audio_data, sr = sf.read(io.BytesIO(resp.content))
                        # Honour the server's rate, otherwise the stitched file plays
                        # back at the wrong speed.
                        self.sample_rate = int(sr)
                        return audio_data.astype(np.float32)
                except Exception as e:
                    logger.debug(f"Endpoint {ep} failed: {e}")
                    continue

        return None

    def _encoded_reference(self, path: Path) -> str:
        """Base64-encode a reference clip once, keyed by path and mtime."""
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        if self._ref_cache_key != key:
            self._ref_cache_key = key
            self._ref_cache_value = base64.b64encode(path.read_bytes()).decode("utf-8")
        return self._ref_cache_value

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

        control_prompt = self.build_control_prompt(voice_description, emotion=emotion)

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
                    kwargs: dict[str, Any] = {}
                    if reference_audio and Path(reference_audio).exists():
                        kwargs["reference_wav_path"] = str(reference_audio)
                        if emotion:
                            emotion_ctrl = self.build_control_prompt(None, emotion=emotion)
                            kwargs["text"] = self.format_designed_text(text, emotion_ctrl)
                        else:
                            kwargs["text"] = text
                    else:
                        # Voice Design / Prompt-to-Voice mode
                        kwargs["text"] = self.format_designed_text(text, control_prompt)
                        kwargs["cfg_value"] = 2.5

                    audio_array = await asyncio.to_thread(model.generate, **kwargs)
                    if hasattr(model, "tts_model") and hasattr(model.tts_model, "sample_rate"):
                        self.sample_rate = model.tts_model.sample_rate
                except Exception as e:
                    logger.error(f"Local VoxCPM generation failed: {e}")

        # 3. Fallback to placeholder if unconfigured
        if audio_array is None:
            logger.info("Using synthetic placeholder tone (real model weights not loaded).")
            self.used_placeholder = True
            audio_array = self._generate_synthetic_placeholder(text)

        sf.write(output_file, audio_array, self.sample_rate)
        return output_file

    @staticmethod
    async def _report(progress_callback, pct: int, msg: str) -> None:
        """Invoke a progress callback that may be sync or async."""
        if not progress_callback:
            return
        result = progress_callback(pct, msg)
        if inspect.isawaitable(result):
            await result

    async def synthesize_chapter(
        self,
        chapter: Chapter,
        output_dir: Path,
        use_translated: bool = True,
        voice_description: Optional[str] = None,
        reference_audio: Optional[Path] = None,
        knowledge: Optional[Any] = None,
        progress_callback: Optional[Callable[[int, str], Any]] = None,
    ) -> Path:
        """
        Synthesize entire chapter into a single master audio file.
        Stitches paragraph audio chunks with natural pauses.
        Uses character reference voice anchors when available, falling back to narrator.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        final_file = output_dir / f"chapter_{chapter.id}.wav"

        valid_paras: List[Paragraph] = [
            p for p in chapter.paragraphs
            if (p.translated_text if use_translated else p.text) and (p.translated_text if use_translated else p.text).strip()
        ]

        total = max(len(valid_paras), 1)
        audio_segments: List[np.ndarray] = []
        # The engine may only learn its true rate after the first synthesis
        # (a remote server picks it), so the 350ms pause is sized per chunk below.
        chunk_rate: Optional[int] = None

        narrator_desc = voice_description
        if not narrator_desc and knowledge and getattr(knowledge, "narrator_voice_description", None):
            narrator_desc = knowledge.narrator_voice_description
        default_desc = narrator_desc or "เสียงบรรยายผู้ชาย นุ่มลึก ชัดถ้อยชัดคำ เหมาะกับการเล่านิยายแฟนตาซี"

        # Ensure a persistent narrator reference voice exists so all narration paragraphs sound identical
        effective_narrator_ref = reference_audio
        if effective_narrator_ref is None and knowledge and getattr(knowledge, "narrator_voice_ref_audio", None):
            k_ref = Path(knowledge.narrator_voice_ref_audio)
            if k_ref.exists() and k_ref.stat().st_size > 0:
                effective_narrator_ref = k_ref

        if effective_narrator_ref is None or not Path(effective_narrator_ref).exists():
            voices_dir = output_dir.parent / "voices"
            voices_dir.mkdir(parents=True, exist_ok=True)
            narrator_sample = voices_dir / "narrator_ref.wav"
            if not narrator_sample.exists():
                logger.info("Generating canonical narrator reference voice anchor...")
                sample_text = "นี่คือเสียงผู้บรรยายประจำนิยายเรื่องนี้ สำหรับการอ่านออกเสียงภาษาไทย"
                await self.synthesize(
                    text=sample_text,
                    output_file=narrator_sample,
                    voice_description=default_desc,
                    reference_audio=None,
                )
            effective_narrator_ref = narrator_sample

        for idx, p in enumerate(valid_paras):
            text_to_speak = (p.translated_text if use_translated else p.text).strip()
            
            # Progress reporting
            pct = int((idx / total) * 95)
            speaker_tag = f"[{p.speaker}] " if p.speaker else ""
            short_text = (text_to_speak[:25] + "...") if len(text_to_speak) > 25 else text_to_speak
            await self._report(progress_callback, pct, f"Synthesizing {idx + 1}/{total}: {speaker_tag}{short_text}")

            # Determine voice & emotion for this paragraph (character-specific voice or narrator)
            para_voice_desc = default_desc
            para_ref_audio = effective_narrator_ref
            para_emotion = p.emotion

            if knowledge and p.speaker and p.speaker.strip().lower() not in ("narrator", "ผู้บรรยาย"):
                char = knowledge.find_character(p.speaker)
                if char:
                    # Check if character has dedicated reference audio anchor
                    char_ref = None
                    if char.voice_ref_audio and Path(char.voice_ref_audio).exists():
                        char_ref = Path(char.voice_ref_audio)
                    else:
                        voices_dir = output_dir.parent / "voices"
                        safe_k = re.sub(r"[\s\-_]+", "_", char.name_en.strip().lower())
                        for ext in [".wav", ".mp3", ".m4a", ".flac"]:
                            cand = voices_dir / f"{safe_k}_ref{ext}"
                            if cand.exists() and cand.stat().st_size > 0:
                                char_ref = cand
                                break

                    if char_ref:
                        para_ref_audio = char_ref

                    if char.voice_description:
                        para_voice_desc = char.voice_description

            # Key chunks by Paragraph.index so /api/audio/.../para/{index} resolves them.
            chunk_file = output_dir / f"para_{chapter.id}_{p.index}.wav"
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
                if chunk_rate is None:
                    chunk_rate = int(sr)
                elif int(sr) != chunk_rate:
                    logger.warning(
                        f"Sample rate changed mid-chapter ({chunk_rate} -> {sr}); "
                        f"skipping {chunk_file} to avoid distorted playback."
                    )
                    continue
                audio_segments.append(data.astype(np.float32))
                audio_segments.append(np.zeros(int(chunk_rate * 0.35), dtype=np.float32))
                p.audio_path = str(chunk_file)
            except Exception as e:
                logger.warning(f"Failed to read paragraph audio chunk {chunk_file}: {e}")

        await self._report(progress_callback, 96, "Stitching chapter audio...")

        if chunk_rate is None:
            chunk_rate = self.sample_rate
        if not audio_segments:
            audio_segments.append(np.zeros(chunk_rate, dtype=np.float32))

        combined = np.concatenate(audio_segments)
        sf.write(final_file, combined, chunk_rate)
        chapter.audio_path = str(final_file)

        done_msg = (
            "Audiobook generated with PLACEHOLDER tones -- no VoxCPM weights or "
            "VOXCPM_API_URL configured."
            if self.used_placeholder
            else "Audiobook generation complete!"
        )
        await self._report(progress_callback, 100, done_msg)

        return final_file
