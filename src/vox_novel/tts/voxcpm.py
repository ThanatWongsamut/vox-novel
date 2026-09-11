import asyncio
import base64
import hashlib
import inspect
import logging
import os
import re
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, List, Optional
import httpx
import numpy as np
import soundfile as sf
from vox_novel.models.domain import Chapter, Paragraph
from vox_novel.storage.file import character_voice_key
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
        # Keyed by (path, mtime, size). A dialogue chapter alternates between the
        # narrator and several characters, so a single slot would thrash.
        self._ref_cache: "OrderedDict[tuple, str]" = OrderedDict()
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

    # The patch below is a copy of VoxCPM2Model._inference from this exact version,
    # with a one-step delay added after the stop flag. Bump only after diffing the
    # new upstream implementation and re-applying the change by hand.
    PATCHED_VOXCPM_VERSION = "2.0.3"

    # Upstream's parameter list at PATCHED_VOXCPM_VERSION. Checked before patching so
    # a signature change fails loudly instead of silently reverting upstream to stale
    # inference logic.
    _EXPECTED_INFERENCE_PARAMS = (
        "self", "text", "text_mask", "feat", "feat_mask", "min_len", "max_len",
        "inference_timesteps", "cfg_value", "streaming", "streaming_prefix_len",
    )

    # Classifier-free guidance strength. Voice Design has only the text prompt to
    # steer it, so it needs a stronger pull than cloning, which already has a
    # reference clip anchoring the timbre.
    CFG_VOICE_DESIGN = 2.5
    CFG_CLONE = 2.0

    # Floor on generated acoustic patches, as a fraction of input length. Without it
    # the stop head can fire on an early pause and truncate the line; ~0.08 patches
    # per character keeps short utterances from ending before the text is spoken.
    MIN_LEN_PER_CHAR = 0.08
    MIN_LEN_FLOOR = 2

    @classmethod
    def _min_len_for(cls, text: str) -> int:
        return max(cls.MIN_LEN_FLOOR, int(len(text) * cls.MIN_LEN_PER_CHAR))

    @classmethod
    def _patch_voxcpm_inference(cls):
        """
        Patch VoxCPM2Model._inference to avoid premature stop cutoffs:
        1. Once stop_flag == 1 is first detected, generate 1 extra acoustic patch
           so the vocal tract / diffusion model smoothly completes the final phoneme
           release and natural decay into silence.
        2. Supports both non-streaming and streaming generation.

        Raises RuntimeError if the installed voxcpm no longer matches the version
        this patch was written against -- applying it blindly would replace a newer
        upstream implementation with this stale copy.
        """
        import inspect

        import torch

        from voxcpm.model.voxcpm2 import VoxCPM2Model

        if getattr(VoxCPM2Model, "_voxnovel_patched", False):
            return

        installed = cls._installed_voxcpm_version()
        params = tuple(inspect.signature(VoxCPM2Model._inference).parameters)
        if params != cls._EXPECTED_INFERENCE_PARAMS or installed != cls.PATCHED_VOXCPM_VERSION:
            raise RuntimeError(
                f"Cannot apply the VoxCPM cutoff patch: it targets voxcpm "
                f"{cls.PATCHED_VOXCPM_VERSION} but found {installed or 'unknown'} "
                f"with signature {params}. Re-derive the patch from the installed "
                f"version, then update PATCHED_VOXCPM_VERSION."
            )

        # Upstream runs inference under torch.inference_mode(); without it every
        # synthesis builds autograd graphs across up to max_len diffusion steps.
        @torch.inference_mode()
        def patched_inference(
            self,
            text,
            text_mask,
            feat,
            feat_mask,
            min_len=2,
            max_len=2000,
            inference_timesteps=10,
            cfg_value=2.0,
            streaming=False,
            streaming_prefix_len=4,
        ):
            import torch
            from einops import rearrange

            B, T, P, D = feat.shape
            prefill_encoder = getattr(self, "_feat_encoder_raw", self.feat_encoder)
            feat_embed = prefill_encoder(feat)
            feat_embed = self.enc_to_lm_proj(feat_embed)
            scale_emb = self.config.lm_config.scale_emb if self.config.lm_config.use_mup else 1.0
            text_embed = self.base_lm.embed_tokens(text) * scale_emb
            combined_embed = text_mask.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed
            prefix_feat_cond = feat[:, -1, ...]

            has_continuation_audio = feat_mask[0, -1].item() == 1
            context_len = 0
            if has_continuation_audio:
                audio_indices = feat_mask.squeeze(0).nonzero(as_tuple=True)[0]
                context_len = min(streaming_prefix_len - 1, len(audio_indices))
                last_audio_indices = audio_indices[-context_len:]
                pred_feat_seq = list(feat[:, last_audio_indices, :, :].split(1, dim=1))
            else:
                pred_feat_seq = []

            enc_outputs, kv_cache_tuple = self.base_lm(inputs_embeds=combined_embed, is_causal=True)
            self.base_lm.kv_cache.fill_caches(kv_cache_tuple)
            enc_outputs = (
                self.fsq_layer(enc_outputs) * feat_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
            )
            lm_hidden = enc_outputs[:, -1, :]

            residual_enc_inputs = self.fusion_concat_proj(
                torch.cat((enc_outputs, feat_mask.unsqueeze(-1) * feat_embed), dim=-1)
            )
            residual_enc_outputs, residual_kv_cache_tuple = self.residual_lm(
                inputs_embeds=residual_enc_inputs, is_causal=True
            )
            self.residual_lm.kv_cache.fill_caches(residual_kv_cache_tuple)
            residual_hidden = residual_enc_outputs[:, -1, :]

            stop_detected = False
            extra_steps_remaining = 1

            for i in range(max_len):
                dit_hidden_1 = self.lm_to_dit_proj(lm_hidden)
                dit_hidden_2 = self.res_to_dit_proj(residual_hidden)
                dit_hidden = torch.cat((dit_hidden_1, dit_hidden_2), dim=-1)

                pred_feat = self.feat_decoder(
                    mu=dit_hidden,
                    patch_size=self.patch_size,
                    cond=prefix_feat_cond.transpose(1, 2).contiguous(),
                    n_timesteps=inference_timesteps,
                    cfg_value=cfg_value,
                ).transpose(1, 2)

                curr_embed = self.feat_encoder(pred_feat.unsqueeze(1))
                curr_embed = self.enc_to_lm_proj(curr_embed)
                pred_feat_seq.append(pred_feat.unsqueeze(1))
                prefix_feat_cond = pred_feat

                if streaming:
                    feat_pred = rearrange(
                        pred_feat.unsqueeze(1), "b t p d -> b d (t p)", b=B, p=self.patch_size
                    )
                    yield feat_pred, pred_feat_seq, context_len
                    if len(pred_feat_seq) > streaming_prefix_len:
                        pred_feat_seq = pred_feat_seq[-streaming_prefix_len:]

                stop_flag = (
                    self.stop_head(self.stop_actn(self.stop_proj(lm_hidden))).argmax(dim=-1)[0].cpu().item()
                )

                # Once the stop head fires, commit to finishing: count the extra
                # release steps down regardless of whether stop_flag jitters back to
                # 0 on a later step, which would otherwise let generation run on.
                if stop_detected:
                    extra_steps_remaining -= 1
                    if extra_steps_remaining <= 0:
                        break
                elif i > min_len and stop_flag == 1:
                    stop_detected = True

                lm_hidden = self.base_lm.forward_step(
                    curr_embed[:, 0, :], torch.tensor([self.base_lm.kv_cache.step()], device=curr_embed.device)
                ).clone()
                lm_hidden = self.fsq_layer(lm_hidden)
                curr_residual_input = self.fusion_concat_proj(torch.cat((lm_hidden, curr_embed[:, 0, :]), dim=-1))
                residual_hidden = self.residual_lm.forward_step(
                    curr_residual_input,
                    torch.tensor([self.residual_lm.kv_cache.step()], device=curr_embed.device),
                ).clone()

            if not streaming:
                pred_feat_seq = torch.cat(pred_feat_seq, dim=1)
                feat_pred = rearrange(pred_feat_seq, "b t p d -> b d (t p)", b=B, p=self.patch_size)
                generated_feat = pred_feat_seq[:, context_len:, :, :].squeeze(0).cpu()
                yield feat_pred, generated_feat, context_len

        VoxCPM2Model._inference = patched_inference
        VoxCPM2Model._voxnovel_patched = True
        logger.info("VoxCPM2Model runtime patched with smooth release and anti-cutoff protection.")

    @staticmethod
    def _installed_voxcpm_version() -> Optional[str]:
        from importlib import metadata

        try:
            return metadata.version("voxcpm")
        except metadata.PackageNotFoundError:
            return None

    def _get_local_model(self):
        """Lazy load VoxCPM model locally."""
        if self._local_model is not None:
            return self._local_model

        try:
            from voxcpm import VoxCPM

            self._patch_voxcpm_inference()
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

    # English voice descriptors accepted verbatim in a control prompt.
    _ENGLISH_VOICE_TERMS: frozenset = frozenset({
        # timbre
        "deep", "soft", "gentle", "husky", "raspy", "breathy", "smooth", "rich",
        "warm", "bright", "crisp", "clear", "mellow", "nasal", "thin", "full",
        "sweet", "melodic", "resonant", "velvety", "gravelly", "airy",
        # pace and delivery
        "slow", "fast", "measured", "steady", "brisk", "deliberate", "articulate",
        "whispering", "hushed", "drawling", "clipped",
        # affect
        "calm", "cool", "confident", "authoritative", "commanding", "serious",
        "solemn", "cheerful", "lively", "playful", "friendly", "kind", "gentle",
        "mysterious", "enchanting", "charming", "seductive", "melancholic", "sad",
        "angry", "fierce", "anxious", "fearful", "excited", "suspenseful",
        "dignified", "elegant", "noble", "humble", "weary", "tired", "energetic",
        "natural", "expressive", "monotone", "dramatic", "theatrical",
        # persona and age
        "young", "elderly", "aged", "teenage", "teenager", "child", "childlike",
        "adult", "mature", "girl", "boy", "lady", "gentleman", "woman", "man",
    })

    # Thai keyword -> English voice-design trait, applied in order. A trailing
    # third element, when present, suppresses the match (more specific term wins).
    _TRAIT_KEYWORDS: tuple = (
        (("คุณหนู",), "noble lady"),
        (("กุลสตรี", "กลุสตรี"), "elegant gentlewoman"),
        (("คุณชาย", "สุภาพบุรุษ"), "noble gentleman"),
        (("เด็กสาว", "สาวน้อย"), "young girl"),
        (("เด็กหนุ่ม", "หนุ่มน้อย"), "young boy"),
        (("เด็ก", "เด็กน้อย"), "child"),
        (("คนชรา", "คนแก่", "สูงวัย", "ชรา"), "elderly"),
        (("วัยรุ่น",), "teenager"),
        (("เย็น", "เยือกเย็น", "ยะเยือก"), "cool, calm"),
        (("ลึกลับ",), "mysterious"),
        (("น่าหลงใหล", "มีเสน่ห์", "ตรึงใจ", "เย้ายวน", "เซ็กซี่"), "enchanting, charming"),
        (("เสียงทุ้ม", "ทุ้ม"), "deep voice", "ทุ้มลึก"),
        (("อ่อนโยน",), "gentle"),
        (("ชัดถ้อยชัดคำ", "ชัดเจน", "ฉะฉาน"), "articulate, clear"),
        (("หวาน", "เสียงหวาน", "หวานใส"), "sweet melodic"),
        (("เสียงใส", "กังวาน"), "crisp, clear"),
        (("สดใส", "ร่าเริง"), "bright, cheerful"),
        (("มีชีวิตชีวา",), "lively"),
        (("ขี้เล่น", "ซุกซน"), "playful"),
        (("สง่างาม", "สุขุม"), "dignified, elegant"),
        (("อบอุ่น",), "warm"),
        (("มั่นใจ", "หนักแน่น"), "confident"),
        (("เข้มขรึม", "เคร่งขรึม"), "serious, solemn"),
        (("ห้าว", "ดุดัน", "แข็งกร้าว"), "husky, fierce"),
        (("ทรงพลัง", "มีอำนาจ", "น่าเกรงขาม"), "authoritative, commanding"),
        (("แฟนตาซี",), "fantasy storytelling"),
        (("เศร้า", "หม่นหมอง", "โศกเศร้า"), "melancholic, sad"),
        (("โกรธ", "ฉุนเฉียว", "ดุ"), "angry"),
        (("ตื่นเต้น", "ลุ้นระทึก"), "suspenseful"),
        (("ตื่นตระหนก", "กลัว"), "fearful"),
        (("กระซิบ",), "whispering"),
        (("น่ารัก",), "cute"),
        (("ใจดี",), "kind"),
        (("เป็นมิตร",), "friendly"),
        (("เป็นธรรมชาติ",), "natural"),
    )

    @classmethod
    def build_control_prompt(cls, voice_description: Optional[str], emotion: Optional[str] = None) -> str:
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
        elif has_female and has_male:
            # A description mentioning both ("a young man talking to a girl") is about
            # one speaker; take whichever term appears first rather than dropping both.
            first_female = min(
                (text_no_narrate.find(k) for k in female_keys if k in text_no_narrate),
                default=len(text_no_narrate),
            )
            first_male = min(
                (text_no_narrate.find(k) for k in male_keys if k in text_no_narrate),
                default=len(text_no_narrate),
            )
            gender = "female" if first_female <= first_male else "male"

        is_narrator = any(k in clean_text for k in ["ผู้บรรยาย", "คนเล่า", "เล่าเรื่อง", "บรรยาย"]) or bool(
            re.search(r"\bnarrat(?:or|ion)\b", clean_text, re.I)
        )

        traits: List[str] = []

        # Timbre terms that need more than a substring test: "หนุ่ม" (young man)
        # contains "นุ่ม" (soft), and "ทุ้มลึก" must not also match plain "ทุ้ม".
        if "นุ่มลึก" in clean_text or "ทุ้มลึก" in clean_text:
            traits.append("soft and deep")
        elif re.search(r"(?<!ห)นุ่ม", clean_text) or "ละมุน" in clean_text or "นุ่มนวล" in clean_text:
            traits.append("soft, gentle")

        for keywords, trait, *exclude in cls._TRAIT_KEYWORDS:
            if exclude and exclude[0] in clean_text:
                continue
            if any(k in clean_text for k in keywords):
                traits.append(trait)

        # Pass through English descriptors, but only ones we recognise as voice
        # qualities. An open pass-through turns any proper noun in the description
        # ("...from the ReadToon novel about Bangkok") into a synthesis instruction.
        for word in re.findall(r"[a-zA-Z]{3,}", clean_text):
            lw = word.lower()
            if lw in cls._ENGLISH_VOICE_TERMS:
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

    # VoxCPM2 only acts on English control prompts -- see the A/B in the test suite.
    # Anything else is read aloud, so an LLM result is only usable if it is ASCII.
    MAX_CONTROL_PROMPT_CHARS = 200
    # A chapter alternates between the narrator and its speaking characters; this
    # comfortably covers a cast while bounding memory in a long-lived process.
    MAX_REF_CACHE_ENTRIES = 32

    # Traits that describe *who* is speaking rather than how one line is delivered.
    # These are excluded when an emotion is folded into a speaker's control prompt,
    # since the speaker's own identity is already established.
    _IDENTITY_TRAITS: frozenset = frozenset({
        # from the Thai trait table
        "noble lady", "elegant gentlewoman", "noble gentleman", "young girl",
        "young boy", "child", "elderly", "teenager", "soft and deep", "deep voice",
        "fantasy storytelling",
        # from the English pass-through vocabulary, which emits these as bare words
        "young", "aged", "teenage", "childlike", "adult", "mature",
        "girl", "boy", "lady", "gentleman", "woman", "man", "noble",
    })
    # Voice design must not stall a chapter because the LLM is slow or down;
    # the keyword table is always available as a fallback.
    VOICE_PROMPT_TIMEOUT_SECONDS = 20

    VOICE_PROMPT_SYSTEM = (
        "You turn a character voice description into a control prompt for a "
        "text-to-speech voice designer.\n"
        "Reply with ONE short English phrase, comma-separated attributes only.\n"
        "Cover whichever of these the description mentions: age, gender, timbre, "
        "pace, accent, emotion.\n"
        "Use plain ASCII English. Do not translate or repeat the sample text, do not "
        "add commentary, quotes, or parentheses, and never exceed 20 words.\n"
        "Example input: เสียงบรรยายผู้หญิง อายุ 30 ปี เย็น ลึกลับ นุ่มลึก\n"
        "Example output: 30-year-old female narrator, cool, mysterious, soft and deep"
    )

    @classmethod
    def _sanitize_control_prompt(cls, candidate: str) -> Optional[str]:
        """Return a usable control prompt, or None if the model gave us something unsafe.

        A control prompt that is not plain ASCII gets spoken aloud instead of acted
        on, so anything questionable is rejected in favour of the keyword table.
        """
        # Models wrap the answer in quotes, backticks and parentheses in any
        # combination, so peel repeatedly rather than assuming one order.
        WRAPPERS = "`\"' \t\r\n"
        text = (candidate or "").strip(WRAPPERS)
        for _ in range(4):
            before = text
            if text.startswith("(") and text.endswith(")"):
                text = text[1:-1]
            text = text.strip(WRAPPERS)
            if text == before:
                break
        text = " ".join(text.split())

        if not text or not text.isascii():
            return None
        # isascii() admits control characters, and str.split() only strips the
        # whitespace ones.
        if any(ch < " " or ch == "\x7f" for ch in text):
            return None
        if len(text) > cls.MAX_CONTROL_PROMPT_CHARS:
            return None
        # Newlines are already collapsed by the split/join above; parentheses would
        # nest inside the control wrapper format_designed_text applies.
        if ")" in text or "(" in text:
            return None
        # A refusal or an explanation rather than a descriptor list.
        if re.search(r"\b(sorry|cannot|as an ai|i can't|unable)\b", text, re.I):
            return None
        # A control prompt is comma-separated descriptors. A colon means we got a
        # label instead -- a moderation router answering "User Safety: safe", or a
        # model prefixing its answer -- which would be synthesized as voice direction.
        if ":" in text:
            return None
        return text

    @classmethod
    async def _resolve_cached_control(
        cls,
        description: Optional[str],
        cached: Optional[str],
        translator: Optional[Any] = None,
    ) -> str:
        """Reuse a stored control prompt, deriving one only on a cache miss."""
        if cached:
            return cached
        return await cls.derive_control_prompt(description, translator=translator)

    @classmethod
    async def derive_control_prompt(
        cls,
        voice_description: Optional[str],
        emotion: Optional[str] = None,
        translator: Optional[Any] = None,
    ) -> str:
        """Build an English control prompt, preferring the LLM over the keyword table.

        The table can only express traits someone hand-coded, so a description like
        "เสียงแหบเล็กน้อยแบบคนเพิ่งตื่นนอน" contributes nothing to it. An LLM handles
        arbitrary wording. The table remains the fallback so voice design keeps
        working with no API key, and so a bad completion cannot make things worse.
        """
        fallback = cls.build_control_prompt(voice_description, emotion=emotion)

        raw = f"{voice_description or ''} {emotion or ''}".strip()
        if not raw or translator is None:
            return fallback

        try:
            answer = await asyncio.wait_for(
                translator.complete(cls.VOICE_PROMPT_SYSTEM, raw),
                timeout=cls.VOICE_PROMPT_TIMEOUT_SECONDS,
            )
        except NotImplementedError:
            return fallback
        except asyncio.TimeoutError:
            logger.warning(
                "Voice description translation timed out after "
                f"{cls.VOICE_PROMPT_TIMEOUT_SECONDS}s; using keyword table."
            )
            return fallback
        except Exception as e:
            logger.warning(f"Voice description translation failed, using keyword table: {e}")
            return fallback

        cleaned = cls._sanitize_control_prompt(answer)
        if not cleaned:
            logger.warning(
                "Voice description translation returned an unusable control prompt "
                f"({answer!r}); using keyword table."
            )
            return fallback
        return cleaned

    @classmethod
    def _emotion_traits(cls, emotion: Optional[str]) -> str:
        """Return only the trait list for an emotion, without the core descriptor.

        build_control_prompt always returns "<core>, <traits>", where the core is
        the age/gender/role phrase. Splicing a whole prompt into another one
        corrupts it -- "female voice, angry" naively stripped of "voice, " becomes
        "female angry" -- so take everything after the first separator instead.
        """
        if not emotion:
            return ""
        built = cls.build_control_prompt(None, emotion=emotion)
        traits = built.split(", ", 1)[1] if ", " in built else ""
        # Drop terms that describe *who* is speaking. A description like
        # "เสียงหญิงชราโกรธ" also yields "elderly", which would append an age to an
        # unrelated speaker and fight the timbre the cloned reference establishes.
        # Subtracting identity is safer than whitelisting affect: a hand-written
        # affect list silently drops legitimate terms as the trait table grows.
        return ", ".join(
            t for t in (x.strip() for x in traits.split(",")) if t and t not in cls._IDENTITY_TRAITS
        )

    @staticmethod
    def prepare_text_for_tts(text: str) -> str:
        """
        Normalize text for TTS synthesis to prevent abrupt token cutoff:
        - Normalizes trailing quotes, ellipses, and unpunctuated Thai/English sentences.
        - Ensures text ends with a natural punctuation mark or trailing space so the
          language model and acoustic vocoder complete the final word cadence.
        """
        t = (text or "").strip()
        if not t:
            return t

        terminal = (".", "!", "?", "…", "...", "—", ":", ";")

        quote_match = re.search(r'([\"\'”’])$', t)
        if quote_match:
            quote_char = quote_match.group(1)
            inner = t[:-1].rstrip()
            if inner and not inner.endswith(terminal):
                return f"{inner}.{quote_char} "
            return f"{inner}{quote_char} "

        # The trailing space is part of the cutoff fix, so it must be applied on
        # every branch -- unpunctuated Thai prose is the common case, not the rare one.
        if not t.endswith(terminal):
            return f"{t}. "
        return f"{t} "

    @staticmethod
    def _apply_tail_fadeout(audio: np.ndarray, sample_rate: int, fade_ms: float = 20.0) -> np.ndarray:
        """
        Apply a smooth cosine fade-out on the last fade_ms milliseconds of audio
        to prevent any abrupt cutoff pops or clicks at chunk boundaries.
        """
        fade_len = int(sample_rate * (fade_ms / 1000.0))
        if len(audio) <= fade_len:
            return audio
        fade_curve = (np.cos(np.linspace(0, np.pi / 2, fade_len)) ** 2).astype(audio.dtype)
        out = audio.copy()
        if out.ndim > 1:
            # A remote server may return multi-channel audio; broadcast the curve
            # down the channel axis rather than failing to align shapes.
            fade_curve = fade_curve[:, np.newaxis]
        out[-fade_len:] *= fade_curve
        return out

    @staticmethod
    def format_designed_text(text: str, control: Optional[str]) -> str:
        """Prefix text with its voice-design control prompt.

        Only an exact repeat of this control counts as already-wrapped: prose can
        legitimately open with a parenthetical ("(เสียงกระซิบ) เขาพูด"), and treating
        that as a control prompt would drop the real one.
        """
        control_clean = (control or "").strip()
        if not control_clean:
            return text
        if text.startswith(f"({control_clean})"):
            return text
        return f"({control_clean}){text}"

    async def _call_remote_api(
        self,
        text: str,
        voice_description: Optional[str] = None,
        reference_audio: Optional[Path] = None,
        emotion: Optional[str] = None,
        control_prompt: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Call remote VoxCPM / vLLM-Omni HTTP API."""
        if not self.api_url:
            return None

        url = self.api_url.rstrip("/")
        headers = {}
        api_key = os.getenv("VOXCPM_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        if control_prompt is None:
            control_prompt = self.build_control_prompt(voice_description, emotion=emotion)

        # Match the local path: a reference clip already fixes the voice identity, so
        # only an emotion is worth prefixing. Sending a full voice design alongside a
        # reference gives the model two conflicting instructions.
        has_reference = bool(reference_audio) and Path(reference_audio).exists()
        if has_reference:
            if emotion:
                emotion_ctrl = self.build_control_prompt(None, emotion=emotion)
                designed_text = self.format_designed_text(text, emotion_ctrl)
            else:
                designed_text = text
        else:
            designed_text = self.format_designed_text(text, control_prompt)

        payload: dict[str, Any] = {
            "model": self.model_name,
            "input": designed_text,
            "text": designed_text,
            "voice": control_prompt,
            "control": control_prompt,
            "cfg_value": self.CFG_CLONE if has_reference else self.CFG_VOICE_DESIGN,
            "response_format": "wav",
        }

        if reference_audio and Path(reference_audio).exists():
            payload["reference_audio"] = self._encoded_reference(Path(reference_audio))

        async with httpx.AsyncClient(timeout=120.0) as client:
            endpoints = [f"{url}/v1/audio/speech", f"{url}/synthesize"]
            failures: List[str] = []
            for ep in endpoints:
                try:
                    resp = await client.post(ep, json=payload, headers=headers)
                    if resp.status_code != 200:
                        # Only the final all-endpoints-failed case is worth a warning;
                        # a server implementing just one endpoint would otherwise log
                        # once per paragraph while working fine.
                        failures.append(f"{ep} -> HTTP {resp.status_code}: {resp.text[:120]}")
                    if resp.status_code == 200:
                        import io
                        audio_data, sr = sf.read(io.BytesIO(resp.content))
                        # Honour the server's rate, otherwise the stitched file plays
                        # back at the wrong speed.
                        self.sample_rate = int(sr)
                        if audio_data.ndim > 1:
                            # The chapter is stitched as mono; downmix here so every
                            # stage downstream sees a consistent shape.
                            audio_data = audio_data.mean(axis=1)
                        return audio_data.astype(np.float32)
                except Exception as e:
                    failures.append(f"{ep} -> {e}")
                    logger.debug(f"Endpoint {ep} failed: {e}")
                    continue

            if failures:
                logger.warning(
                    "Remote VoxCPM synthesis failed on every endpoint: "
                    + "; ".join(failures)
                )
        return None

    def _encoded_reference(self, path: Path) -> str:
        """Base64-encode a reference clip once, keyed by path and mtime."""
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        if key in self._ref_cache:
            # Least-recently-used: FIFO would evict the narrator anchor -- the most
            # reused clip in a chapter -- on every cycle through a large cast.
            self._ref_cache.move_to_end(key)
            return self._ref_cache[key]

        if len(self._ref_cache) >= self.MAX_REF_CACHE_ENTRIES:
            self._ref_cache.popitem(last=False)
        self._ref_cache[key] = base64.b64encode(path.read_bytes()).decode("utf-8")
        return self._ref_cache[key]

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
        control_prompt: Optional[str] = None,
    ) -> Path:
        """Synthesize a single text into audio.

        control_prompt, when given, is used as-is; callers synthesizing many
        paragraphs derive it once rather than paying for it per paragraph.
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)
        audio_array = None

        prepared_text = self.prepare_text_for_tts(text)
        if control_prompt is None:
            control_prompt = self.build_control_prompt(voice_description, emotion=emotion)

        # 1. Try remote API first if configured
        if self.api_url:
            try:
                audio_array = await self._call_remote_api(
                    text=prepared_text,
                    voice_description=voice_description,
                    reference_audio=reference_audio,
                    emotion=emotion,
                    control_prompt=control_prompt,
                )
            except Exception as e:
                logger.warning(f"Remote VoxCPM API error: {e}")

        # 2. Try local model
        if audio_array is None:
            model = self._get_local_model()
            if model is not None:
                try:
                    kwargs: dict[str, Any] = {"min_len": self._min_len_for(prepared_text)}

                    if reference_audio and Path(reference_audio).exists():
                        kwargs["reference_wav_path"] = str(reference_audio)
                        if emotion:
                            emotion_ctrl = self.build_control_prompt(None, emotion=emotion)
                            kwargs["text"] = self.format_designed_text(prepared_text, emotion_ctrl)
                        else:
                            kwargs["text"] = prepared_text
                    else:
                        # Voice Design / Prompt-to-Voice mode
                        kwargs["text"] = self.format_designed_text(prepared_text, control_prompt)
                        kwargs["cfg_value"] = self.CFG_VOICE_DESIGN

                    audio_array = await asyncio.to_thread(model.generate, **kwargs)
                    if hasattr(model, "tts_model") and hasattr(model.tts_model, "sample_rate"):
                        self.sample_rate = model.tts_model.sample_rate
                except Exception as e:
                    logger.error(f"Local VoxCPM generation failed: {e}")

        # 3. Fallback to placeholder if unconfigured
        if audio_array is None:
            logger.info("Using synthetic placeholder tone (real model weights not loaded).")
            self.used_placeholder = True
            audio_array = self._generate_synthetic_placeholder(prepared_text)

        # Every branch above refreshes self.sample_rate to match what it produced;
        # read it once here so the fade and the file write cannot disagree.
        rate = self.sample_rate
        if audio_array is not None and len(audio_array) > 0:
            audio_array = self._apply_tail_fadeout(audio_array, rate)

        sf.write(output_file, audio_array, rate)
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
        translator: Optional[Any] = None,
    ) -> Path:
        """
        Synthesize entire chapter into a single master audio file.
        Stitches paragraph audio chunks with natural pauses.
        Uses character reference voice anchors when available, falling back to narrator.

        Control prompts are resolved once per speaker before the loop, so a chapter
        costs at most one LLM call per distinct voice rather than one per paragraph.
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

        # One control prompt per distinct voice, resolved up front and reused.
        # A per-chapter override derives its own control, and therefore its own
        # anchor file; it never touches the series voice.
        narrator_control = await self._resolve_cached_control(
            description=default_desc,
            cached=(
                None
                if voice_description
                else getattr(knowledge, "narrator_voice_control_prompt", None)
            ),
            translator=translator,
        )
        if knowledge is not None and hasattr(knowledge, "narrator_voice_control_prompt"):
            knowledge.narrator_voice_control_prompt = narrator_control
        control_by_speaker: dict = {}

        # Ensure a persistent narrator reference voice exists so all narration
        # paragraphs sound identical. Voice design re-rolls a timbre per utterance,
        # so a reference clip -- not the control prompt -- is what holds a voice
        # steady across a chapter.
        effective_narrator_ref = reference_audio
        if effective_narrator_ref is None and knowledge and getattr(knowledge, "narrator_voice_ref_audio", None):
            k_ref = Path(knowledge.narrator_voice_ref_audio)
            if k_ref.exists() and k_ref.stat().st_size > 0:
                effective_narrator_ref = k_ref

        if effective_narrator_ref is None or not Path(effective_narrator_ref).exists():
            voices_dir = output_dir.parent / "voices"
            # Installs predating content-addressed anchors have a bare
            # narrator_ref.wav. Adopt it rather than re-voicing the series, and
            # record it so it is treated as the reference from now on.
            legacy = voices_dir / "narrator_ref.wav"
            if legacy.exists() and legacy.stat().st_size > 0:
                # Use it for this run so an audiobook in progress keeps its voice,
                # but do not record it as the series reference: a stored reference
                # outranks the control prompt, which would make every later
                # description edit silently do nothing.
                effective_narrator_ref = legacy
            else:
                effective_narrator_ref = await self._narrator_anchor(
                    voices_dir, default_desc, narrator_control
                )

        return await self._synthesize_paragraphs(
            chapter=chapter,
            output_dir=output_dir,
            valid_paras=valid_paras,
            use_translated=use_translated,
            default_desc=default_desc,
            narrator_control=narrator_control,
            effective_narrator_ref=effective_narrator_ref,
            control_by_speaker=control_by_speaker,
            knowledge=knowledge,
            translator=translator,
            progress_callback=progress_callback,
        )

    async def _narrator_anchor(self, voices_dir: Path, description: str, control: str) -> Path:
        """Return the reference clip for this narrator voice, generating it if absent.

        The filename is derived from the control prompt, so an anchor is immutable:
        a different voice is a different file. That removes a whole class of problem
        the previous mutable `narrator_ref.wav` had -- a stale-detection stamp that
        could disagree with the wav, a heuristic to tell a generated file from a
        user upload, two concurrent jobs racing on one path, and a per-chapter copy
        taken to defend against that race. A one-off `--voice` run now gets its own
        file instead of re-rolling the series narrator.
        """
        voices_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256((control or "").encode("utf-8")).hexdigest()[:8]
        anchor = voices_dir / f"narrator_ref.{digest}.wav"
        if anchor.exists() and anchor.stat().st_size > 0:
            return anchor

        logger.info(f"Generating narrator reference voice anchor ({digest})...")
        sample_text = "นี่คือเสียงผู้บรรยายประจำนิยายเรื่องนี้ สำหรับการอ่านออกเสียงภาษาไทย"
        # sf.write is not atomic, so build aside and link into place.
        fd, staging_name = tempfile.mkstemp(
            dir=voices_dir, prefix=f"narrator_ref.{digest}.", suffix=".partial.wav"
        )
        os.close(fd)
        staging = Path(staging_name)
        try:
            await self.synthesize(
                text=sample_text,
                output_file=staging,
                voice_description=description,
                reference_audio=None,
                # Without this the anchor is built from the keyword table and the
                # derived prompt is discarded -- and since every paragraph clones
                # from this anchor, it would influence nothing at all.
                control_prompt=control,
            )
            try:
                # Link rather than replace: os.replace would overwrite an anchor a
                # concurrently running chapter is already cloning from, swapping its
                # narrator part-way through. Whoever creates it first wins, and the
                # loser simply uses that file -- the voices are interchangeable.
                os.link(staging, anchor)
            except FileExistsError:
                logger.debug(f"Narrator anchor {digest} was created concurrently; using it.")
            except OSError as e:
                # Hard links are unsupported on exFAT/FAT32 and some network mounts.
                # Fall back to a rename: it loses the concurrent-build guarantee, but
                # a synthesizable voice beats a failed chapter.
                logger.warning(f"Could not link the narrator anchor ({e}); renaming instead.")
                os.replace(staging, anchor)
        finally:
            staging.unlink(missing_ok=True)
        return anchor

    async def _synthesize_paragraphs(
        self,
        chapter: Chapter,
        output_dir: Path,
        valid_paras: List[Paragraph],
        use_translated: bool,
        default_desc: str,
        narrator_control: str,
        effective_narrator_ref,
        control_by_speaker: dict,
        knowledge: Optional[Any],
        translator: Optional[Any],
        progress_callback: Optional[Callable[[int, str], Any]],
    ) -> Path:
        """Synthesize each paragraph and stitch the chapter."""
        final_file = output_dir / f"chapter_{chapter.id}.wav"
        audio_segments: List[np.ndarray] = []
        chunk_rate: Optional[int] = None
        total = max(len(valid_paras), 1)

        for idx, p in enumerate(valid_paras):
            text_to_speak = (p.translated_text if use_translated else p.text).strip()
            
            # Progress reporting
            pct = int((idx / total) * 95)
            speaker_tag = f"[{p.speaker}] " if p.speaker else ""
            short_text = (text_to_speak[:25] + "...") if len(text_to_speak) > 25 else text_to_speak
            await self._report(progress_callback, pct, f"Synthesizing {idx + 1}/{total}: {speaker_tag}{short_text}")

            # Determine voice & emotion for this paragraph (character-specific voice or narrator)
            para_voice_desc = default_desc
            para_control = narrator_control
            para_ref_audio = effective_narrator_ref
            para_emotion = p.emotion

            # Per-character voices. Inert for now: nothing assigns Paragraph.speaker,
            # so this branch, control_by_speaker and the emotion suffix never run.
            # Speaker detection is a separate change; this is kept wired and tested
            # so it works the moment paragraphs carry a speaker.
            if knowledge and p.speaker and p.speaker.strip().lower() not in ("narrator", "ผู้บรรยาย"):
                char = knowledge.find_character(p.speaker)
                if char:
                    # Check if character has dedicated reference audio anchor
                    char_ref = None
                    if char.voice_ref_audio and Path(char.voice_ref_audio).exists():
                        char_ref = Path(char.voice_ref_audio)
                    else:
                        voices_dir = output_dir.parent / "voices"
                        # Use the shared helper: it strips path separators, which a
                        # bare whitespace regex does not. Character names can come
                        # from the LLM auto-learn path.
                        safe_k = character_voice_key(char.name_en)
                        for ext in [".wav", ".mp3", ".m4a", ".flac"]:
                            cand = voices_dir / f"{safe_k}_ref{ext}"
                            if cand.exists() and cand.stat().st_size > 0:
                                char_ref = cand
                                break

                    if char_ref:
                        para_ref_audio = char_ref

                    if char.voice_description:
                        para_voice_desc = char.voice_description
                        key = char.name_en
                        if key not in control_by_speaker:
                            resolved = await self._resolve_cached_control(
                                description=char.voice_description,
                                cached=getattr(char, "voice_control_prompt", None),
                                translator=translator,
                            )
                            char.voice_control_prompt = resolved
                            control_by_speaker[key] = resolved
                        para_control = control_by_speaker[key]

            # Key chunks by Paragraph.index so /api/audio/.../para/{index} resolves them.
            chunk_file = output_dir / f"para_{chapter.id}_{p.index}.wav"
            # An emotion is per-paragraph, so it cannot come from the cached control.
            effective_control = para_control
            if para_emotion:
                emotion_traits = self._emotion_traits(para_emotion)
                if emotion_traits:
                    combined = f"{para_control}, {emotion_traits}"
                    # Keep the same ceiling the derived prompt was validated against.
                    if len(combined) <= self.MAX_CONTROL_PROMPT_CHARS:
                        effective_control = combined

            await self.synthesize(
                text=text_to_speak,
                output_file=chunk_file,
                voice_description=para_voice_desc,
                reference_audio=para_ref_audio,
                emotion=para_emotion,
                control_prompt=effective_control,
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
