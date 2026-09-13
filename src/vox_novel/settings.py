import os
import re
from pathlib import Path
from typing import Dict, Any, Optional
import httpx
from dotenv import load_dotenv

load_dotenv()


def get_env_path() -> Path:
    """Locate the .env file for the project."""
    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir.parents[1] / ".env",  # repo root
        Path.cwd() / ".env",
        Path.home() / ".vox_novel" / ".env",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def mask_key(key: Optional[str]) -> str:
    """Mask an API key for safe display (e.g. sk-or-v1-0c18...f9e)."""
    if not key or len(key) < 10:
        return ""
    return f"{key[:10]}...{key[-4:]}"


def get_app_settings() -> Dict[str, Any]:
    """Return the current application settings for the Web UI."""
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    model = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
    # Optional: run the editor pass on a stronger model than the draft.
    polish_model = os.getenv("OPENROUTER_POLISH_MODEL", "")
    base_url = os.getenv("OPENROUTER_BASE_URL", "")
    voxcpm_url = os.getenv("VOXCPM_API_URL", "")
    voxcpm_device = os.getenv("VOXCPM_DEVICE", "auto")

    return {
        "openrouter_api_key": mask_key(openrouter_key),
        "openrouter_api_key_set": bool(openrouter_key.strip()),
        "openrouter_model": model,
        "openrouter_polish_model": polish_model,
        "openrouter_base_url": base_url,
        "gemini_api_key": mask_key(gemini_key),
        "gemini_api_key_set": bool(gemini_key.strip()),
        "voxcpm_api_url": voxcpm_url,
        "voxcpm_device": voxcpm_device,
    }


def save_app_settings(data: Dict[str, Any]) -> Dict[str, Any]:
    """Persist updated API keys and model configuration to .env and runtime environment."""
    env_path = get_env_path()
    lines = []
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    updates = {}
    new_openrouter_key = data.get("openrouter_api_key", "").strip()
    if new_openrouter_key and "..." not in new_openrouter_key:
        updates["OPENROUTER_API_KEY"] = new_openrouter_key
        os.environ["OPENROUTER_API_KEY"] = new_openrouter_key

    new_model = data.get("openrouter_model", "").strip()
    if new_model:
        updates["OPENROUTER_MODEL"] = new_model
        os.environ["OPENROUTER_MODEL"] = new_model

    # Empty is meaningful here: it means "use the draft model for polish too".
    if "openrouter_base_url" in data:
        new_base = (data.get("openrouter_base_url") or "").strip()
        updates["OPENROUTER_BASE_URL"] = new_base
        os.environ["OPENROUTER_BASE_URL"] = new_base

    if "openrouter_polish_model" in data:
        new_polish = (data.get("openrouter_polish_model") or "").strip()
        updates["OPENROUTER_POLISH_MODEL"] = new_polish
        os.environ["OPENROUTER_POLISH_MODEL"] = new_polish

    new_gemini_key = data.get("gemini_api_key", "").strip()
    if new_gemini_key and "..." not in new_gemini_key:
        updates["GEMINI_API_KEY"] = new_gemini_key
        os.environ["GEMINI_API_KEY"] = new_gemini_key

    if "voxcpm_api_url" in data:
        new_voxcpm_url = data.get("voxcpm_api_url", "").strip()
        updates["VOXCPM_API_URL"] = new_voxcpm_url
        os.environ["VOXCPM_API_URL"] = new_voxcpm_url

    if "voxcpm_device" in data:
        new_voxcpm_device = data.get("voxcpm_device", "auto").strip()
        updates["VOXCPM_DEVICE"] = new_voxcpm_device
        os.environ["VOXCPM_DEVICE"] = new_voxcpm_device

    # Update or append keys in .env
    updated_keys = set()
    new_lines = []
    for line in lines:
        matched = False
        for k, v in updates.items():
            if re.match(rf"^\s*{re.escape(k)}\s*=", line):
                new_lines.append(f'{k}="{v}"\n')
                updated_keys.add(k)
                matched = True
                break
        if not matched:
            new_lines.append(line)

    for k, v in updates.items():
        if k not in updated_keys:
            new_lines.append(f'{k}="{v}"\n')

    env_path.parent.mkdir(parents=True, exist_ok=True)
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    return {"status": "ok", "message": "Settings updated and saved to .env"}


async def verify_openrouter_api_key(api_key: Optional[str] = None) -> Dict[str, Any]:
    """Check whether the provided or saved OpenRouter key is valid and inspect its tier/credits."""
    key = (api_key or "").strip()
    if not key or "..." in key:
        key = os.getenv("OPENROUTER_API_KEY", "")

    if not key:
        return {"valid": False, "error": "No API key provided"}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {key}"},
            )
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                is_free = data.get("is_free_tier", True)
                limit = data.get("limit")
                usage = data.get("usage", 0)
                label = data.get("label", mask_key(key))
                return {
                    "valid": True,
                    "label": label,
                    "is_free_tier": is_free,
                    "limit": limit,
                    "usage": usage,
                    "message": "Key is active and verified!",
                }
            elif resp.status_code == 401:
                return {"valid": False, "error": "Invalid API Key (Unauthorized 401)"}
            else:
                return {"valid": False, "error": f"OpenRouter check failed (Status {resp.status_code})"}
    except Exception as e:
        return {"valid": False, "error": f"Connection error: {str(e)}"}
