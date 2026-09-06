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
    model = os.getenv("OPENROUTER_MODEL", "minimax/minimax-m3:free")
    voxcpm_url = os.getenv("VOXCPM_API_URL", "")
    voxcpm_device = os.getenv("VOXCPM_DEVICE", "auto")
    readtoon_auth_token = os.getenv("READTOON_AUTH_TOKEN", "")
    readtoon_device_token = os.getenv("READTOON_DEVICE_TOKEN", "")
    readtoon_auto_purchase = os.getenv("READTOON_AUTO_PURCHASE", "").lower() in ("1", "true", "yes")

    from vox_novel.scrapers.readtoon import ReadtoonScraper
    profile_dir = ReadtoonScraper.get_profile_dir()
    readtoon_profile_exists = profile_dir.exists() and any(profile_dir.iterdir())

    return {
        "openrouter_api_key": mask_key(openrouter_key),
        "openrouter_api_key_set": bool(openrouter_key.strip()),
        "openrouter_model": model,
        "gemini_api_key": mask_key(gemini_key),
        "gemini_api_key_set": bool(gemini_key.strip()),
        "voxcpm_api_url": voxcpm_url,
        "voxcpm_device": voxcpm_device,
        "readtoon_auth_token": mask_key(readtoon_auth_token) if readtoon_auth_token else "",
        "readtoon_auth_token_set": bool(readtoon_auth_token.strip()),
        "readtoon_device_token": mask_key(readtoon_device_token) if readtoon_device_token else "",
        "readtoon_auto_purchase": readtoon_auto_purchase,
        "readtoon_profile_exists": readtoon_profile_exists,
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

    if "readtoon_auth_token" in data:
        new_readtoon_token = data.get("readtoon_auth_token", "").strip()
        if new_readtoon_token and "..." not in new_readtoon_token:
            updates["READTOON_AUTH_TOKEN"] = new_readtoon_token
            os.environ["READTOON_AUTH_TOKEN"] = new_readtoon_token
        elif new_readtoon_token == "":
            updates["READTOON_AUTH_TOKEN"] = ""
            os.environ["READTOON_AUTH_TOKEN"] = ""

    if "readtoon_device_token" in data:
        new_device_token = data.get("readtoon_device_token", "").strip()
        if new_device_token and "..." not in new_device_token:
            updates["READTOON_DEVICE_TOKEN"] = new_device_token
            os.environ["READTOON_DEVICE_TOKEN"] = new_device_token
        elif new_device_token == "":
            updates["READTOON_DEVICE_TOKEN"] = ""
            os.environ["READTOON_DEVICE_TOKEN"] = ""

    if "readtoon_auto_purchase" in data:
        val = "true" if data.get("readtoon_auto_purchase") in (True, "true", "1") else "false"
        updates["READTOON_AUTO_PURCHASE"] = val
        os.environ["READTOON_AUTO_PURCHASE"] = val


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


async def verify_readtoon_session(token: Optional[str] = None) -> Dict[str, Any]:
    """Check whether the provided token or persistent profile is authenticated on ReadToon."""
    import json
    from vox_novel.scrapers.readtoon import ReadtoonScraper

    profile_dir = ReadtoonScraper.get_profile_dir()
    profile_exists = profile_dir.exists() and any(profile_dir.iterdir())
    session_file = profile_dir / "user_session.json"
    storage_file = profile_dir / "storage_state.json"

    cookies = {}

    # 1. If explicit token passed
    key = (token or "").strip()
    if key and "..." not in key:
        if ";" in key or "=" in key:
            for part in key.split(";"):
                if "=" in part:
                    k, v = part.strip().split("=", 1)
                    cookies[k.strip()] = v.strip().strip("'\"")
        else:
            cookies["auth_token"] = key
    else:
        # 2. Extract from storage_state.json if available
        if storage_file.exists():
            try:
                st = json.loads(storage_file.read_text(encoding="utf-8"))
                for c in st.get("cookies", []):
                    if c.get("name") in ("auth_token", "device_token"):
                        cookies[c["name"]] = c["value"]
            except Exception:
                pass

        # 3. Extract from environment variables if not already set
        if "auth_token" not in cookies and os.getenv("READTOON_AUTH_TOKEN"):
            cookies["auth_token"] = os.getenv("READTOON_AUTH_TOKEN", "").strip()
        if "device_token" not in cookies and os.getenv("READTOON_DEVICE_TOKEN"):
            cookies["device_token"] = os.getenv("READTOON_DEVICE_TOKEN", "").strip()

    if "auth_token" in cookies:
        try:
            async with httpx.AsyncClient(
                timeout=10.0,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                    "Referer": "https://readtoon.com/",
                },
            ) as client:
                resp = await client.get(
                    "https://readtoon.com/api/trpc/user.profile.getMe?input=%7B%22json%22%3Anull%2C%22meta%22%3A%7B%22values%22%3A%5B%22undefined%22%5D%2C%22v%22%3A1%7D%7D",
                    cookies=cookies,
                )
                if resp.status_code == 200:
                    data = resp.json().get("result", {}).get("data", {}).get("json", {})
                    if data and (data.get("id") or data.get("nickname") or data.get("username")):
                        name = data.get("nickname") or data.get("username")
                        raw_coins = data.get("coins") if data.get("coins") is not None else (data.get("coin") or 0)
                        try:
                            c_val = float(raw_coins)
                            coins = str(int(c_val)) if c_val.is_integer() else f"{c_val:.2f}"
                        except Exception:
                            coins = str(raw_coins)

                        user_info = {"id": data.get("id"), "nickname": name, "coins": coins}
                        try:
                            session_file.write_text(json.dumps(user_info, ensure_ascii=False), encoding="utf-8")
                        except Exception:
                            pass

                        return {
                            "valid": True,
                            "nickname": name,
                            "coins": coins,
                            "message": f"Authenticated as {name} ({coins} coins)",
                            "mode": "persistent" if profile_exists else "token",
                        }
                elif resp.status_code == 401:
                    if session_file.exists():
                        try:
                            session_file.unlink()
                        except Exception:
                            pass
                    return {
                        "valid": False,
                        "error": "Unauthenticated (กรุณาเข้าสู่ระบบ). The ReadToon session is expired or invalid. Please log in.",
                        "mode": "token",
                    }
        except Exception as e:
            if token is not None:
                return {"valid": False, "error": f"Connection error checking ReadToon: {str(e)}", "mode": "token"}

    if session_file.exists():
        try:
            session_file.unlink()
        except Exception:
            pass

    return {
        "valid": False,
        "error": "Not logged in. Please click 'Log In to ReadToon' or run 'vox-novel login readtoon'.",
        "mode": "none",
    }
