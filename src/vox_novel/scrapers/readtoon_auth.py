import asyncio
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional


class ReadtoonAuthManager:
    """Manages ReadToon authentication session, interactive browser login, and profile persistence."""

    _instance: Optional["ReadtoonAuthManager"] = None

    def __init__(self):
        self.status: str = "idle"  # idle, running, success, error, cancelled
        self.message: str = ""
        self.user: Optional[Dict[str, Any]] = None
        self._cancel_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._context = None

        session_file = self.get_profile_dir() / "user_session.json"
        if session_file.exists():
            try:
                import json
                self.user = json.loads(session_file.read_text(encoding="utf-8"))
            except Exception:
                pass

    @classmethod
    def get_instance(cls) -> "ReadtoonAuthManager":
        if cls._instance is None:
            cls._instance = ReadtoonAuthManager()
        return cls._instance

    @staticmethod
    def get_profile_dir() -> Path:
        from vox_novel.scrapers.readtoon import ReadtoonScraper
        return ReadtoonScraper.get_profile_dir()

    async def get_current_session(self) -> Dict[str, Any]:
        """Check if there is an active session right now by testing live validity."""
        from vox_novel.settings import verify_readtoon_session
        res = await verify_readtoon_session()
        if res.get("valid"):
            self.user = {
                "nickname": res.get("nickname"),
                "coins": res.get("coins"),
            }
        else:
            self.user = None
        return res

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "user": self.user,
        }

    async def start_browser_login(self, timeout: int = 300) -> Dict[str, Any]:
        if self.status == "running":
            return {"status": "running", "message": "Login already in progress."}

        self.status = "running"
        self.message = "Launching Chrome browser on your screen..."
        self.user = None
        self._cancel_event.clear()

        # Run login in background asyncio task
        self._task = asyncio.create_task(self._run_login_flow(timeout))
        return {"status": "running", "message": "Browser login initiated."}

    async def cancel_login(self) -> Dict[str, Any]:
        if self.status == "running":
            self.status = "cancelled"
            self.message = "Login cancelled by user."
            self._cancel_event.set()
            if self._context:
                try:
                    await self._context.close()
                except Exception:
                    pass
            if self._task and not self._task.done():
                self._task.cancel()
        return {"status": "cancelled", "message": "Login cancelled."}

    async def logout(self) -> Dict[str, Any]:
        profile_dir = self.get_profile_dir()
        if profile_dir.exists():
            shutil.rmtree(profile_dir, ignore_errors=True)

        from vox_novel.settings import save_app_settings
        save_app_settings({"readtoon_auth_token": "", "readtoon_device_token": ""})
        self.status = "idle"
        self.user = None
        self.message = "Logged out successfully."
        return {"status": "ok", "message": "Logged out and cleared ReadToon session."}

    async def _run_login_flow(self, timeout: int):
        from playwright.async_api import async_playwright
        profile_dir = self.get_profile_dir()
        profile_dir.mkdir(parents=True, exist_ok=True)

        try:
            async with async_playwright() as p:
                context = None
                for opts in [{"channel": "chrome"}, {}]:
                    try:
                        context = await p.chromium.launch_persistent_context(
                            user_data_dir=str(profile_dir),
                            headless=False,
                            args=["--disable-blink-features=AutomationControlled"],
                            ignore_default_args=["--enable-automation"],
                            viewport={"width": 1280, "height": 850},
                            **opts,
                        )
                        break
                    except Exception:
                        continue

                if not context:
                    self.status = "error"
                    self.message = "Failed to launch Google Chrome on your system."
                    return

                self._context = context
                page = context.pages[0] if context.pages else await context.new_page()
                await page.goto("https://readtoon.com/auth/sign-in")

                self.message = "Chrome window is open. Please log into your ReadToon account in Chrome..."
                start_time = time.time()
                logged_in = None

                while time.time() - start_time < timeout:
                    if self._cancel_event.is_set():
                        break

                    try:
                        resp_data = await page.evaluate("""
                            fetch('/api/trpc/user.profile.getMe?input=%7B%22json%22%3Anull%2C%22meta%22%3A%7B%22values%22%3A%5B%22undefined%22%5D%2C%22v%22%3A1%7D%7D')
                                .then(r => r.json())
                                .catch(e => null)
                        """)
                        if resp_data and "result" in resp_data:
                            user_info = resp_data["result"].get("data", {}).get("json", {})
                            if user_info and (user_info.get("id") or user_info.get("nickname") or user_info.get("username")):
                                logged_in = user_info
                                break
                    except Exception:
                        pass

                    await asyncio.sleep(1.5)

                if logged_in:
                    nickname = logged_in.get("nickname") or logged_in.get("username") or "User"
                    raw_coins = logged_in.get("coins") if logged_in.get("coins") is not None else (logged_in.get("coin") or 0)
                    try:
                        c_val = float(raw_coins)
                        coins_display = str(int(c_val)) if c_val.is_integer() else f"{c_val:.2f}"
                    except Exception:
                        coins_display = str(raw_coins)

                    self.user = {
                        "id": logged_in.get("id"),
                        "nickname": nickname,
                        "coins": coins_display,
                    }
                    self.status = "success"
                    self.message = f"Successfully logged in as {nickname} ({coins_display} coins)!"

                    session_file = profile_dir / "user_session.json"
                    try:
                        import json
                        session_file.write_text(json.dumps(self.user, ensure_ascii=False), encoding="utf-8")
                    except Exception:
                        pass

                    try:
                        # Flush full Playwright storage state (cookies + localStorage)
                        await context.storage_state(path=str(profile_dir / "storage_state.json"))
                    except Exception:
                        pass

                    try:
                        cookies = await context.cookies()
                        auth_c = next((c["value"] for c in cookies if c["name"] == "auth_token"), None)
                        device_c = next((c["value"] for c in cookies if c["name"] == "device_token"), None)
                        updates = {}
                        if auth_c:
                            updates["readtoon_auth_token"] = auth_c
                        if device_c:
                            updates["readtoon_device_token"] = device_c
                        if updates:
                            from vox_novel.settings import save_app_settings
                            save_app_settings(updates)
                    except Exception:
                        pass

                    await asyncio.sleep(1)
                else:
                    if self.status != "cancelled":
                        self.status = "error"
                        self.message = "Login timed out after waiting for authentication."

                await context.close()
                self._context = None
        except asyncio.CancelledError:
            self.status = "cancelled"
            self.message = "Login cancelled."
        except Exception as e:
            self.status = "error"
            self.message = f"Error during browser login: {str(e)}"
        finally:
            self._context = None
