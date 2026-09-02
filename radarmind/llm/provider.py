"""DeepSeek via DEEPSEEK_API_KEY. No key / API fail → fallback. Never log secrets."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from radarmind.llm.envload import load_dotenv
from radarmind.llm.fallback import render
from radarmind.llm.prompt import SYSTEM_PROMPT, build_user_payload

load_dotenv()


class LLMService:
    def __init__(self, timeout: float | None = None, retries: int | None = None, force_offline: bool | None = None):
        self.timeout = float(os.environ.get("DEEPSEEK_TIMEOUT", timeout or 30))
        self.retries = int(os.environ.get("DEEPSEEK_RETRIES", retries or 2))
        env_off = os.environ.get("RADARMIND_OFFLINE_DEMO", "").strip().lower() in ("1", "true", "yes")
        self.force_offline = bool(env_off if force_offline is None else force_offline)
        self._ui_mode = "Template Fallback" if self.force_offline or not self.has_key() else "DeepSeek Online"

    def has_key(self) -> bool:
        return bool(os.environ.get("DEEPSEEK_API_KEY", "").strip())

    def model_name(self) -> str:
        return os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat"

    def base_url(self) -> str:
        return os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions").strip()

    def backend(self) -> str:
        return "deepseek" if self._ui_mode == "DeepSeek Online" else "fallback"

    def ui_mode(self) -> str:
        """Public UI label. Only 'DeepSeek Online' or 'Template Fallback'."""
        return self._ui_mode

    def explain(
        self,
        task: str,
        intelligence: dict | None = None,
        timeline: dict | None = None,
        question: str | None = None,
    ) -> dict:
        fallback = render(task, intelligence, timeline, question)
        if self.force_offline or not self.has_key():
            self._ui_mode = "Template Fallback"
            return {"text": fallback, "backend": "fallback", "ok": True, "model": None, "latency_s": 0.0}
        user = build_user_payload(task, intelligence, timeline, question)
        last_err = None
        t0 = time.perf_counter()
        for attempt in range(self.retries + 1):
            try:
                text = self._call_deepseek(user)
                self._ui_mode = "DeepSeek Online"
                return {
                    "text": text,
                    "backend": "deepseek",
                    "ok": True,
                    "model": self.model_name(),
                    "latency_s": time.perf_counter() - t0,
                    "attempts": attempt + 1,
                }
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < self.retries:
                    time.sleep(0.8 * (attempt + 1))
        self._ui_mode = "Template Fallback"
        return {
            "text": fallback,
            "backend": "fallback",
            "ok": False,
            "model": self.model_name(),
            "latency_s": time.perf_counter() - t0,
            "error": type(last_err).__name__,
            "note": "DeepSeek 调用失败，已改用模板说明。",
        }

    def _call_deepseek(self, user: str) -> str:
        key = os.environ["DEEPSEEK_API_KEY"].strip()
        payload = {
            "model": self.model_name(),
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
        }
        req = urllib.request.Request(
            self.base_url(),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"deepseek_http_{exc.code}") from None
        except urllib.error.URLError as exc:
            raise RuntimeError(f"deepseek_url_{type(exc.reason).__name__}") from None
        return body["choices"][0]["message"]["content"].strip()
