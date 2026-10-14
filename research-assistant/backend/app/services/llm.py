"""Provider-agnostic LLM adapter (Part 19). Anthropic / OpenAI-compatible / Gemini, or `none` for offline heuristics.

All callers use `llm.available` to choose between an LLM path and a deterministic fallback, so the whole system
runs end-to-end without any API key (extractive mode) and gets better when a key is configured.
"""
import json
import re

import httpx

from ..config import settings

_DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4o-mini", "gemini": "gemini-2.0-flash"}


class LLMClient:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.model = settings.LLM_MODEL or _DEFAULT_MODELS.get(self.provider, "")

    @property
    def available(self) -> bool:
        key = {"anthropic": settings.ANTHROPIC_API_KEY, "openai": settings.OPENAI_API_KEY,
               "gemini": settings.GEMINI_API_KEY}.get(self.provider)
        return bool(key)

    def complete(self, system: str, user: str, max_tokens: int = 1200, temperature: float = 0.0) -> str:
        if not self.available:
            raise RuntimeError("LLM not configured")
        with httpx.Client(timeout=90) as c:
            if self.provider == "anthropic":
                r = c.post("https://api.anthropic.com/v1/messages",
                           headers={"x-api-key": settings.ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01"},
                           json={"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
                                 "system": system, "messages": [{"role": "user", "content": user}]})
                r.raise_for_status()
                return "".join(b.get("text", "") for b in r.json()["content"])
            if self.provider == "openai":
                r = c.post(f"{settings.OPENAI_BASE_URL}/chat/completions",
                           headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
                           json={"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
                                 "messages": [{"role": "system", "content": system},
                                              {"role": "user", "content": user}]})
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            if self.provider == "gemini":
                r = c.post(f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                           params={"key": settings.GEMINI_API_KEY},
                           json={"systemInstruction": {"parts": [{"text": system}]},
                                 "contents": [{"role": "user", "parts": [{"text": user}]}],
                                 "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature}})
                r.raise_for_status()
                return "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])
        raise RuntimeError(f"Unknown provider {self.provider}")

    def complete_json(self, system: str, user: str, max_tokens: int = 1500) -> dict | list | None:
        """Structured-output prompting: ask for JSON only, strip fences, parse defensively."""
        try:
            raw = self.complete(system + "\nReturn ONLY valid JSON. No prose, no markdown fences.", user, max_tokens)
        except Exception:
            return None
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
        m = re.search(r"[\[{].*[\]}]", raw, re.S)
        try:
            return json.loads(m.group(0) if m else raw)
        except Exception:
            return None


llm = LLMClient()
