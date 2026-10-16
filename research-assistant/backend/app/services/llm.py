"""Provider-agnostic LLM adapter. Anthropic / OpenAI-compatible / Groq / Gemini, or `none`.

All callers use `llm.available` to choose between an LLM path and a deterministic fallback, so the whole system
runs end-to-end without any API key (extractive mode) and gets better when a key is configured.
"""
import json
import re

import httpx

from ..config import settings

_DEFAULT_MODELS = {"anthropic": "claude-sonnet-5", "openai": "gpt-4o-mini",
                   "groq": "openai/gpt-oss-20b", "gemini": "gemini-3.8-flash"}


class LLMClient:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.model = settings.LLM_MODEL or _DEFAULT_MODELS.get(self.provider, "")

    @property
    def available(self) -> bool:
        key = {"anthropic": settings.ANTHROPIC_API_KEY, "openai": settings.OPENAI_API_KEY,
               "groq": settings.GROQ_API_KEY, "gemini": settings.GEMINI_API_KEY}.get(self.provider)
        return bool(key)

    def complete(self, system: str, user: str, max_tokens: int = 1200, temperature: float = 0.0) -> str:
        if not self.available:
            raise RuntimeError("LLM not configured")
        with httpx.Client(timeout=90) as c:
            if self.provider == "anthropic":
                r = c.post("https://api.anthropic.com/v1/messages",
                           headers={"x-api-key": settings.ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01"},
                           json={"model": self.model, "max_tokens": max_tokens,
                                 "system": system, "messages": [{"role": "user", "content": user}]})
                r.raise_for_status()
                return "".join(b.get("text", "") for b in r.json()["content"])
            if self.provider in ("openai", "groq"):
                is_groq = self.provider == "groq"
                base_url = settings.GROQ_BASE_URL if is_groq else settings.OPENAI_BASE_URL
                key = settings.GROQ_API_KEY if is_groq else settings.OPENAI_API_KEY
                payload = {"model": self.model, "temperature": temperature,
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": user}]}
                payload["max_completion_tokens" if is_groq else "max_tokens"] = max_tokens
                if is_groq and self.model.startswith("openai/gpt-oss-"):
                    # Groq counts reasoning and visible output in the same limit.
                    # Preserve the callers' answer budget with bounded reasoning headroom.
                    payload["max_completion_tokens"] += 1024
                    payload.update(reasoning_effort="low", include_reasoning=False)
                r = c.post(f"{base_url.rstrip('/')}/chat/completions",
                           headers={"Authorization": f"Bearer {key}"}, json=payload)
                r.raise_for_status()
                choice = r.json()["choices"][0]
                content = choice["message"].get("content")
                if choice.get("finish_reason") == "length" or not isinstance(content, str) or not content.strip():
                    raise RuntimeError("LLM returned an empty or truncated completion")
                return content
            if self.provider == "gemini":
                r = c.post(f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                           params={"key": settings.GEMINI_API_KEY},
                           json={"systemInstruction": {"parts": [{"text": system}]},
                                 "contents": [{"role": "user", "parts": [{"text": user}]}],
                                 "generationConfig": {"maxOutputTokens": max_tokens}})
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


def get_verifier() -> LLMClient:
    """Use the configured verifier model with the same explicitly selected provider/key."""
    client = LLMClient()
    client.provider = llm.provider
    client.model = settings.LLM_VERIFIER_MODEL or llm.model
    return client
