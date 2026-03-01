from __future__ import annotations

import os


class LLMClient:
    def __init__(self) -> None:
        self._client, self._kind, self._model = self._init_client()

    # ------------------------------------------------------------------
    def _init_client(self):
        if key := os.getenv("GROQ_API_KEY"):
            from groq import Groq
            model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            return Groq(api_key=key), "groq", model

        if key := os.getenv("OPENAI_API_KEY"):
            from openai import OpenAI
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            return OpenAI(api_key=key), "openai", model

        if key := os.getenv("ANTHROPIC_API_KEY"):
            import anthropic
            model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
            return anthropic.Anthropic(api_key=key), "anthropic", model

        return None, "none", ""

    # ------------------------------------------------------------------
    @property
    def info(self) -> str:
        if self._kind == "none":
            return "no LLM configured (set GROQ_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY)"
        return f"{self._kind} / {self._model}"

    # ------------------------------------------------------------------
    def generate(self, prompt: str, max_tokens: int = 1500) -> str:
        if self._client is None:
            return "LLM unavailable — set an API key in .env"

        try:
            if self._kind in ("groq", "openai"):
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content.strip()

            if self._kind == "anthropic":
                resp = self._client.messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text.strip()

        except Exception as exc:
            return f"LLM unavailable — {exc}"

        return "LLM unavailable"
