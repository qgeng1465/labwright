"""Thin OpenAI-compatible chat client.

Defaults to DeepSeek (OpenAI-compatible endpoint). Point ``base_url`` and
``model`` elsewhere to use any other provider (OpenAI, local vLLM, Ollama, ...).
The API key is read from ``LABWRIGHT_API_KEY`` or ``DEEPSEEK_API_KEY``.
"""

from __future__ import annotations

import os

from openai import OpenAI


class LLMClient:
    """A minimal chat-completion client exposing tool-calling."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> None:
        self.api_key = api_key or os.environ.get("LABWRIGHT_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key found. Set LABWRIGHT_API_KEY (or DEEPSEEK_API_KEY), or pass api_key=."
            )
        self.base_url = base_url or os.environ.get("LABWRIGHT_BASE_URL") or "https://api.deepseek.com"
        self.model = model or os.environ.get("LABWRIGHT_MODEL") or "deepseek-chat"
        self.temperature = temperature
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(self, messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 2048):
        """One chat-completion call; returns the raw assistant message."""
        kwargs: dict = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            kwargs["tools"] = tools
        kwargs["max_tokens"] = max_tokens
        return self._client.chat.completions.create(**kwargs).choices[0].message


__all__ = ["LLMClient"]
