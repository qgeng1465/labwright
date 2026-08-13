"""Thin OpenAI-compatible chat client.

Defaults to DeepSeek (OpenAI-compatible endpoint). Point ``base_url`` and
``model`` elsewhere to use any other provider (OpenAI, local vLLM, Ollama, ...).
The API key is read from ``LABWRIGHT_API_KEY`` or ``DEEPSEEK_API_KEY``.
"""

from __future__ import annotations

import os
import time

import openai
from openai import OpenAI

#: Transient failures worth retrying — rate limits (429), timeouts, 5xx and
#: connection errors. Without retries a rate-limited row silently scores as a
#: failure, biasing the benchmark and the API comparison down.
_RETRYABLE = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
    openai.APIStatusError,
)
_MAX_RETRIES = 3


class LLMClient:
    """A minimal chat-completion client exposing tool-calling."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        disable_thinking: bool = True,
    ) -> None:
        self.api_key = api_key or os.environ.get("LABWRIGHT_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key found. Set LABWRIGHT_API_KEY (or DEEPSEEK_API_KEY), or pass api_key=."
            )
        self.base_url = base_url or os.environ.get("LABWRIGHT_BASE_URL") or "https://api.deepseek.com"
        self.model = model or os.environ.get("LABWRIGHT_MODEL") or "deepseek-v4-flash"
        # Providers differ: DeepSeek accepts temperature 0.2; Kimi Code only
        # allows 0.6 and rejects any other value with a 400. The benchmark
        # defaults to 0.2; set LABWRIGHT_TEMPERATURE to override per provider.
        self.temperature = temperature if temperature is not None else float(
            os.environ.get("LABWRIGHT_TEMPERATURE", "0.2")
        )
        # DeepSeek v4 models run hidden reasoning by default and can spend the
        # whole output budget thinking before emitting anything. For a design
        # copilot the arithmetic lives in the calculators, not the model, so we
        # turn reasoning off: ~30x faster and it actually answers. Set
        # LABWRIGHT_DISABLE_THINKING=0 to restore reasoning for any model.
        self.disable_thinking = disable_thinking and os.environ.get("LABWRIGHT_DISABLE_THINKING", "1") != "0"
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(self, messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 8192):
        """One chat-completion call, with exponential-backoff retry; returns the raw assistant message."""
        kwargs: dict = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            kwargs["tools"] = tools
        kwargs["max_tokens"] = max_tokens
        if self.disable_thinking:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        last: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return self._client.chat.completions.create(**kwargs).choices[0].message
            except _RETRYABLE as exc:
                last = exc
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(1.5 * (2 ** attempt))  # 1.5s, 3s, then give up
        raise last  # type: ignore[misc]


__all__ = ["LLMClient"]
