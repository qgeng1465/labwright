"""The Labwright agent — an LLM that can only *propose* numbers, never invent them.

The agent's brain is a pluggable OpenAI-compatible API (DeepSeek by default).
It may call any registered calculator tool and must finish by calling
``submit_design`` with raw inputs; Labwright then derives and verifies every
number itself.
"""

from labwright.agent.agent import AgentResult, DesignAgent
from labwright.agent.llm import LLMClient

__all__ = ["AgentResult", "DesignAgent", "LLMClient"]
