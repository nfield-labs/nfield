"""
FormatShield AutoGen integration.

Drop-in replacement for AutoGen's LLM client that automatically applies
FormatShield routing.

Usage::

    from formatshield.integrations.autogen import FormatShieldAutoGen
    from autogen import AssistantAgent

    llm_config = {"config_list": [{"model": "groq/llama-3.3-70b-versatile"}]}
    shield_client = FormatShieldAutoGen(model="groq/llama-3.3-70b-versatile")
    agent = AssistantAgent("assistant", llm_config=llm_config)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class FormatShieldAutoGen:
    """
    AutoGen-compatible LLM client wrapper that routes through FormatShield.

    This integration wraps FormatShield's generate() method in an interface
    compatible with AutoGen's model client protocol. It does not require AutoGen
    to be installed — it only imports AutoGen types when present.

    Parameters
    ----------
    model:
        Model identifier in ``"provider/model"`` format.
    schema:
        Optional JSON schema for structured output.
    debug:
        When ``True``, prints FormatShield routing traces.
    **kwargs:
        Additional keyword arguments forwarded to :class:`~formatshield.core.FormatShield`.

    Example::

        from formatshield.integrations.autogen import FormatShieldAutoGen

        client = FormatShieldAutoGen(model="groq/llama-3.3-70b-versatile")
        result = client.generate_sync("What is 2+2?")
    """

    def __init__(
        self,
        model: str = "groq/llama-3.3-70b-versatile",
        schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        from formatshield.core import FormatShield

        self._shield = FormatShield(model=model, **kwargs)
        self.model = model
        self._schema = schema

    def generate_sync(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response synchronously.

        Args:
            prompt: The user prompt string.
            schema: Optional JSON schema override.
            **kwargs: Additional kwargs passed to FormatShield.generate_sync().

        Returns:
            The model's response text.
        """
        effective_schema = schema if schema is not None else self._schema
        result = self._shield.generate_sync(prompt, schema=effective_schema, **kwargs)
        return result.output

    async def generate(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response asynchronously.

        Args:
            prompt: The user prompt string.
            schema: Optional JSON schema override.
            **kwargs: Additional kwargs passed to FormatShield.generate().

        Returns:
            The model's response text.
        """
        effective_schema = schema if schema is not None else self._schema
        result = await self._shield.generate(prompt, schema=effective_schema, **kwargs)
        return result.output

    def create(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """AutoGen model client protocol: create() method.

        Accepts AutoGen's messages format and returns a response dict
        compatible with AutoGen's expected format.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            **kwargs: Additional kwargs.

        Returns:
            Dict with 'choices' list containing the response.
        """
        prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                prompt = str(msg.get("content", ""))
                break
        if not prompt and messages:
            prompt = str(messages[-1].get("content", ""))

        response_text = self.generate_sync(prompt, **kwargs)
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ],
            "model": self.model,
        }
