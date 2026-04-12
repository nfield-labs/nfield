"""
FormatShield LangChain integration.

Drop-in replacement for ChatGroq / ChatOpenAI that automatically applies
FormatShield routing.

Usage::

    from formatshield.integrations.langchain import FormatShieldLLM

    llm = FormatShieldLLM(model="groq/llama-3.1-70b-versatile")
    chain = prompt_template | llm | output_parser
    result = chain.invoke({"input": user_message})
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger(__name__)


class FormatShieldLLM:
    """
    LangChain-compatible LLM wrapper that routes through FormatShield.

    This is a minimal integration that wraps FormatShield's generate() method
    in a LangChain-compatible interface.  It does not require LangChain to be
    installed — it only imports LangChain types when they are available.

    Parameters
    ----------
    model:
        Model identifier in ``"provider/model"`` format.
    debug:
        When ``True``, prints FormatShield routing traces.
    **kwargs:
        Additional keyword arguments forwarded to :class:`~formatshield.core.FormatShield`.

    Example::

        from formatshield.integrations.langchain import FormatShieldLLM
        from langchain_core.prompts import ChatPromptTemplate

        llm = FormatShieldLLM(model="groq/llama-3.1-70b-versatile")
        prompt = ChatPromptTemplate.from_messages([("human", "{input}")])
        chain = prompt | llm
        result = chain.invoke({"input": "What is 2+2?"})
    """

    def __init__(self, model: str = "groq/llama-3.1-70b-versatile", **kwargs: Any) -> None:
        from formatshield.core import FormatShield

        self._shield = FormatShield(model=model, **kwargs)
        self.model = model

    def invoke(self, input: str | dict[str, Any], **kwargs: Any) -> str:
        """Synchronous invocation compatible with LangChain chains."""
        prompt = self._extract_prompt(input)
        result = self._shield.generate_sync(prompt)
        return result.output

    async def ainvoke(self, input: str | dict[str, Any], **kwargs: Any) -> str:
        """Asynchronous invocation compatible with LangChain async chains."""
        prompt = self._extract_prompt(input)
        result = await self._shield.generate(prompt)
        return result.output

    def stream(self, input: str | dict[str, Any], **kwargs: Any) -> Iterator[str]:
        """Streaming invocation."""
        prompt = self._extract_prompt(input)

        async def _collect() -> list[str]:
            tokens: list[str] = []
            async for event in self._shield.stream(prompt):
                if event.type == "output" and event.token:
                    tokens.append(event.token)
            return tokens

        tokens = asyncio.run(_collect())
        yield from tokens

    def _extract_prompt(self, input: str | dict[str, Any]) -> str:
        """Extract prompt string from various LangChain input formats."""
        if isinstance(input, str):
            return input
        if isinstance(input, dict):
            for key in ("input", "text", "content", "prompt", "question"):
                if key in input:
                    val = input[key]
                    return str(val)
            # Fallback: join all string values
            return " ".join(str(v) for v in input.values() if isinstance(v, str))
        return str(input)

    # Make it work as a LangChain Runnable (duck-typing)
    def __or__(self, other: Any) -> Any:
        """Support pipe operator: llm | parser."""
        try:
            from langchain_core.runnables import RunnableSequence  # type: ignore[import]

            return RunnableSequence(first=self, last=other)
        except ImportError as exc:
            raise ImportError(
                "pip install langchain-core to use the | operator with FormatShieldLLM"
            ) from exc
