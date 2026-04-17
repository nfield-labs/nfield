"""
FormatShield LangGraph integration.

Drop-in node for LangGraph workflows that automatically applies
FormatShield routing for structured output.

Usage::

    from formatshield.integrations.langgraph import FormatShieldNode

    node = FormatShieldNode(model="groq/llama-3.3-70b-versatile")
    # Add as a node in a LangGraph StateGraph
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class FormatShieldNode:
    """
    LangGraph-compatible node that routes through FormatShield.

    This integration wraps FormatShield's generate() method as a callable
    node compatible with LangGraph StateGraph. It does not require LangGraph
    to be installed.

    Parameters
    ----------
    model:
        Model identifier in ``"provider/model"`` format.
    schema:
        Optional JSON schema for structured output.
    prompt_key:
        Key in the state dict to use as the prompt. Defaults to ``"prompt"``.
    output_key:
        Key in the state dict to write the response to. Defaults to ``"response"``.
    **kwargs:
        Additional keyword arguments forwarded to :class:`~formatshield.core.FormatShield`.

    Example::

        from formatshield.integrations.langgraph import FormatShieldNode
        from langgraph.graph import StateGraph

        node = FormatShieldNode(
            model="groq/llama-3.3-70b-versatile",
            schema={"type": "object", "properties": {"answer": {"type": "string"}}},
        )
        graph = StateGraph(dict)
        graph.add_node("llm", node)
    """

    def __init__(
        self,
        model: str = "groq/llama-3.3-70b-versatile",
        schema: dict[str, Any] | None = None,
        prompt_key: str = "prompt",
        output_key: str = "response",
        **kwargs: Any,
    ) -> None:
        from formatshield.core import FormatShield

        self._shield = FormatShield(model=model, **kwargs)
        self.model = model
        self._schema = schema
        self._prompt_key = prompt_key
        self._output_key = output_key

    def __call__(self, state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Invoke the node synchronously with a LangGraph state dict.

        Args:
            state: LangGraph state dictionary containing the prompt.
            **kwargs: Additional kwargs passed to FormatShield.

        Returns:
            Updated state dict with the response written to output_key.
        """
        prompt = str(state.get(self._prompt_key, ""))
        result = self._shield.generate_sync(prompt, schema=self._schema, **kwargs)
        return {**state, self._output_key: result.output}

    async def ainvoke(self, state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Invoke the node asynchronously with a LangGraph state dict.

        Args:
            state: LangGraph state dictionary containing the prompt.
            **kwargs: Additional kwargs passed to FormatShield.

        Returns:
            Updated state dict with the response written to output_key.
        """
        prompt = str(state.get(self._prompt_key, ""))
        result = await self._shield.generate(prompt, schema=self._schema, **kwargs)
        return {**state, self._output_key: result.output}

    def invoke(self, state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Synchronous invoke alias for LangGraph compatibility.

        Args:
            state: LangGraph state dictionary.
            **kwargs: Additional kwargs.

        Returns:
            Updated state dict.
        """
        return self(state, **kwargs)
