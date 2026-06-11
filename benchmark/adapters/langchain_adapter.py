"""LangChain baseline — ``.with_structured_output()`` over the same model.

LangChain wraps a single model call with its structured-output runnable, which
accepts a JSON Schema directly. Like the other Track-A baselines it sends the
whole schema in one call with no decomposition or retrieval.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from . import _common

if TYPE_CHECKING:
    from ._base import AdapterOutput

_TEMPERATURE: float = 0.0


@dataclass(frozen=True, slots=True)
class LangChainAdapter:
    """Single ``ChatGroq.with_structured_output(schema)`` call.

    Args:
        api_key: Provider credential; ``None`` uses the SDK's env pickup.
        base_url: Optional provider base URL.
    """

    name: str = field(default="langchain", init=False)
    api_key: str | None = None
    base_url: str | None = None

    def run(
        self,
        document: str,
        schema: dict[str, Any],
        *,
        model: str,
        context_window: int,
        max_output_tokens: int,
        instructions: str = "",
    ) -> AdapterOutput:
        """Extract via LangChain's structured-output runnable in a single call."""
        from langchain_groq import ChatGroq

        started = time.perf_counter()
        try:
            chat = ChatGroq(
                model=_common.model_id(model),
                api_key=self.api_key,  # type: ignore[arg-type]  # coerced to SecretStr at runtime
                temperature=_TEMPERATURE,
                max_tokens=max_output_tokens,
                max_retries=_common.MAX_TRANSIENT_RETRIES,
            )
            structured = chat.with_structured_output(schema, method="json_mode")
            prompt = _common.messages(
                document,
                schema,
                context_window=context_window,
                max_output_tokens=max_output_tokens,
                instructions=instructions,
            )
            result = structured.invoke([(m["role"], m["content"]) for m in prompt])
            data = result if isinstance(result, dict) else dict(result)
        except Exception as exc:  # record fairly, never abort the sweep
            return _common.failure_output(schema, round(time.perf_counter() - started, 3), exc)
        return _common.success_output(data, schema, round(time.perf_counter() - started, 3))
