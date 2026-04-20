"""
RCOCR Engine — Reasoning-Compatible Output Constraint Recovery.

The RCOCR engine implements the two-pass generation pattern that decouples
reasoning from structured formatting:

  Pass 1 — Unconstrained reasoning
    The model reasons freely inside <think>...</think> tags with no JSON
    grammar or schema constraints.  This preserves the full reasoning
    capability of the model.

  Pass 2 — Constrained formatting
    The model reads its own reasoning and extracts the final answer as a
    structured JSON object.  The reasoning context is provided as a prefix,
    allowing the model to stay grounded in its own chain-of-thought.

This architecture solves the grammar-reasoning conflict documented in:
- vLLM issue #34650 (reasoning + structured output)
- llama.cpp issue #12204 (grammar disabled with reasoning_format)
- CRANE paper (arXiv:2502.09061)

Usage::

    import asyncio
    from rcocr import RCOCREngine

    engine = RCOCREngine(backend=my_backend)
    thinking, output = asyncio.run(
        engine.generate(
            prompt="Extract the order details.",
            schema={"type": "object", "properties": {"order_id": {"type": "string"}}},
        )
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rcocr.protocol import RCOCRBackend

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_THINK_INSTRUCTION = (
    "Think step by step inside <think>...</think> tags. "
    "Reason carefully before producing any structured output. "
    "Do NOT produce any JSON inside the think block."
)

_FORMAT_INSTRUCTION = (
    "Based on your reasoning above, produce ONLY a valid JSON object that "
    "matches the required schema. Output nothing other than the JSON object."
)


def _build_think_prompt(prompt: str) -> str:
    """Build the Pass 1 unconstrained reasoning prompt."""
    return f"{_THINK_INSTRUCTION}\n\n{prompt}"


def _build_format_prompt(
    original_prompt: str,
    thinking: str,
    schema: dict[str, Any] | None,
) -> str:
    """Build the Pass 2 constrained formatting prompt."""
    parts: list[str] = []

    if thinking:
        parts.append(f"<think>{thinking}</think>")

    parts.append(original_prompt)

    if schema is not None:
        schema_text = json.dumps(schema, indent=2)
        parts.append(f"Schema:\n{schema_text}")

    parts.append(_FORMAT_INSTRUCTION)
    return "\n\n".join(parts)


def _extract_thinking(raw: str) -> str:
    """Extract content from the first ``<think>...</think>`` block."""
    match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # No tags — treat everything before the first '{' as thinking
    brace_idx = raw.find("{")
    if brace_idx > 0:
        return raw[:brace_idx].strip()
    return raw.strip()


# ---------------------------------------------------------------------------
# Self-consistency helpers
# ---------------------------------------------------------------------------


async def _self_consistency_pass1(
    backend: RCOCRBackend,
    think_prompt: str,
    k: int,
    **generate_kwargs: Any,
) -> tuple[str, str]:
    """Run *k* parallel Pass 1 calls and return the best ``(thinking, raw)`` pair.

    "Best" is defined as the longest extracted thinking text, which correlates
    with more thorough reasoning.  When ``k == 1`` a single call is made.
    """
    if k <= 1:
        raw = await backend.generate(think_prompt, constraints=None, **generate_kwargs)
        return _extract_thinking(raw), raw

    tasks = [
        backend.generate(think_prompt, constraints=None, **generate_kwargs)
        for _ in range(k)
    ]
    results: list[str] = await asyncio.gather(*tasks, return_exceptions=False)

    best_raw = results[0]
    best_thinking = _extract_thinking(best_raw)
    for raw in results[1:]:
        thinking = _extract_thinking(raw)
        if len(thinking) > len(best_thinking):
            best_thinking = thinking
            best_raw = raw

    return best_thinking, best_raw


# ---------------------------------------------------------------------------
# RCOCREngine
# ---------------------------------------------------------------------------


class RCOCREngine:
    """Two-pass RCOCR generation engine.

    Parameters
    ----------
    backend:
        Any object implementing :class:`~rcocr.protocol.RCOCRBackend`.
    self_consistency_k:
        Number of parallel Pass 1 traces to generate and select from.
        ``1`` (default) disables self-consistency.  Values ≥ 2 run *k*
        parallel calls and pick the most thorough reasoning trace.

    Example::

        engine = RCOCREngine(backend=my_backend, self_consistency_k=3)
        thinking, output = await engine.generate(
            prompt="Extract the order.",
            schema={"type": "object", "properties": {"id": {"type": "string"}}},
        )
    """

    def __init__(
        self,
        backend: RCOCRBackend,
        self_consistency_k: int = 1,
    ) -> None:
        self._backend = backend
        self._k = max(1, self_consistency_k)

    async def generate(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
    ) -> tuple[str, str]:
        """Run two-pass RCOCR generation.

        Parameters
        ----------
        prompt:
            The original user prompt.
        schema:
            Optional JSON Schema dict for the expected output structure.
        temperature:
            Sampling temperature forwarded to both passes.
        max_tokens:
            Token limit for Pass 1.  ``None`` uses the backend default.
        seed:
            Random seed for reproducibility.

        Returns
        -------
        tuple[str, str]
            ``(thinking_text, json_output)`` where *thinking_text* is the
            extracted reasoning and *json_output* is the raw JSON string.
        """
        generate_kwargs: dict[str, Any] = {}
        if temperature is not None:
            generate_kwargs["temperature"] = temperature
        if max_tokens is not None:
            generate_kwargs["max_tokens"] = max_tokens
        if seed is not None:
            generate_kwargs["seed"] = seed

        think_prompt = _build_think_prompt(prompt)

        logger.debug(
            "RCOCR: Pass 1 — unconstrained reasoning (k=%d backend=%s)",
            self._k,
            self._backend.name,
        )

        thinking_text, _raw_thinking = await _self_consistency_pass1(
            self._backend,
            think_prompt,
            self._k,
            **generate_kwargs,
        )

        logger.debug("RCOCR: Pass 1 complete — %d chars of thinking", len(thinking_text))

        format_prompt = _build_format_prompt(prompt, thinking_text, schema)

        logger.debug("RCOCR: Pass 2 — constrained formatting (backend=%s)", self._backend.name)

        json_output = await self._backend.generate(
            format_prompt,
            constraints="json",
            **generate_kwargs,
        )

        logger.debug("RCOCR: Pass 2 complete — %d chars of JSON", len(json_output))

        return thinking_text, json_output
