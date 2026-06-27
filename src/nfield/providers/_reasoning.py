"""Disable reasoning-model thinking and strip any trace it leaves.

Reasoning models (Qwen3, DeepSeek-R1, QwQ) think before answering. The thinking
shares the answer's output budget, so on a bounded per-call budget it consumes
the budget and the answer is truncated to nothing; left inline it also leaks a
``<think>…</think>`` block a line-based parser would mis-read. When the caller
declares a reasoning model (``ExtractionConfig.reasoning_model``):

  - ``reasoning_suppression_kwargs`` turns thinking off on each call;
    ``is_unsupported_reasoning_param_error`` recognises an endpoint that rejects
    the parameter so the caller can drop it rather than fail.
  - ``strip_reasoning`` removes any inline ``<think>…</think>`` block that still
    arrives, before the line-based parser sees it.
"""

from __future__ import annotations

import re
from typing import Any

# base_url fragments identifying a self-hosted server (Ollama / vLLM). These take
# ``enable_thinking`` through the chat template and reject the hosted
# ``reasoning_effort`` parameter.
_LOCAL_ENDPOINT_MARKERS: tuple[str, ...] = ("11434", "ollama", ":8000", "vllm")

# Substrings that mark a 400 as "this thinking-off parameter is not supported for
# this model" (e.g. Groq: "`reasoning_effort` is not supported with this model").
# Matched only on a 400 so unrelated bad-request errors are not swallowed.
_UNSUPPORTED_PARAM_MARKERS: tuple[str, ...] = (
    "reasoning_effort",
    "enable_thinking",
    "not supported",
)

# A non-greedy, dot-matches-newline match of a complete think block, plus any
# trailing whitespace. An unclosed ``<think>`` (truncated output) is intentionally
# not matched — there is no answer to recover, so the content is left as-is.
_THINK_BLOCK: re.Pattern[str] = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)


def reasoning_suppression_kwargs(base_url: str | None) -> dict[str, Any]:
    """Return request kwargs that turn thinking off, chosen by endpoint.

    A hosted gateway takes ``reasoning_effort="none"``; a self-hosted server takes
    ``enable_thinking=false`` through the chat template. An endpoint that rejects
    the parameter raises the error :func:`is_unsupported_reasoning_param_error`
    recognises, so the caller can drop it and retry.

    Args:
        base_url: The endpoint URL, or ``None`` for the provider default.

    Returns:
        ``{"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}``
        for a self-hosted server, otherwise ``{"reasoning_effort": "none"}``.

    Example:
        >>> reasoning_suppression_kwargs("https://api.groq.com/openai/v1")
        {'reasoning_effort': 'none'}
        >>> reasoning_suppression_kwargs("http://localhost:11434/v1")
        {'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}}
    """
    if base_url is not None and any(
        marker in base_url.lower() for marker in _LOCAL_ENDPOINT_MARKERS
    ):
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    return {"reasoning_effort": "none"}


def is_unsupported_reasoning_param_error(exc: Exception) -> bool:
    """Whether *exc* is a 400 rejecting the thinking-off parameter.

    True only for a 400 whose message names the parameter or says "not supported",
    so unrelated bad requests (context length, malformed JSON) are left to
    propagate.

    Args:
        exc: The exception raised by the chat-completions call.

    Returns:
        ``True`` if the parameter should be dropped and the call retried.
    """
    if getattr(exc, "status_code", None) != 400:
        return False
    message = str(exc).lower()
    return any(marker in message for marker in _UNSUPPORTED_PARAM_MARKERS)


def strip_reasoning(text: str) -> str:
    """Remove ``<think>…</think>`` reasoning blocks from model output.

    A true no-op when the text contains no closed think block, so non-reasoning
    output is returned unchanged.

    Args:
        text: Raw model output that may contain reasoning blocks.

    Returns:
        The text with any reasoning blocks removed (and surrounding whitespace
        trimmed), or the original text unchanged when none are present.

    Example:
        >>> strip_reasoning("<think>the name is Alice</think>\\nname = Alice")
        'name = Alice'
        >>> strip_reasoning("name = Alice")
        'name = Alice'
    """
    cleaned = _THINK_BLOCK.sub("", text)
    return cleaned.strip() if cleaned != text else text
