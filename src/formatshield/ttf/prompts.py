"""
TTF prompt builders and thinking-text extraction.

Implements the two-prompt pattern from CRANE (arXiv 2502.09061):

* Pass 1 — build an unconstrained "think" prompt that wraps the original
  request with an instruction to reason inside ``<think>…</think>`` tags
  before committing to a structured answer.
* Pass 2 — build a "format" prompt that appends the Pass 1 thinking text
  and instructs the model to now produce the final JSON.
* ``extract_thinking`` — parse the Pass 1 response and return only the
  content inside the ``<think>`` tags (or the whole response as a fallback).

Public API also exposes the canonical template strings so callers can
customise them without monkey-patching the module:

- :data:`THINK_PROMPT_TEMPLATE`
- :data:`FORMAT_PROMPT_TEMPLATE`
- :data:`EXTRACTION_THINK_TEMPLATE`
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Canonical template strings (part of the public API)
# ---------------------------------------------------------------------------

THINK_PROMPT_TEMPLATE: str = """\
{original_prompt}

Think through this step by step before producing the final structured output.
Use <think>...</think> tags to show your reasoning process.
Be thorough in your analysis — the quality of your reasoning directly affects output accuracy.
"""

FORMAT_PROMPT_TEMPLATE: str = """\
{think_prompt}

{thinking}

Based on your reasoning above, now produce the final structured output.
Your response must be valid JSON matching the required schema exactly.
Do not add any explanation — output JSON only.
"""

EXTRACTION_THINK_TEMPLATE: str = """\
Analyze the following text carefully:

{original_prompt}

Step 1: Identify all relevant entities and relationships.
Step 2: Note any ambiguities or edge cases.
Step 3: Determine the most accurate structured representation.

Use <think>...</think> to record your analysis.
"""

# ---------------------------------------------------------------------------
# Internal backend-compatible templates (used by build_* helpers below)
# ---------------------------------------------------------------------------

_THINK_SUFFIX = """

Think through this carefully, step by step.
Write your reasoning inside <think>...</think> tags.
Do NOT produce any JSON or structured output yet — reasoning only.
"""

_FORMAT_PREFIX = """

Now, based on your reasoning above, produce the final structured output.
Respond with valid JSON only — no explanations, no markdown fences."""

_FORMAT_WITH_SCHEMA_PREFIX = """

Now, based on your reasoning above, produce the final structured output.
The output MUST conform exactly to the following JSON schema:

{schema_json}

Respond with valid JSON only — no explanations, no markdown fences."""

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def build_think_prompt(original_prompt: str) -> str:
    """Build the Pass 1 (unconstrained reasoning) prompt.

    Appends a standard instruction suffix that asks the model to reason freely
    inside ``<think>…</think>`` tags before producing any structured output.

    Parameters
    ----------
    original_prompt:
        The user's original prompt, unchanged.

    Returns
    -------
    str
        The modified prompt to send to the backend with ``constraints=None``.

    Example::

        think_prompt = build_think_prompt("Extract obligations from this contract: ...")
        # → "Extract obligations from this contract: ...\\n\\nThink through this ..."
    """
    return THINK_PROMPT_TEMPLATE.format(original_prompt=original_prompt)


def build_format_prompt(
    think_prompt: str,
    thinking: str,
    schema: dict | None = None,  # type: ignore[type-arg]
    # Legacy alias accepted as a keyword argument for backwards compatibility
    thinking_text: str | None = None,
) -> str:
    """Build the Pass 2 (constrained formatting) prompt.

    Concatenates the Pass 1 think prompt, the model's thinking response, and a
    formatting instruction.  If a JSON schema is provided it is embedded in the
    suffix so the model has the exact target shape visible at format time.

    Parameters
    ----------
    think_prompt:
        The Pass 1 prompt (as returned by :func:`build_think_prompt`).
    thinking:
        The extracted thinking text produced by the model in Pass 1
        (content of the ``<think>`` tags, as returned by
        :func:`extract_thinking`).
    schema:
        Optional JSON Schema dict.  When provided, the schema is serialised
        into the prompt so the model can see the exact expected structure.
    thinking_text:
        Deprecated alias for *thinking*.  Provided for backwards compatibility
        with code written against the earlier API.  Ignored when *thinking*
        is non-empty.

    Returns
    -------
    str
        The Pass 2 prompt to send with ``constraints="json"``.
    """
    import json

    # Resolve legacy alias
    effective_thinking = thinking if thinking else (thinking_text or "")

    if schema:
        schema_json = json.dumps(schema, indent=2)
        format_suffix = _FORMAT_WITH_SCHEMA_PREFIX.format(schema_json=schema_json)
    else:
        format_suffix = _FORMAT_PREFIX

    # Build context block + single formatting instruction.
    # Not using FORMAT_PROMPT_TEMPLATE here — it already contains an instruction suffix
    # that would duplicate format_suffix below.
    return f"{think_prompt}\n\n{effective_thinking}" + format_suffix


def build_extraction_think_prompt(original_prompt: str) -> str:
    """Build an extraction-optimised thinking prompt.

    Uses :data:`EXTRACTION_THINK_TEMPLATE` which provides explicit numbered
    reasoning steps tailored for entity / relationship extraction tasks.

    Parameters
    ----------
    original_prompt:
        The raw extraction request.

    Returns
    -------
    str
        The extraction-optimised thinking prompt.
    """
    return EXTRACTION_THINK_TEMPLATE.format(original_prompt=original_prompt)


def extract_thinking(response_text: str) -> str:
    """Extract the content inside ``<think>…</think>`` tags from *response_text*.

    Handles several real-world variants:
    * ``<think>…</think>`` — canonical form (DeepSeek R1 / CRANE)
    * ``<thinking>…</thinking>`` — Anthropic extended thinking
    * No tags at all — the model may have reasoned inline; return the full
      response in that case.

    The extraction is case-insensitive and supports multi-line content.
    If multiple ``<think>`` blocks are present (unusual but possible), the
    content of all of them is concatenated with a newline separator.

    Parameters
    ----------
    response_text:
        The raw string returned by the model for the Pass 1 generate call.

    Returns
    -------
    str
        The extracted thinking text.  Returns the original *response_text*
        (stripped) when no recognised tag pattern is found — the whole Pass 1
        output is treated as reasoning in that case.

    Example::

        raw = "<think>Let me calculate: 3 * 4 = 12</think>"
        text = extract_thinking(raw)
        assert text == "Let me calculate: 3 * 4 = 12"
    """
    # Try <think>...</think> first (canonical, DeepSeek R1 / CRANE style)
    think_pattern = re.compile(
        r"<think>(.*?)</think>",
        re.DOTALL | re.IGNORECASE,
    )
    matches = think_pattern.findall(response_text)
    if matches:
        return "\n".join(m.strip() for m in matches)

    # Try <thinking>...</thinking> (Anthropic extended thinking style)
    thinking_pattern = re.compile(
        r"<thinking>(.*?)</thinking>",
        re.DOTALL | re.IGNORECASE,
    )
    matches = thinking_pattern.findall(response_text)
    if matches:
        return "\n".join(m.strip() for m in matches)

    # No tags found — treat the entire response as reasoning text
    return response_text.strip()
