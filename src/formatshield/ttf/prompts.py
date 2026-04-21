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

Schema-aware extensions:

* ``build_schema_phi_think_prompt`` — schema-aware Pass 1 prompt that injects
  field dependency order, enum constraints, Φ routing depth, and a
  vocabulary bridge (ΔK hints) so the model reasons about the *actual* schema
  instead of a generic task.

Public API also exposes the canonical template strings so callers can
customise them without monkey-patching the module:

- :data:`THINK_PROMPT_TEMPLATE`
- :data:`FORMAT_PROMPT_TEMPLATE`
- :data:`EXTRACTION_THINK_TEMPLATE`
"""

from __future__ import annotations

import re
import string
from typing import Any

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
# Schema-Φ Think Prompt helpers
# ---------------------------------------------------------------------------

_DK_VOCAB_BRIDGE_THRESHOLD: float = 0.50
"""ΔK threshold above which vocabulary bridge hints are injected into the prompt."""

_PHI_DEPTH_LABELS: list[tuple[float, str]] = [
    (0.95, "MAXIMUM — schema is highly complex and semantically distant; reason exhaustively"),
    (0.80, "DEEP — schema has significant nesting or constraint coupling; reason carefully"),
    (0.65, "STANDARD — moderate schema complexity; reason field-by-field"),
    (0.0, "LIGHT — schema is simple; brief reasoning is sufficient"),
]


def _phi_depth_label(phi: float) -> str:
    """Return a human-readable reasoning depth label for a given Φ score."""
    for threshold, label in _PHI_DEPTH_LABELS:
        if phi >= threshold:
            return label
    return _PHI_DEPTH_LABELS[-1][1]


def _collect_schema_field_info(
    schema: dict[str, Any],
    prefix: str = "",
    _depth: int = 0,
) -> list[dict[str, Any]]:
    """Walk *schema* and collect per-field metadata in dependency order.

    Returns a list of dicts, each with keys:
    ``path``, ``type``, ``required``, ``enum``, ``description``.

    Parent fields appear before their children (dependency order).
    """
    if _depth > 12 or not isinstance(schema, dict):
        return []

    results: list[dict[str, Any]] = []
    props: dict[str, Any] = schema.get("properties", {})
    parent_required: set[str] = set(schema.get("required", []))

    for name, sub in props.items():
        full_path = f"{prefix}.{name}" if prefix else name
        if not isinstance(sub, dict):
            results.append(
                {
                    "path": full_path,
                    "type": "any",
                    "required": name in parent_required,
                    "enum": None,
                    "description": "",
                }
            )
            continue

        ftype = sub.get("type", "object" if sub.get("properties") else "any")
        enum_vals = sub.get("enum")
        description = sub.get("description", "")
        is_required = name in parent_required

        results.append(
            {
                "path": full_path,
                "type": ftype,
                "required": is_required,
                "enum": enum_vals,
                "description": description,
            }
        )

        # Recurse into nested objects
        if sub.get("properties"):
            results.extend(_collect_schema_field_info(sub, full_path, _depth=_depth + 1))

        # Recurse into array item schemas
        items = sub.get("items", {})
        if isinstance(items, dict) and items.get("properties"):
            results.extend(_collect_schema_field_info(items, full_path + "[]", _depth=_depth + 1))

    # Handle allOf / anyOf / oneOf sub-schemas
    for kw in ("allOf", "anyOf", "oneOf"):
        for clause in schema.get(kw, []):
            if isinstance(clause, dict):
                results.extend(_collect_schema_field_info(clause, prefix, _depth=_depth + 1))

    return results


def _vocabulary_bridge_hints(
    prompt: str,
    schema_fields: list[dict[str, Any]],
    max_hints: int = 5,
) -> list[str]:
    """Return vocabulary bridge hints for schema fields absent from the prompt.

    Finds schema field names (leaf names) that do not appear in the prompt
    text — these are the vocabulary mismatches the model must bridge.

    Parameters
    ----------
    prompt:
        The original user prompt.
    schema_fields:
        Output of :func:`_collect_schema_field_info`.
    max_hints:
        Maximum number of hints to return.

    Returns
    -------
    list[str]
        Lines of the form ``"schema field 'X' — not mentioned in prompt"``
        for the top-K mismatched leaf fields.
    """
    # Normalize prompt to lowercase words, strip punctuation
    translator = str.maketrans("", "", string.punctuation)
    prompt_words: set[str] = set(prompt.lower().translate(translator).split())

    hints: list[str] = []
    seen_paths: set[str] = set()

    for field in schema_fields:
        path: str = field["path"]
        # Use the leaf name (last segment, strip array markers)
        leaf = path.split(".")[-1].replace("[]", "").strip()
        if not leaf or leaf in seen_paths:
            continue
        seen_paths.add(leaf)

        # Decompose snake_case / camelCase into tokens
        tokens = re.sub(r"([a-z])([A-Z])", r"\1 \2", leaf).lower()
        tokens = tokens.replace("_", " ").replace("-", " ")
        leaf_words = set(tokens.split())

        if not leaf_words.intersection(prompt_words):
            ftype = field.get("type", "any")
            hint = f"  schema field '{leaf}' ({ftype}) — not found in prompt vocabulary"
            if field.get("enum"):
                allowed = ", ".join(repr(v) for v in field["enum"][:4])
                hint += f"; allowed values: [{allowed}]"
            hints.append(hint)

        if len(hints) >= max_hints:
            break

    return hints


#: Number of context words captured before/after each numeric anchor.
_ANCHOR_CONTEXT_WINDOW: int = 4

#: Maximum numeric anchors to inject (avoids over-cluttering the prompt).
_MAX_NUMERIC_ANCHORS: int = 6

#: ΔK threshold above which numeric anchor injection is enabled.
#: Below this value the prompt vocabulary largely matches the schema — no anchoring needed.
_ANCHOR_DK_THRESHOLD: float = 0.30


def _extract_numeric_anchors(prompt: str, max_anchors: int = _MAX_NUMERIC_ANCHORS) -> list[str]:
    """Extract key numeric values from *prompt* with surrounding context.

    Finds all decimal numbers (integers and floats) in the prompt, captures
    a window of context words around each one, and returns anchor lines to
    inject into the TTF Pass 1 reasoning prompt.  These lines tell the model
    to ground its calculations in the actual values from the prompt rather
    than inventing plausible-sounding numbers.

    Parameters
    ----------
    prompt:
        The original user prompt.
    max_anchors:
        Maximum number of anchor lines to return.

    Returns
    -------
    list[str]
        Lines of the form ``"  <value> — context: '<surrounding words>'"``
        ready to embed in the prompt context block.
    """
    # Tokenise to words for context window extraction
    words = re.split(r"\s+", prompt.strip())

    # Find all numeric tokens (integers and floats, skip pure year-like 4-digit ints)
    numeric_re = re.compile(r"^-?\d+(?:\.\d+)?%?$")
    anchors: list[str] = []
    seen_values: set[str] = set()

    for i, word in enumerate(words):
        # Strip common surrounding punctuation before matching
        clean = word.strip(".,;:()[]{}\"'")
        if not numeric_re.match(clean):
            continue

        # Skip very large integers that are likely counts/years, not domain values
        try:
            as_float = float(clean.rstrip("%"))
        except ValueError:
            continue
        if as_float > 1_000_000 or (as_float == int(as_float) and as_float > 9_999):
            continue

        # Deduplicate by value
        norm_key = f"{as_float:.4g}"
        if norm_key in seen_values:
            continue
        seen_values.add(norm_key)

        # Capture context window
        start = max(0, i - _ANCHOR_CONTEXT_WINDOW)
        end = min(len(words), i + _ANCHOR_CONTEXT_WINDOW + 1)
        ctx_words = [w for w in words[start:end] if w != word]
        ctx = " ".join(ctx_words).strip()
        ctx_snippet = ctx[:60] if ctx else "—"

        anchors.append(f"  {clean} — context: '{ctx_snippet}'")

        if len(anchors) >= max_anchors:
            break

    return anchors


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def build_schema_phi_think_prompt(
    original_prompt: str,
    schema: dict[str, Any],
    phi: float,
    tau: float,
    delta_k: float,
    lambda2: float = 0.0,
) -> str:
    """Build a schema-aware Pass 1 reasoning prompt.

    Unlike :func:`build_think_prompt`, this function injects the schema's
    field structure, constraint hints, and Φ routing signal directly into
    the prompt so the model reasons *about the actual schema* rather than
    performing generic task reasoning.

    When ΔK exceeds :data:`_DK_VOCAB_BRIDGE_THRESHOLD`, vocabulary bridge
    hints are appended — listing schema field names that do not appear in
    the prompt, to prevent the model from inventing field names or missing
    required mappings.

    Parameters
    ----------
    original_prompt:
        The user's original prompt, unchanged.
    schema:
        JSON Schema dict describing the expected output.
    phi:
        Composite routing score Φ ∈ [0, 1].
    tau:
        Schema constraint tightness τ ∈ [0, 1].
    delta_k:
        NCD alignment gap ΔK ∈ [0, 1].
    lambda2:
        Normalized Fiedler value λ̃₂ ∈ [0, 1].

    Returns
    -------
    str
        The schema-aware think prompt to send as Pass 1 with
        ``constraints=None``.
    """
    fields = _collect_schema_field_info(schema)

    # ── Routing context block ──────────────────────────────────────────────
    depth_label = _phi_depth_label(phi)
    context_lines = [
        "── Schema-Guided Reasoning Context ──────────────────────────────────",
        f"Routing complexity  Φ={phi:.3f}  λ̃₂={lambda2:.3f}  τ={tau:.3f}  ΔK={delta_k:.3f}",
        f"Reasoning depth required: {depth_label}",
    ]

    # ── Field dependency order block ───────────────────────────────────────
    if fields:
        context_lines.append("")
        context_lines.append(
            "Schema fields — reason about them in this order (parents before children):"
        )
        for i, f in enumerate(fields, 1):
            req_marker = "required" if f["required"] else "optional"
            type_str = f["type"]
            desc_suffix = f"  # {f['description']}" if f.get("description") else ""
            context_lines.append(f"  {i:2d}. {f['path']}  ({type_str}, {req_marker}){desc_suffix}")

    # ── Constrained fields block (enum / boolean / const) ─────────────────
    constrained = [f for f in fields if f.get("enum") or f.get("type") == "boolean"]
    if constrained:
        context_lines.append("")
        context_lines.append(
            "Constrained fields — output MUST use one of the allowed values exactly:"
        )
        for f in constrained:
            if f.get("enum"):
                allowed = ", ".join(repr(v) for v in f["enum"])
                context_lines.append(f"  {f['path']}: one of [{allowed}]")
            elif f.get("type") == "boolean":
                context_lines.append(f"  {f['path']}: true or false")

    # ── Cross-field constraints block ──────────────────────────────────────
    schema_constraints: dict[str, Any] = schema.get("constraints", {})
    if schema_constraints:
        context_lines.append("")
        context_lines.append(
            "Cross-field constraints — your reasoning MUST satisfy ALL of these:"
        )
        for name, rule in schema_constraints.items():
            context_lines.append(f"  [{name}] {rule}")

    # ── Numeric anchors block ─────────────────────────────────────────────────
    # Always inject when ΔK exceeds the anchor threshold — high vocabulary mismatch
    # means the model is likely to invent plausible numbers rather than use the ones
    # already supplied in the prompt.  By showing the exact values and their context,
    # we ground the model's calculations in the prompt data.
    if delta_k > _ANCHOR_DK_THRESHOLD:
        anchors = _extract_numeric_anchors(original_prompt)
        if anchors:
            context_lines.append("")
            context_lines.append(
                "Numeric anchors — ground your calculations in these values from the prompt"
                " (do NOT invent numbers that contradict these):"
            )
            context_lines.extend(anchors)

    # ── Vocabulary bridge hints ─────────────────────────────────────────────
    if delta_k > _DK_VOCAB_BRIDGE_THRESHOLD and fields:
        hints = _vocabulary_bridge_hints(original_prompt, fields)
        if hints:
            context_lines.append("")
            context_lines.append(
                f"Vocabulary bridge (ΔK={delta_k:.3f} — prompt/schema vocabulary gap detected):"
            )
            context_lines.append("  The following schema fields were NOT found in the prompt.")
            context_lines.append(
                "  You must still populate them using context clues from the prompt:"
            )
            context_lines.extend(hints)

    context_block = "\n".join(context_lines)

    # ── Reasoning instructions ─────────────────────────────────────────────
    reasoning_steps = [
        "Think through this step by step:",
        "  1. For each required field (in the order listed above), determine what value"
        " the prompt implies.",
        "  2. For constrained fields, select only from the listed allowed values — no variations.",
        "  3. Resolve any vocabulary bridge gaps: if a schema field is not directly"
        " mentioned in the prompt, infer it from context.",
        "  4. For nested / parent fields, reason about the parent's shape before its children.",
    ]
    if schema_constraints:
        reasoning_steps.append(
            "  5. Verify your computed values satisfy ALL cross-field constraints listed above"
            " before finalising — adjust if any constraint is violated."
        )
    reasoning_steps += [
        "",
        "Anti-fabrication rule (NON-NEGOTIABLE):",
        "  Do NOT invent, guess, or hallucinate values that are absent from the prompt.",
        "  If a required field's value is not present or cannot be reliably inferred:",
        "    - strings → use \"\" or \"unknown\"",
        "    - numbers → use 0 only if semantically correct; otherwise flag in your reasoning",
        "    - enums → select only if the choice is clearly inferable from the prompt",
        "    - arrays → use [] if no items are mentioned",
        "  Fabricating schema-valid but semantically wrong data is worse than returning empty.",
        "",
        "Use <think>...</think> tags to record your reasoning.",
        "Do NOT produce any JSON or structured output yet — reasoning only.",
    ]
    reasoning_block = "\n".join(reasoning_steps)

    return f"{original_prompt}\n\n{context_block}\n\n{reasoning_block}"


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


# ---------------------------------------------------------------------------
# Static cache prefix builder for Pass 2
# ---------------------------------------------------------------------------


def build_cache_prefix_for_format_prompt(
    schema: dict[str, Any] | None,
    constraints: str | None = "json",
) -> str:
    """Build the **static cacheable prefix** for a Pass 2 format prompt.

    Restructures Pass 2 so the part that is *identical across all requests
    sharing the same schema* comes first.  Backends that support KV-cache
    prefix reuse can amortise the encoding cost across requests.

    The prefix contains:
    - A system-level formatting instruction
    - The full JSON schema serialized for the model
    - The output contract (JSON only, no surrounding text)

    The variable part (Pass 1 thinking trace + user prompt) is appended by
    :func:`build_format_prompt` when building the final Pass 2 prompt.

    Parameters
    ----------
    schema:
        JSON Schema dict.  When ``None``, a generic JSON instruction is
        returned instead.
    constraints:
        ``"json"`` (default) requests JSON-only output.

    Returns
    -------
    str
        The static cacheable prefix text.  Guaranteed to be ≥
        :data:`_CACHE_PREFIX_MIN_CHARS` characters when a schema is supplied
        (by padding with the full schema JSON).
    """
    import json as _json

    if constraints == "json" and schema:
        schema_json = _json.dumps(schema, indent=2)
        prefix = (
            "You are a structured-output assistant.  Your task is to produce "
            "a valid JSON object from the user's input and the reasoning trace "
            "that follows.\n\n"
            "OUTPUT CONTRACT\n"
            "───────────────\n"
            "• Respond with a single JSON object — no markdown fences, no "
            "explanation, no surrounding text.\n"
            "• The JSON object MUST conform exactly to the schema below.  Every "
            "required field must be present.  All values must match their "
            "declared types and constraints.\n"
            "• If a required field cannot be determined from the reasoning, "
            "use the most reasonable default (empty string for strings, 0 for "
            "numbers, false for booleans).\n\n"
            "TARGET JSON SCHEMA\n"
            "──────────────────\n"
            f"{schema_json}\n\n"
            "REASONING TRACE (for context — do not reproduce in output)\n"
            "────────────────────────────────────────────────────────────"
        )
    elif constraints == "json":
        prefix = (
            "You are a structured-output assistant.  Your task is to produce "
            "a valid JSON object from the user's input and the reasoning trace "
            "that follows.\n\n"
            "OUTPUT CONTRACT\n"
            "───────────────\n"
            "• Respond with a single JSON object — no markdown fences, no "
            "explanation, no surrounding text.\n\n"
            "REASONING TRACE (for context — do not reproduce in output)\n"
            "────────────────────────────────────────────────────────────"
        )
    else:
        prefix = (
            "Based on the reasoning trace below, produce the final structured "
            "output as requested.\n\n"
            "REASONING TRACE\n"
            "───────────────"
        )

    return prefix


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
    * ``<thinking>…</thinking>`` — extended thinking variant
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

    # Try <thinking>...</thinking> (extended thinking variant)
    thinking_pattern = re.compile(
        r"<thinking>(.*?)</thinking>",
        re.DOTALL | re.IGNORECASE,
    )
    matches = thinking_pattern.findall(response_text)
    if matches:
        return "\n".join(m.strip() for m in matches)

    # No tags found — treat the entire response as reasoning text
    return response_text.strip()
