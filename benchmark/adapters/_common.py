"""Shared helpers for the Track-A (orchestration-layer) competitor adapters.

Every Track-A baseline runs on the *same* hosted model as nfield with the *same*
single-call budget (fairness rules, design §7). The differences between them are
only in *how* they ask for structure — free text, provider JSON mode, a
Pydantic-validating loop, a framework wrapper — so the call setup, prompt, and
result accounting are factored here and each adapter stays a thin shell.

None of these baselines decompose the schema or retrieve: they send the whole
schema and as much of the document as fits in one call. That single-call ceiling
is exactly the capability the benchmark measures them against.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from ._base import AdapterOutput
from ._errors import classify_exc, is_transport

if TYPE_CHECKING:
    from groq import Groq

# Every method is run under ONE shared budget (see benchmark.budget), so the
# baselines apply it exactly as nfield does: the document is fit to the input
# window (context_window minus the output reservation) and max_tokens is set to
# the budget's output ceiling. The budget is identical across methods, so the only
# difference measured is what each method does with the same window — not who got a
# bigger one. A low single-call score is then the genuine single-call ceiling.
_CHARS_PER_TOKEN: int = 4
# Generous timeout so a slow single call (e.g. a full 32k-token native completion)
# returns a real result, not a transport timeout we caused by under-waiting.
_REQUEST_TIMEOUT_SECONDS: float = 180.0
# Transient-failure (429 / 5xx) retry budget for the raw-SDK baselines, matched to
# nfield's DEFAULT_MAX_API_RETRIES so a shared rate-limit storm penalizes no method.
# The Groq SDK honors Retry-After natively; its own default is only 2 — too few to
# outlast a TPM window, which would mis-score a 429 as a competitor failure.
MAX_TRANSIENT_RETRIES: int = 10

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

_SYSTEM_PROMPT = (
    "You extract structured data from a document. Return a single JSON object "
    "that conforms to the provided JSON Schema. Use values found in the document; "
    "use null for any field the document does not state. Do not invent values and "
    "do not add fields outside the schema."
)


def groq_client(api_key: str | None, base_url: str | None) -> Groq:
    """Construct a Groq SDK client, falling back to the SDK's env credential pickup."""
    from groq import Groq

    if base_url:
        return Groq(
            api_key=api_key,
            base_url=base_url,
            timeout=_REQUEST_TIMEOUT_SECONDS,
            max_retries=MAX_TRANSIENT_RETRIES,
        )
    return Groq(
        api_key=api_key,
        timeout=_REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_TRANSIENT_RETRIES,
    )


def model_id(model: str) -> str:
    """Strip the ``provider/`` prefix, e.g. ``groq/llama-3.3-70b`` -> ``llama-3.3-70b``."""
    return model.split("/", 1)[1] if "/" in model else model


def messages(
    document: str,
    schema: dict[str, Any],
    *,
    context_window: int,
    max_output_tokens: int,
    instructions: str = "",
) -> list[dict[str, str]]:
    """Build the shared chat messages: system contract + (instructions +) schema + document.

    The document is fit to the budget's input window (``context_window`` minus the
    ``max_output_tokens`` reservation), so prompt + completion stay within the same
    window every method shares. The caller's domain ``instructions`` lead the USER
    message — the same channel and string nfield threads to each leaf — so every
    method receives identical guidance in the channel the model actually follows.
    """
    fitted = _fit_document(document, context_window, max_output_tokens)
    user_core = (
        f"JSON Schema:\n{json.dumps(schema)}\n\n"
        f"Document:\n{fitted}\n\n"
        "Return only the JSON object."
    )
    user = f"{instructions}\n\n{user_core}" if instructions else user_core
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from a model response, tolerating code fences.

    Args:
        text: Raw model output, possibly wrapped in ```json fences or prose.

    Returns:
        The parsed object.

    Raises:
        ValueError: If no JSON object can be recovered from the text.
    """
    stripped = _FENCE.sub("", text).strip()
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError:
        match = _JSON_OBJECT.search(stripped)
        if match is None:
            raise ValueError("no JSON object in model response") from None
        loaded = json.loads(match.group(0))
    if not isinstance(loaded, dict):
        raise ValueError(f"expected a JSON object, got {type(loaded).__name__}")
    return loaded


def count_nonempty_leaves(data: Any) -> int:
    """Count leaf values that are present (not None / empty string / empty container)."""
    if isinstance(data, dict):
        return sum(count_nonempty_leaves(v) for v in data.values())
    if isinstance(data, list):
        return sum(count_nonempty_leaves(v) for v in data)
    return 0 if data is None or data == "" else 1


def schema_field_count(schema: dict[str, Any]) -> int:
    """Count leaf fields in a JSON Schema (nested objects recursed, leaves counted)."""
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return 0
    total = 0
    for node in properties.values():
        if isinstance(node, dict) and node.get("type") == "object":
            total += schema_field_count(node)
        else:
            total += 1
    return total


def success_output(data: dict[str, Any], schema: dict[str, Any], elapsed: float) -> AdapterOutput:
    """Build a successful single-call :class:`AdapterOutput` (k=1, no decomposition)."""
    return AdapterOutput(
        data=data,
        fields_total=schema_field_count(schema),
        fields_extracted=count_nonempty_leaves(data),
        k=1,
        k_min=1,
        elapsed_seconds=elapsed,
    )


def failure_output(schema: dict[str, Any], elapsed: float, exc: Exception) -> AdapterOutput:
    """Build a failed :class:`AdapterOutput`, classifying the failure honestly.

    The exception is mapped to a :class:`~benchmark.adapters._errors.FailureKind`.
    A *transport* failure (429-exhausted / connection / timeout) means infra never
    returned a body, so every field is credited to call-failed (not the method's
    fault). A *capability* failure (single-call output ceiling, request too large,
    truncated / invalid JSON, validation reject) is the method itself failing to
    produce N fields — scored as a real miss in the denominator, ``call_failed=0``.
    """
    kind, message = classify_exc(exc)
    total = schema_field_count(schema)
    return AdapterOutput(
        data={},
        fields_total=total,
        fields_extracted=0,
        k=0,
        k_min=0,
        call_failed=total if is_transport(kind) else 0,
        elapsed_seconds=elapsed,
        error=f"{kind.value}: {message}",
        error_category=kind.value,
    )


def _fit_document(document: str, context_window: int, max_output_tokens: int) -> str:
    # Reserve the output budget so prompt + completion fit the shared window; the
    # rest is the document's input budget. Same rule for every baseline.
    input_tokens = max(0, context_window - max_output_tokens)
    budget_chars = input_tokens * _CHARS_PER_TOKEN
    return document if len(document) <= budget_chars else document[:budget_chars]
