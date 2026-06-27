"""Classify a failed call into an honest category, for fair reporting.

A raw SDK exception (e.g. a multi-KB ``InstructorRetryException`` blob) tells a
reader nothing and makes a genuine single-call limit look like a crash. Each
failure is mapped to a stable :class:`FailureKind` plus a one-line reason, and to
whether it is a *transport* failure (infra: the call never returned through no
fault of the method - credited to call-failed) or a *capability* failure (the
single call itself could not produce the output - a real miss in the denominator).

Classification works on the error *text* so it serves both live runs (the
exception) and re-scoring already-written results (the stored error string).
"""

from __future__ import annotations

from enum import Enum

__all__ = ["FailureKind", "classify", "classify_exc", "is_transport"]


class FailureKind(Enum):
    """A failure's honest category. Value is the stable label used in results."""

    # Capability failures: the single call could not produce the output.
    SINGLE_CALL_OUTPUT_CEILING = "single_call_output_ceiling"
    REQUEST_EXCEEDS_CONTEXT = "request_exceeds_context"
    OUTPUT_TRUNCATED = "output_truncated"
    JSON_TRUNCATED = "json_truncated"
    SCHEMA_VALIDATION_FAILED = "schema_validation_failed"
    # Transport failures: infra, not the method's fault.
    RATE_LIMITED = "rate_limited"
    TRANSPORT = "transport"
    OTHER = "other"

    @property
    def message(self) -> str:
        """A one-line, reader-facing explanation of this category."""
        return _MESSAGES[self]


_MESSAGES: dict[FailureKind, str] = {
    FailureKind.SINGLE_CALL_OUTPUT_CEILING: (
        "single-call output ceiling: the server aborts a completion that runs past "
        "~120s (≈16k output tokens), so one call cannot emit this many fields"
    ),
    FailureKind.REQUEST_EXCEEDS_CONTEXT: (
        "request exceeds the context window (messages + max_tokens too large for one call)"
    ),
    FailureKind.OUTPUT_TRUNCATED: (
        "output truncated at the token limit; the method returned no usable object"
    ),
    FailureKind.JSON_TRUNCATED: "model output was not valid JSON (truncated or malformed)",
    FailureKind.SCHEMA_VALIDATION_FAILED: "output rejected by schema validation",
    FailureKind.RATE_LIMITED: "rate limited (429) after exhausting retries",
    FailureKind.TRANSPORT: "transport error (timeout / connection)",
    FailureKind.OTHER: "uncategorized failure",
}

# Transport kinds are infra (the method is not at fault) → credited to call-failed.
_TRANSPORT_KINDS = frozenset({FailureKind.RATE_LIMITED, FailureKind.TRANSPORT})

# Ordered (specific → generic): the first matching probe wins. Each probe is a set
# of case-insensitive substrings; any one present classifies the text.
_PROBES: tuple[tuple[FailureKind, tuple[str, ...]], ...] = (
    (
        FailureKind.REQUEST_EXCEEDS_CONTEXT,
        ("reduce the length", "maximum context", "context_length"),
    ),
    (FailureKind.SCHEMA_VALIDATION_FAILED, ("validation error", "input should be", "model_type")),
    (FailureKind.OUTPUT_TRUNCATED, ("max_tokens length", "incomplete due to", "max_tokens limit")),
    (
        FailureKind.SINGLE_CALL_OUTPUT_CEILING,
        ("service_unavailable", "internalservererror", " 502", "503"),
    ),
    (FailureKind.RATE_LIMITED, ("rate limit", "ratelimiterror", "429", "too many requests")),
    (
        FailureKind.JSON_TRUNCATED,
        ("jsondecodeerror", "expecting", "no json object", "not valid json"),
    ),
    (
        FailureKind.TRANSPORT,
        ("apitimeout", "timed out", "timeout", "apiconnection", "connection error"),
    ),
)


def classify(text: str) -> tuple[FailureKind, str]:
    """Classify an error string into a :class:`FailureKind` and reader-facing message.

    Args:
        text: The raw error string (a stored ``error`` field or ``str(exc)``).

    Returns:
        ``(kind, message)``. Unmatched text yields ``FailureKind.OTHER`` and a
        whitespace-collapsed snippet of the original (so nothing is silently lost).
    """
    haystack = text.lower()
    for kind, needles in _PROBES:
        if any(needle in haystack for needle in needles):
            return kind, kind.message
    snippet = " ".join(text.split())
    return FailureKind.OTHER, snippet[:160] if snippet else FailureKind.OTHER.message


def classify_exc(exc: Exception) -> tuple[FailureKind, str]:
    """Classify a live exception (its type name + message), as :func:`classify`."""
    return classify(f"{type(exc).__name__}: {exc}")


def is_transport(kind: FailureKind) -> bool:
    """Return ``True`` if ``kind`` is an infra/transport failure (credited to call-failed)."""
    return kind in _TRANSPORT_KINDS
