"""Error-classifier tests — every real failure string maps to an honest category."""

from __future__ import annotations

import pytest

from benchmark.adapters._errors import FailureKind, classify, classify_exc, is_transport

# Representative error strings taken verbatim from real benchmark raw outputs.
_CASES = [
    (
        "InternalServerError: Error code: 502 - {'error': {'code': 'service_unavailable'}}",
        FailureKind.SINGLE_CALL_OUTPUT_CEILING,
    ),
    (
        "InstructorRetryException: ... Error code: 400 - {'message': "
        "'Please reduce the length of the messages or completion.'}",
        FailureKind.REQUEST_EXCEEDS_CONTEXT,
    ),
    (
        "InstructorRetryException: 2 validation errors for Extraction "
        "economy.gini_index Input should be an object",
        FailureKind.SCHEMA_VALIDATION_FAILED,
    ),
    (
        "InstructorRetryException: The output is incomplete due to a max_tokens length limit.",
        FailureKind.OUTPUT_TRUNCATED,
    ),
    ("JSONDecodeError: Expecting ',' delimiter: line 402 column 4", FailureKind.JSON_TRUNCATED),
    ("ValueError: no JSON object in model response", FailureKind.JSON_TRUNCATED),
    (
        "RateLimitError: Error code: 429 - rate limit reached for ... tokens per minute",
        FailureKind.RATE_LIMITED,
    ),
    ("APITimeoutError: Request timed out.", FailureKind.TRANSPORT),
    ("APIConnectionError: Connection error.", FailureKind.TRANSPORT),
    ("RuntimeError: something nobody anticipated", FailureKind.OTHER),
]


@pytest.mark.parametrize(("text", "expected"), _CASES)
def test_classify_maps_real_errors(text: str, expected: FailureKind):
    kind, message = classify(text)
    assert kind is expected
    assert message  # always a non-empty, reader-facing reason


def test_transport_split_is_correct():
    # Only infra failures are transport (credited to call-failed); capability
    # failures are real method misses.
    assert is_transport(FailureKind.RATE_LIMITED)
    assert is_transport(FailureKind.TRANSPORT)
    for capability in (
        FailureKind.SINGLE_CALL_OUTPUT_CEILING,
        FailureKind.REQUEST_EXCEEDS_CONTEXT,
        FailureKind.OUTPUT_TRUNCATED,
        FailureKind.JSON_TRUNCATED,
        FailureKind.SCHEMA_VALIDATION_FAILED,
    ):
        assert not is_transport(capability)


def test_classify_exc_uses_type_and_message():
    kind, _ = classify_exc(ValueError("no JSON object in model response"))
    assert kind is FailureKind.JSON_TRUNCATED


def test_other_keeps_a_snippet_of_the_original():
    kind, message = classify("RuntimeError: a very specific unexpected thing happened")
    assert kind is FailureKind.OTHER
    assert "very specific unexpected thing" in message


def test_specific_probes_win_over_generic():
    # A 400 "reduce the length" wrapped in an instructor blob must classify as
    # REQUEST_EXCEEDS_CONTEXT, not get swallowed by a later generic probe.
    kind, _ = classify(
        "InstructorRetryException: Error code: 400 - Please reduce the length of the messages"
    )
    assert kind is FailureKind.REQUEST_EXCEEDS_CONTEXT
