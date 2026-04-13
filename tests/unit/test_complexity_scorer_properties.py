"""
Property-based tests for ComplexityScorer using Hypothesis.

These tests verify INVARIANTS — properties that must hold for ALL inputs,
not just the specific examples in test_complexity_scorer.py.

Run fast subset:  pytest tests/unit/ -k "properties" --hypothesis-seed=0
Run full search:  pytest tests/unit/ -k "properties" -p no:randomly
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from formatshield.scorer.complexity_scorer import ComplexityScorer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_scorer = ComplexityScorer()


# ---------------------------------------------------------------------------
# Schema strategies
# ---------------------------------------------------------------------------

# Simple flat schemas
_flat_schema = st.fixed_dictionaries(
    {"type": st.sampled_from(["string", "number", "integer", "boolean", "null"])}
)

# Object schemas with 0–5 string properties
_object_schema = st.fixed_dictionaries(
    {
        "type": st.just("object"),
        "properties": st.dictionaries(
            st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L",))),
            _flat_schema,
            max_size=5,
        ),
    }
)

# Any JSON-schema-like dict (flat, object, or empty)
_any_schema: st.SearchStrategy[dict[str, Any]] = st.one_of(
    _flat_schema,
    _object_schema,
    st.just({}),
)

# Prompts: printable ASCII, 1–500 chars
_prompt = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=500,
)


# ---------------------------------------------------------------------------
# Invariant 1 — score is always in [0.0, 1.0]
# ---------------------------------------------------------------------------


@given(prompt=_prompt, schema=_any_schema)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_score_always_in_unit_interval(prompt: str, schema: dict[str, Any]) -> None:
    """ComplexityScorer.compute_score() must return a float in [0.0, 1.0] for any input."""
    features = _scorer.score(prompt=prompt, schema=schema)
    score = _scorer.compute_score(features)
    assert isinstance(score, float), f"score must be float, got {type(score)}"
    assert 0.0 <= score <= 1.0, f"score={score} out of [0, 1] for prompt={prompt!r}"


# ---------------------------------------------------------------------------
# Invariant 2 — statelessness: two identical calls return identical results
# ---------------------------------------------------------------------------


@given(prompt=_prompt, schema=_any_schema)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_score_is_deterministic(prompt: str, schema: dict[str, Any]) -> None:
    """Same (prompt, schema) always yields the same score — scorer must be stateless."""
    features_a = _scorer.score(prompt=prompt, schema=schema)
    score_a = _scorer.compute_score(features_a)

    features_b = _scorer.score(prompt=prompt, schema=schema)
    score_b = _scorer.compute_score(features_b)

    assert score_a == score_b, (
        f"Non-deterministic score: {score_a} vs {score_b} for prompt={prompt!r}"
    )


# ---------------------------------------------------------------------------
# Invariant 3 — schema_depth is always non-negative
# ---------------------------------------------------------------------------


@given(schema=_any_schema)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_schema_depth_non_negative(schema: dict[str, Any]) -> None:
    """schema_depth feature must always be >= 0."""
    features = _scorer.score(prompt="test prompt", schema=schema)
    assert features.schema_depth >= 0, f"schema_depth={features.schema_depth} is negative"


# ---------------------------------------------------------------------------
# Invariant 4 — deeply nested schema scores higher than flat schema
#               (schema depth monotonicity property)
# ---------------------------------------------------------------------------


@given(depth=st.integers(min_value=3, max_value=6))
@settings(max_examples=50)
def test_deeper_schema_scores_higher_or_equal(depth: int) -> None:
    """
    A schema nested to 'depth' levels must score >= a flat schema with the same prompt.
    This is a monotonicity invariant — more constrained schemas can't score lower.
    """
    prompt = "Extract information from this document step by step."

    flat: dict[str, Any] = {"type": "string"}

    nested: dict[str, Any] = {"type": "string"}
    for _ in range(depth):
        nested = {"type": "object", "properties": {"value": nested}, "required": ["value"]}

    flat_features = _scorer.score(prompt=prompt, schema=flat)
    nested_features = _scorer.score(prompt=prompt, schema=nested)

    flat_score = _scorer.compute_score(flat_features)
    nested_score = _scorer.compute_score(nested_features)

    assert nested_score >= flat_score, (
        f"depth={depth}: nested_score={nested_score:.4f} < flat_score={flat_score:.4f}"
    )


# ---------------------------------------------------------------------------
# Invariant 5 — empty schema is handled without raising
# ---------------------------------------------------------------------------


@given(prompt=_prompt)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_empty_schema_never_raises(prompt: str) -> None:
    """ComplexityScorer must handle empty schema dict without raising any exception."""
    try:
        features = _scorer.score(prompt=prompt, schema={})
        score = _scorer.compute_score(features)
        assert 0.0 <= score <= 1.0
    except Exception as exc:
        pytest.fail(f"Unexpected exception for empty schema: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Invariant 6 — longer prompts score >= shorter prompts (length bucket monotonicity)
# ---------------------------------------------------------------------------


@given(
    base=st.text(alphabet=st.characters(whitelist_categories=("L",)), min_size=5, max_size=50),
    extension=st.text(
        alphabet=st.characters(whitelist_categories=("L",)), min_size=100, max_size=400
    ),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_longer_prompt_has_higher_or_equal_length_bucket(base: str, extension: str) -> None:
    """Longer prompts must have prompt_length_bucket >= shorter prompts."""
    schema: dict[str, Any] = {}

    short_features = _scorer.score(prompt=base, schema=schema)
    long_features = _scorer.score(prompt=base + " " + extension, schema=schema)

    assert long_features.prompt_length_bucket >= short_features.prompt_length_bucket, (
        f"Longer prompt has lower bucket: "
        f"{long_features.prompt_length_bucket} < {short_features.prompt_length_bucket}"
    )
