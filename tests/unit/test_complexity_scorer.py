"""
Unit tests for formatshield.scorer.complexity_scorer.ComplexityScorer.

Each test verifies a specific, observable behaviour of the scorer — not just
that it runs without raising an exception.  Assertions check concrete values
or well-defined ranges derived from the implementation's documented contract.
"""

from __future__ import annotations

import pytest

from formatshield.scorer.complexity_scorer import ComplexityScorer
from formatshield.scorer.features import ComplexityFeatures

# ---------------------------------------------------------------------------
# Fixtures local to this module
# ---------------------------------------------------------------------------


@pytest.fixture
def scorer() -> ComplexityScorer:
    """Fresh ComplexityScorer using the default cl100k_base encoding."""
    return ComplexityScorer()


# ---------------------------------------------------------------------------
# Tests: return type and structure
# ---------------------------------------------------------------------------


def test_score_returns_complexity_features(scorer: ComplexityScorer) -> None:
    """score() must return a ComplexityFeatures dataclass with all six fields."""
    result = scorer.score("What is the capital of France?")
    assert isinstance(result, ComplexityFeatures)
    # Verify all six fields are present and have the expected Python types
    assert isinstance(result.token_entropy, float)
    assert isinstance(result.schema_depth, int)
    assert isinstance(result.required_reasoning_ops, int)
    assert isinstance(result.instruction_tune_score, float)
    assert isinstance(result.prompt_length_bucket, int)
    assert isinstance(result.schema_constraint_count, int)


def test_compute_score_range(scorer: ComplexityScorer) -> None:
    """compute_score() must return a float strictly in [0.0, 1.0]."""
    features = scorer.score("Explain quantum entanglement step by step.")
    scalar = scorer.compute_score(features)
    assert isinstance(scalar, float)
    assert 0.0 <= scalar <= 1.0


# ---------------------------------------------------------------------------
# Tests: relative ordering (complex > simple)
# ---------------------------------------------------------------------------


def test_complex_prompt_scores_higher(scorer: ComplexityScorer) -> None:
    """
    A multi-step reasoning prompt must yield a higher composite score than a
    trivial single-word extraction prompt.
    """
    simple_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
    }
    nested_schema = {
        "type": "object",
        "properties": {
            "result": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "object",
                        "properties": {"answer": {"type": "number"}},
                        "required": ["answer"],
                    }
                },
                "required": ["value"],
            }
        },
        "required": ["result"],
    }

    simple_features = scorer.score(
        "Extract the name from this text: John.",
        schema=simple_schema,
        model_id="gpt-3.5-turbo",
    )
    complex_features = scorer.score(
        (
            "Analyze and calculate the compound interest step by step. "
            "Because the rate varies, reason through each period and derive "
            "the final amount. Prove your answer by comparing intermediate "
            "values. Solve for a principal of $10,000 at 5% compounded "
            "monthly for 10 years. Calculate each month, evaluate the total, "
            "and explain why the result differs from simple interest."
        ),
        schema=nested_schema,
        model_id="gpt-4",
    )

    simple_score = scorer.compute_score(simple_features)
    complex_score = scorer.compute_score(complex_features)
    assert complex_score > simple_score, (
        f"Expected complex_score ({complex_score:.4f}) > simple_score ({simple_score:.4f})"
    )


# ---------------------------------------------------------------------------
# Tests: token entropy
# ---------------------------------------------------------------------------


def test_token_entropy_increases_with_diversity(scorer: ComplexityScorer) -> None:
    """
    A prompt with highly diverse vocabulary must produce higher token_entropy
    than a prompt consisting of a single repeated character.
    """
    repetitive = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    diverse = (
        "quantum entanglement thermodynamics photosynthesis mitochondria "
        "cryptocurrency blockchain differential calculus trigonometry "
        "etymology linguistics anthropology paleontology neuroscience"
    )
    rep_features = scorer.score(repetitive)
    div_features = scorer.score(diverse)
    assert div_features.token_entropy > rep_features.token_entropy, (
        f"Diverse entropy ({div_features.token_entropy:.4f}) should exceed "
        f"repetitive entropy ({rep_features.token_entropy:.4f})"
    )


# ---------------------------------------------------------------------------
# Tests: schema features
# ---------------------------------------------------------------------------


def test_schema_depth_detected(scorer: ComplexityScorer) -> None:
    """A 3-level nested schema must produce schema_depth >= 3."""
    nested_schema = {
        "type": "object",
        "properties": {
            "level1": {
                "type": "object",
                "properties": {
                    "level2": {
                        "type": "object",
                        "properties": {
                            "level3": {"type": "string"},
                        },
                    }
                },
            }
        },
    }
    features = scorer.score("Prompt text.", schema=nested_schema)
    assert features.schema_depth >= 3, (
        f"Expected schema_depth >= 3 for 3-level nesting, got {features.schema_depth}"
    )


def test_schema_constraint_count(scorer: ComplexityScorer) -> None:
    """A schema with 5 required fields must produce schema_constraint_count >= 5."""
    schema_5_required = {
        "type": "object",
        "properties": {
            "field_a": {"type": "string"},
            "field_b": {"type": "string"},
            "field_c": {"type": "integer"},
            "field_d": {"type": "number"},
            "field_e": {"type": "boolean"},
        },
        "required": ["field_a", "field_b", "field_c", "field_d", "field_e"],
    }
    features = scorer.score("Extract all fields.", schema=schema_5_required)
    assert features.schema_constraint_count >= 5, (
        f"Expected schema_constraint_count >= 5, got {features.schema_constraint_count}"
    )


# ---------------------------------------------------------------------------
# Tests: CoT keyword detection
# ---------------------------------------------------------------------------


def test_cot_keywords_detected(scorer: ComplexityScorer) -> None:
    """Prompt containing 'calculate' and 'step' must yield required_reasoning_ops >= 1."""
    prompt = "Calculate the answer step by step and explain your reasoning."
    features = scorer.score(prompt)
    assert features.required_reasoning_ops >= 1, (
        f"Expected required_reasoning_ops >= 1 for CoT prompt, "
        f"got {features.required_reasoning_ops}"
    )


# ---------------------------------------------------------------------------
# Tests: prompt length buckets
# ---------------------------------------------------------------------------


def test_short_prompt_bucket_0(scorer: ComplexityScorer) -> None:
    """A prompt under 50 tokens must land in bucket 0 (short)."""
    # "What is 2 + 2?" is well under 50 tokens
    features = scorer.score("What is 2 + 2?")
    assert features.prompt_length_bucket == 0, (
        f"Expected bucket 0 for short prompt, got {features.prompt_length_bucket}"
    )


def test_long_prompt_bucket_3(scorer: ComplexityScorer) -> None:
    """A prompt over 1000 tokens must land in bucket 3 (very long)."""
    # Generate a prompt that is definitely > 1000 tokens
    # Each word ~ 1 token; 1100 words is safe
    long_prompt = " ".join([f"word{i}" for i in range(1100)])
    features = scorer.score(long_prompt)
    assert features.prompt_length_bucket == 3, (
        f"Expected bucket 3 for very long prompt, got {features.prompt_length_bucket}"
    )


# ---------------------------------------------------------------------------
# Tests: instruction-tune score
# ---------------------------------------------------------------------------


def test_native_model_high_tune_score(scorer: ComplexityScorer) -> None:
    """model_id starting with 'o1' must yield instruction_tune_score >= 0.9."""
    features = scorer.score("Solve this problem.", model_id="o1")
    assert features.instruction_tune_score >= 0.9, (
        f"Expected instruction_tune_score >= 0.9 for 'o1', got {features.instruction_tune_score}"
    )


def test_instruction_tune_score_o3_mini(scorer: ComplexityScorer) -> None:
    """model_id 'o3-mini' must yield instruction_tune_score == 1.0."""
    features = scorer.score("Solve this problem.", model_id="o3-mini")
    assert features.instruction_tune_score == 1.0


def test_instruction_tune_score_default_for_unknown(scorer: ComplexityScorer) -> None:
    """Unknown model IDs must return the default instruction-tune score (0.4)."""
    features = scorer.score("Hello.", model_id="unknown-model-xyz-9000")
    assert features.instruction_tune_score == 0.4


# ---------------------------------------------------------------------------
# Tests: error handling / robustness
# ---------------------------------------------------------------------------


def test_malformed_schema_returns_neutral(scorer: ComplexityScorer) -> None:
    """ComplexityScorer.score() must not raise on None or non-dict schemas."""
    # None schema
    result_none = scorer.score("Some prompt.", schema=None)
    assert isinstance(result_none, ComplexityFeatures)

    # Non-dict schema value passed via keyword (score() accepts dict | None,
    # so we call _score_impl via the public wrapper which catches errors)
    # The public score() catches all exceptions and returns neutral features,
    # so we verify it returns valid features without raising.
    result_zero = scorer.score("Some prompt.", schema={})
    assert isinstance(result_zero, ComplexityFeatures)


def test_empty_prompt_returns_neutral(scorer: ComplexityScorer) -> None:
    """score() must handle an empty string prompt gracefully."""
    result = scorer.score("")
    assert isinstance(result, ComplexityFeatures)
    # Empty prompt: 0 tokens → bucket 0, 0 entropy, 0 reasoning ops
    assert result.prompt_length_bucket == 0
    assert result.token_entropy == 0.0
    assert result.required_reasoning_ops == 0


def test_compute_score_on_neutral_features(scorer: ComplexityScorer) -> None:
    """compute_score() must return a float in [0,1] for neutral feature values."""
    neutral = ComplexityFeatures(
        token_entropy=0.5,
        schema_depth=1,
        required_reasoning_ops=0,
        instruction_tune_score=0.5,
        prompt_length_bucket=1,
        schema_constraint_count=1,
    )
    score = scorer.compute_score(neutral)
    assert 0.0 <= score <= 1.0


def test_feature_vector_has_six_elements(scorer: ComplexityScorer) -> None:
    """to_feature_vector() must produce a list of exactly 6 floats."""
    features = scorer.score("A test prompt with some words in it.")
    vector = features.to_feature_vector()
    assert len(vector) == 6
    assert all(isinstance(v, float) for v in vector)


def test_schema_depth_flat_schema(scorer: ComplexityScorer) -> None:
    """A flat object schema (no nested properties) must have schema_depth == 1."""
    flat_schema = {
        "type": "object",
        "properties": {
            "x": {"type": "string"},
            "y": {"type": "number"},
        },
    }
    features = scorer.score("Extract x and y.", schema=flat_schema)
    assert features.schema_depth == 1
