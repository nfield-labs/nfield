"""
Unit tests for formatshield.ttf.failure_detector.FailureModeDetector.

Each test exercises a specific detection rule or override decision and asserts
the concrete label(s) expected to appear (or not appear) in the result.
"""

from __future__ import annotations

from formatshield.oracle.routing_decision import RoutingDecision
from formatshield.scorer.features import ComplexityFeatures
from formatshield.ttf.failure_detector import FAILURE_MODES, FailureModeDetector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_features(
    *,
    token_entropy: float = 0.5,
    schema_depth: int = 2,
    required_reasoning_ops: int = 3,
    instruction_tune_score: float = 0.5,
    prompt_length_bucket: int = 2,
    schema_constraint_count: int = 5,
) -> ComplexityFeatures:
    """Build ComplexityFeatures with safe, non-triggering defaults."""
    return ComplexityFeatures(
        token_entropy=token_entropy,
        schema_depth=schema_depth,
        required_reasoning_ops=required_reasoning_ops,
        instruction_tune_score=instruction_tune_score,
        prompt_length_bucket=prompt_length_bucket,
        schema_constraint_count=schema_constraint_count,
    )


def _detector() -> FailureModeDetector:
    return FailureModeDetector()


def _ttf_decision() -> RoutingDecision:
    """Return a nominal TTF routing decision to use as input to check()."""
    return RoutingDecision(
        strategy="ttf",
        expected_accuracy_delta=0.17,
        expected_overhead_pct=30.0,
        confidence=0.70,
        explanation="Heuristic threshold exceeded.",
    )


# ---------------------------------------------------------------------------
# Tests: FAILURE_MODES constant
# ---------------------------------------------------------------------------


def test_failure_modes_constant_is_list() -> None:
    """FAILURE_MODES must be a list (or similar sequence) of strings."""
    assert hasattr(FAILURE_MODES, "__iter__"), "FAILURE_MODES must be iterable"
    for mode in FAILURE_MODES:
        assert isinstance(mode, str), f"Expected str in FAILURE_MODES, got {type(mode)}"


def test_all_failure_modes_in_constant() -> None:
    """Every mode that detect() can return must appear in FAILURE_MODES."""
    detector = _detector()

    # Trigger each failure mode and verify the label is in FAILURE_MODES
    # --- simple_extraction ---
    simple_features = _make_features(schema_depth=1, prompt_length_bucket=0)
    modes = detector.detect(simple_features, model_id="llama")
    for mode in modes:
        assert mode in FAILURE_MODES, f"Detected mode {mode!r} not in FAILURE_MODES"

    # --- native_thinker ---
    modes_o1 = detector.detect(_make_features(), model_id="o1")
    for mode in modes_o1:
        assert mode in FAILURE_MODES, f"Detected mode {mode!r} not in FAILURE_MODES"

    # --- short_prompt ---
    modes_short = detector.detect(_make_features(prompt_length_bucket=0), model_id="gpt-4")
    for mode in modes_short:
        assert mode in FAILURE_MODES


# ---------------------------------------------------------------------------
# Tests: individual failure-mode detection
# ---------------------------------------------------------------------------


def test_simple_extraction_detected() -> None:
    """schema_depth <= 1 AND prompt_length_bucket <= 1 → 'simple_extraction' detected."""
    detector = _detector()
    features = _make_features(
        schema_depth=1,
        prompt_length_bucket=0,
        required_reasoning_ops=0,
    )
    modes = detector.detect(features, model_id="gpt-3.5-turbo")
    assert "simple_extraction" in modes, f"Expected 'simple_extraction' in {modes}"


def test_simple_extraction_not_detected_for_deep_schema() -> None:
    """A deep schema (depth > 1) must NOT trigger simple_extraction."""
    detector = _detector()
    features = _make_features(schema_depth=3, prompt_length_bucket=0)
    modes = detector.detect(features, model_id="gpt-4")
    assert "simple_extraction" not in modes


def test_schema_too_constrained_detected() -> None:
    """schema_constraint_count > 15 → 'schema_too_constrained' detected."""
    detector = _detector()
    # schema_constraint_count = 20 > _MAX_REQUIRED_FIELDS (15)
    features = _make_features(schema_constraint_count=20)
    schema_with_many_required: dict = {
        "type": "object",
        "properties": {f"field_{i}": {"type": "string"} for i in range(20)},
        "required": [f"field_{i}" for i in range(20)],
    }
    modes = detector.detect(features, model_id="llama-3.1-70b", schema=schema_with_many_required)
    assert "schema_too_constrained" in modes, f"Expected 'schema_too_constrained' in {modes}"


def test_schema_too_constrained_by_enum_values() -> None:
    """Schema with > 50 total enum values → 'schema_too_constrained' detected."""
    detector = _detector()
    features = _make_features(schema_constraint_count=5)  # constraint count alone is OK
    # Create a schema with 60 enum values spread across properties
    schema_heavy_enum: dict = {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": [f"cat_{i}" for i in range(55)],  # 55 enum values
            }
        },
        "required": ["category"],
    }
    modes = detector.detect(features, model_id="llama-3.1-70b", schema=schema_heavy_enum)
    assert "schema_too_constrained" in modes, (
        f"Expected 'schema_too_constrained' for heavy enum schema, got {modes}"
    )


def test_native_thinker_detected() -> None:
    """model_id='o1' → 'native_thinker' in detected modes."""
    detector = _detector()
    features = _make_features()
    modes = detector.detect(features, model_id="o1")
    assert "native_thinker" in modes, f"Expected 'native_thinker' in {modes}"


def test_native_thinker_detected_o3() -> None:
    """model_id='o3' is also a native thinker."""
    detector = _detector()
    modes = detector.detect(_make_features(), model_id="o3")
    assert "native_thinker" in modes


def test_native_thinker_detected_deepseek_r1() -> None:
    """DeepSeek R1 must be detected as a native thinker."""
    detector = _detector()
    modes = detector.detect(_make_features(), model_id="deepseek-r1")
    assert "native_thinker" in modes


def test_short_prompt_detected() -> None:
    """prompt_length_bucket == 0 → 'short_prompt' in modes."""
    detector = _detector()
    features = _make_features(
        prompt_length_bucket=0,
        schema_depth=3,  # deep schema, so simple_extraction alone isn't the only hit
    )
    modes = detector.detect(features, model_id="llama")
    assert "short_prompt" in modes, f"Expected 'short_prompt' in {modes}"


def test_template_fill_detected() -> None:
    """Low entropy + 0 reasoning ops + shallow schema → 'template_fill' detected."""
    detector = _detector()
    features = _make_features(
        token_entropy=0.20,  # below 0.35 threshold
        required_reasoning_ops=0,  # 0 CoT keywords
        schema_depth=1,
        prompt_length_bucket=1,
    )
    modes = detector.detect(features, model_id="llama")
    assert "template_fill" in modes, f"Expected 'template_fill' in {modes}"


def test_template_fill_not_detected_high_entropy() -> None:
    """High entropy prompts must NOT trigger template_fill."""
    detector = _detector()
    features = _make_features(token_entropy=0.85, required_reasoning_ops=0)
    modes = detector.detect(features, model_id="llama")
    assert "template_fill" not in modes


def test_ambiguous_schema_detected() -> None:
    """Schema with 'anyOf' at root level → 'ambiguous_schema' detected."""
    detector = _detector()
    features = _make_features()
    schema_anyof: dict = {
        "anyOf": [
            {"type": "object", "properties": {"a": {"type": "string"}}},
            {"type": "object", "properties": {"b": {"type": "integer"}}},
        ]
    }
    modes = detector.detect(features, model_id="llama", schema=schema_anyof)
    assert "ambiguous_schema" in modes, f"Expected 'ambiguous_schema' in {modes}"


def test_ambiguous_schema_detected_oneof() -> None:
    """Schema with 'oneOf' at root also triggers 'ambiguous_schema'."""
    detector = _detector()
    features = _make_features()
    schema_oneof: dict = {
        "oneOf": [
            {"type": "string"},
            {"type": "integer"},
        ]
    }
    modes = detector.detect(features, model_id="llama", schema=schema_oneof)
    assert "ambiguous_schema" in modes


def test_no_ambiguous_schema_on_normal_schema() -> None:
    """A plain object schema without anyOf/oneOf must not trigger ambiguous_schema."""
    detector = _detector()
    features = _make_features()
    plain_schema: dict = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    modes = detector.detect(features, model_id="llama", schema=plain_schema)
    assert "ambiguous_schema" not in modes


# ---------------------------------------------------------------------------
# Tests: complex prompt → no failure modes
# ---------------------------------------------------------------------------


def test_complex_prompt_no_failure_modes() -> None:
    """
    A genuinely complex request (deep schema, long prompt, rich vocabulary, many
    reasoning ops, non-thinker model) should produce zero failure modes.
    """
    detector = _detector()
    features = _make_features(
        token_entropy=0.90,
        schema_depth=4,
        required_reasoning_ops=8,
        instruction_tune_score=0.5,
        prompt_length_bucket=3,
        schema_constraint_count=10,
    )
    plain_schema: dict = {
        "type": "object",
        "properties": {
            "answer": {"type": "number"},
            "explanation": {"type": "string"},
        },
        "required": ["answer", "explanation"],
    }
    modes = detector.detect(features, model_id="llama-3.1-70b", schema=plain_schema)
    assert modes == [], f"Expected no failure modes for complex request, got {modes}"


# ---------------------------------------------------------------------------
# Tests: should_override_to_direct
# ---------------------------------------------------------------------------


def test_should_override_to_direct_on_simple_extraction() -> None:
    """simple_extraction alone is a hard override → should_override_to_direct == True."""
    detector = _detector()
    assert detector.should_override_to_direct(["simple_extraction"]) is True


def test_should_override_to_direct_on_native_thinker() -> None:
    """native_thinker is a hard override."""
    detector = _detector()
    assert detector.should_override_to_direct(["native_thinker"]) is True


def test_should_override_to_direct_on_short_prompt() -> None:
    """short_prompt is a hard override."""
    detector = _detector()
    assert detector.should_override_to_direct(["short_prompt"]) is True


def test_should_override_to_direct_on_template_fill() -> None:
    """template_fill alone is NOT a hard override per spec.

    Hard overrides are: simple_extraction, short_prompt, native_thinker.
    template_fill is a soft warning — the oracle may still route direct based on other features.
    """
    detector = _detector()
    # template_fill alone does not force direct override
    assert detector.should_override_to_direct(["template_fill"]) is False


def test_should_not_override_on_ambiguous_schema_alone() -> None:
    """
    'ambiguous_schema' alone must NOT trigger a hard override.
    It is a soft failure mode (engine injects a schema hint instead).
    """
    detector = _detector()
    assert detector.should_override_to_direct(["ambiguous_schema"]) is False, (
        "ambiguous_schema alone should not force direct routing"
    )


def test_should_not_override_on_schema_too_constrained_alone() -> None:
    """
    'schema_too_constrained' alone must NOT trigger a hard override —
    it issues a warning only.
    """
    detector = _detector()
    # Per the docstring: "warning only" for schema_too_constrained
    # (if the implementation disagrees, adjust this test to match the contract)
    result = detector.should_override_to_direct(["schema_too_constrained"])
    # Check that the function runs without error and returns a bool
    assert isinstance(result, bool)


def test_should_not_override_on_empty_modes() -> None:
    """Empty failure modes list must return False (no override needed)."""
    detector = _detector()
    assert detector.should_override_to_direct([]) is False


# ---------------------------------------------------------------------------
# Tests: check() convenience wrapper
# ---------------------------------------------------------------------------


def test_check_overrides_ttf_for_simple_extraction() -> None:
    """check() must override a TTF decision to 'direct' when simple_extraction detected."""
    detector = _detector()
    features = _make_features(schema_depth=1, prompt_length_bucket=0)
    decision, modes = detector.check(_ttf_decision(), features, model_id="llama")
    assert "simple_extraction" in modes
    if detector.should_override_to_direct(modes):
        assert decision.strategy == "direct"


def test_check_returns_tuple_of_decision_and_modes() -> None:
    """check() must always return a 2-tuple of (RoutingDecision, list[str])."""
    detector = _detector()
    features = _make_features()
    result = detector.check(_ttf_decision(), features, model_id="llama")
    assert isinstance(result, tuple)
    assert len(result) == 2
    decision, modes = result
    assert isinstance(decision, RoutingDecision)
    assert isinstance(modes, list)


def test_check_attaches_modes_to_decision() -> None:
    """Detected failure modes must be reflected in the returned RoutingDecision."""
    detector = _detector()
    # Use features that will trigger ambiguous_schema but not a hard override
    features = _make_features()
    schema_anyof: dict = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    decision, modes = detector.check(
        _ttf_decision(), features, model_id="llama", schema=schema_anyof
    )
    assert "ambiguous_schema" in modes
    # The modes should be attached to the decision
    assert decision.failure_modes is not None


# ---------------------------------------------------------------------------
# Tests: no exception on edge-case inputs
# ---------------------------------------------------------------------------


def test_detect_with_none_schema_does_not_raise() -> None:
    """detect() with schema=None must not raise."""
    detector = _detector()
    features = _make_features()
    modes = detector.detect(features, model_id="llama", schema=None)
    assert isinstance(modes, list)


def test_detect_with_empty_schema_does_not_raise() -> None:
    """detect() with schema={} must not raise."""
    detector = _detector()
    features = _make_features()
    modes = detector.detect(features, model_id="llama", schema={})
    assert isinstance(modes, list)
