"""
End-to-end integration test for FormatShield — the north star proof condition.

This test exercises the single most important workflow:
  A user instantiates FormatShield with DryRunBackend, calls generate() with
  a realistic prompt and schema, and receives a GenerationResult with a
  routing trace showing which strategy (TTF or Direct) was used and why.

Rules (from build-skill/agents/integration-writer.md):
  - Written BEFORE gaps were filled — it is the proof target, not the current state.
  - Do NOT modify this test. If an implementation does not match, fix the code.
  - Uses DryRunBackend so it passes without API keys in CI.
  - Asserts on public API only — no internal module imports except FormatShield's own.

Written: 2026-04-13 (Phase 1 — integration-writer agent)
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

import formatshield as fs
from formatshield.backends.dryrun_backend import DryRunBackend
from formatshield.core import FormatShield, GenerationResult
from formatshield.oracle.routing_decision import RoutingDecision

# ---------------------------------------------------------------------------
# Schema for primary test
# ---------------------------------------------------------------------------


class FinancialMetrics(BaseModel):
    """Realistic schema: medium complexity, 3 required fields, one nested."""

    revenue_millions: float
    operating_margin_pct: float
    yoy_growth_pct: float
    period: str = "Q4 2025"


# ---------------------------------------------------------------------------
# Fixture — realistic input that exercises the routing decision
# ---------------------------------------------------------------------------


@pytest.fixture
def realistic_input() -> dict[str, object]:
    """A medium-complexity prompt (schema_depth=2, multi-step reasoning)."""
    return {
        "prompt": (
            "Analyze the following earnings report and extract the key financial metrics. "
            "The company reported total revenue of $2.4B in Q4 2025, representing 18% "
            "year-over-year growth. Operating expenses totaled $1.7B, giving operating "
            "income of $700M. "
            "Calculate the operating margin percentage and confirm the growth rate. "
            "Show your reasoning step by step before extracting the final numbers."
        ),
        "schema": FinancialMetrics,
        "model": "dryrun/test",
    }


# ---------------------------------------------------------------------------
# Primary test class — three methods, one fixture
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """Proof condition: FormatShield routes, generates, and returns valid structured output."""

    def test_complete_routing_workflow(self, realistic_input: dict[str, object]) -> None:
        """
        The system scores complexity, routes to TTF or Direct, generates structured
        output via DryRunBackend, validates against schema, and returns GenerationResult.
        """
        shield = FormatShield(model="dryrun/test", backend=DryRunBackend())

        result = shield.generate_sync(
            prompt=str(realistic_input["prompt"]),
            schema=realistic_input["schema"],  # type: ignore[arg-type]
        )

        # Must return a GenerationResult
        assert isinstance(result, GenerationResult)

        # Output must be a non-empty string
        assert isinstance(result.output, str)
        assert len(result.output) > 0

        # Routing trace must be present
        assert isinstance(result.routing, RoutingDecision)

        # Complexity score must be in [0, 1]
        assert isinstance(result.complexity_score, float)
        assert 0.0 <= result.complexity_score <= 1.0

        # Schema validation must have been attempted
        assert isinstance(result.schema_valid, bool)

        # No silent failures — latency recorded
        assert isinstance(result.latency_ms, float)
        assert result.latency_ms >= 0.0

    def test_routing_trace_contains_decision(self, realistic_input: dict[str, object]) -> None:
        """
        Routing trace shows which strategy was used (TTF / Direct / Hybrid) and why,
        with a human-readable explanation and confidence score.
        """
        shield = FormatShield(model="dryrun/test", backend=DryRunBackend())

        result = shield.generate_sync(
            prompt=str(realistic_input["prompt"]),
            schema=realistic_input["schema"],  # type: ignore[arg-type]
        )

        trace = result.routing

        # Strategy must be one of the three valid values
        assert trace.strategy in ("ttf", "direct", "hybrid"), (
            f"Unknown routing strategy: {trace.strategy!r}. Must be ttf, direct, or hybrid."
        )

        # Confidence must be a bounded float
        assert isinstance(trace.confidence, float)
        assert 0.0 <= trace.confidence <= 1.0, f"Confidence {trace.confidence} out of [0,1]"

        # Explanation must be non-empty — this powers debug mode
        assert isinstance(trace.explanation, str)
        assert len(trace.explanation) > 0, "Routing explanation must be non-empty for debug mode"

        # Expected delta is the paper's key measurement
        assert isinstance(trace.expected_accuracy_delta, float)

    def test_output_schema_validation(self, realistic_input: dict[str, object]) -> None:
        """
        Output from DryRunBackend is valid JSON that can be parsed.
        GenerationResult.schema_valid reflects the validation outcome.
        """
        shield = FormatShield(model="dryrun/test", backend=DryRunBackend())

        result = shield.generate_sync(
            prompt=str(realistic_input["prompt"]),
            schema=realistic_input["schema"],  # type: ignore[arg-type]
        )

        # Output must be parseable JSON
        try:
            parsed_output = json.loads(result.output)
        except json.JSONDecodeError as exc:
            pytest.fail(f"Output is not valid JSON: {exc}\nOutput was: {result.output!r}")

        assert isinstance(parsed_output, dict), "Output JSON must be a dict, not a list or scalar"

        # schema_valid must reflect the actual validation state
        assert result.schema_valid is True, (
            f"DryRunBackend should always produce schema-valid output. "
            f"schema_valid={result.schema_valid}, output={result.output!r}"
        )


# ---------------------------------------------------------------------------
# Additional smoke tests (outside the class — fast, no fixture dependency)
# ---------------------------------------------------------------------------


def test_module_level_generate_function_exists() -> None:
    """The top-level fs.generate() function is importable and is a callable."""
    assert callable(fs.generate), "fs.generate must be callable"


def test_formatshield_version_is_set() -> None:
    """Package version is set (not None, not empty) — required for PyPI."""
    assert isinstance(fs.__version__, str)
    assert len(fs.__version__) > 0


def test_generation_result_model_dump_is_json_serializable() -> None:
    """GenerationResult.model_dump() must produce a JSON-serializable dict — required for CLI."""
    shield = FormatShield(model="dryrun/test", backend=DryRunBackend())
    result = shield.generate_sync("What is 2+2?")

    dumped = result.model_dump()
    assert isinstance(dumped, dict)

    # Must be JSON-serializable (no non-serializable objects like RoutingDecision dataclass)
    serialized = json.dumps(dumped)
    assert len(serialized) > 0


def test_dryrun_backend_routes_without_api_key() -> None:
    """Confirms the core requirement: DryRunBackend works with zero external dependencies."""
    backend = DryRunBackend()
    shield = FormatShield(model="dryrun/test", backend=backend)
    result = shield.generate_sync("Test prompt", schema={"type": "object", "properties": {}})

    assert result is not None
    assert result.backend == "dryrun"
