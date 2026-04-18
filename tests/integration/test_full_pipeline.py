"""
Full pipeline e2e test — the north star test.

All components working together with MockBackend:
  ComplexityScorer → ThresholdOracle → FailureModeDetector → TTFEngine → MockBackend

No real API keys required. Every test here validates end-to-end behavior.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from formatshield.core import FormatShield, GenerationResult


class SimpleAnswer(BaseModel):
    answer: str
    confidence: float = 0.9


class MathAnswer(BaseModel):
    steps: list[str]
    final_answer: float


SIMPLE_PROMPT = "What is 2+2?"
COMPLEX_PROMPT = (
    "A train leaves station A traveling at 60 mph. Another train leaves station B "
    "traveling at 80 mph. The stations are 280 miles apart. They travel toward each other. "
    "Calculate: (1) when they meet, (2) how far each traveled, "
    "(3) what time it is if they departed at 8 AM. "
    "Show all reasoning steps before giving the final answer."
)


@pytest.fixture
def shield_with_mock(mock_backend, monkeypatch) -> FormatShield:
    """FormatShield instance with MockBackend injected."""
    shield = FormatShield.__new__(FormatShield)
    shield.model = "groq/llama-3.1-70b-versatile"
    shield.backend_name = "groq"
    shield._backend = mock_backend
    shield._latency_budget_ms = None
    shield._cost_aware = False
    shield._ttf_fallback = True
    shield._expose_thinking = False
    shield._debug = False

    from formatshield.observability.logger import StructuredLogger
    from formatshield.observability.metrics import MetricsCollector
    from formatshield.oracle.oracle_x import OracleX
    from formatshield.oracle.threshold_oracle import ThresholdOracle
    from formatshield.scorer.complexity_scorer import ComplexityScorer
    from formatshield.ttf.failure_detector import FailureModeDetector
    from formatshield.hooks import Hooks

    shield._scorer = ComplexityScorer()
    shield._oracle = ThresholdOracle()
    shield._oracle_x = OracleX()
    shield._detector = FailureModeDetector()
    shield._metrics = MetricsCollector()
    shield._logger = StructuredLogger(level="WARNING")
    shield._hooks = Hooks()
    return shield


@pytest.mark.asyncio
async def test_generate_returns_generation_result(shield_with_mock) -> None:
    result = await shield_with_mock.generate(SIMPLE_PROMPT)
    assert isinstance(result, GenerationResult)


@pytest.mark.asyncio
async def test_generation_result_has_all_fields(shield_with_mock) -> None:
    result = await shield_with_mock.generate(SIMPLE_PROMPT)
    assert result.output is not None
    assert result.routing is not None
    assert isinstance(result.complexity_score, float)
    assert isinstance(result.latency_ms, float)
    assert isinstance(result.failure_modes, list)
    assert result.backend == "groq"
    assert result.model == "groq/llama-3.1-70b-versatile"


@pytest.mark.asyncio
async def test_routing_decision_populated(shield_with_mock) -> None:
    result = await shield_with_mock.generate(COMPLEX_PROMPT)
    assert result.routing.strategy in {"ttf", "direct", "hybrid"}
    assert 0.0 <= result.routing.confidence <= 1.0
    assert isinstance(result.routing.explanation, str)
    assert len(result.routing.explanation) > 0


@pytest.mark.asyncio
async def test_simple_prompt_routes_to_direct(shield_with_mock) -> None:
    # Very short simple prompt should be detected as direct
    result = await shield_with_mock.generate("Hi")
    # Short prompts trigger failure mode detection → direct
    assert result.routing.strategy == "direct"


@pytest.mark.asyncio
async def test_native_thinker_always_direct(mock_backend) -> None:
    shield = FormatShield.__new__(FormatShield)
    shield.model = "groq/o1"
    shield.backend_name = "groq"
    shield._backend = mock_backend
    shield._latency_budget_ms = None
    shield._cost_aware = False
    shield._ttf_fallback = True
    shield._expose_thinking = False
    shield._debug = False

    from formatshield.hooks import Hooks
    from formatshield.observability.logger import StructuredLogger
    from formatshield.observability.metrics import MetricsCollector
    from formatshield.oracle.oracle_x import OracleX
    from formatshield.oracle.threshold_oracle import ThresholdOracle
    from formatshield.scorer.complexity_scorer import ComplexityScorer
    from formatshield.ttf.failure_detector import FailureModeDetector

    shield._scorer = ComplexityScorer()
    shield._oracle = ThresholdOracle()
    shield._oracle_x = OracleX()
    shield._detector = FailureModeDetector()
    shield._metrics = MetricsCollector()
    shield._logger = StructuredLogger()
    shield._hooks = Hooks()

    result = await shield.generate(COMPLEX_PROMPT)
    assert result.routing.strategy == "direct"


@pytest.mark.asyncio
async def test_schema_validation_works(shield_with_mock) -> None:
    result = await shield_with_mock.generate(SIMPLE_PROMPT, schema=SimpleAnswer)
    # MockBackend returns valid JSON → parsed should be set
    assert result.parsed is not None


@pytest.mark.asyncio
async def test_generate_sync_works(shield_with_mock) -> None:
    result = shield_with_mock.generate_sync(SIMPLE_PROMPT)
    assert isinstance(result, GenerationResult)
    assert result.output


@pytest.mark.asyncio
async def test_stream_yields_events(shield_with_mock) -> None:
    events = []
    async for event in shield_with_mock.stream(SIMPLE_PROMPT):
        events.append(event)
        if len(events) > 10:  # safety limit
            break

    assert len(events) >= 1
    types = {e.type for e in events}
    assert types <= {"thinking", "output", "complete"}


@pytest.mark.asyncio
async def test_debug_mode_prints_trace(shield_with_mock, capsys) -> None:
    await shield_with_mock.generate(SIMPLE_PROMPT, debug=True)
    captured = capsys.readouterr()
    assert "[FormatShield]" in captured.out


@pytest.mark.asyncio
async def test_full_pipeline_no_exceptions(shield_with_mock) -> None:
    """Smoke test: entire pipeline runs without exceptions."""
    result = await shield_with_mock.generate(
        COMPLEX_PROMPT,
        schema={"type": "object", "properties": {"answer": {"type": "string"}}},
    )
    assert result is not None
    assert isinstance(result.output, str)
