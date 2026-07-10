"""Engine-level usage wiring: token counts and cost land on the result's metadata.

A mock provider reports fixed token counts per call the way BaseProvider subclasses
do; the engine's per-run counter must fold them into ``Metadata.tokens_prompt`` /
``tokens_completion``, and price them into ``Metadata.cost`` only when the config
carries a ``pricing``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from nfield import AsyncNField
from nfield.config import ExtractionConfig
from nfield.providers._usage import record_usage

if TYPE_CHECKING:
    from nfield.types import ExtractionResult

_DOC = "Name: Alice. Age: 30."
_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
}
_ECHO = "name = Alice\nage = 30"
_PROMPT_TOKENS_PER_CALL = 100
_COMPLETION_TOKENS_PER_CALL = 10


class _ReportingMock:
    """Mock provider that reports fixed usage per call, like a real provider."""

    model_name = "mock/echo"
    context_window = 8192
    max_output_tokens = 8192

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages: list[dict[str, str]], *, max_tokens: int) -> str:
        self.calls += 1
        record_usage(_PROMPT_TOKENS_PER_CALL, _COMPLETION_TOKENS_PER_CALL)
        return _ECHO


async def _run(
    config: ExtractionConfig, monkeypatch: pytest.MonkeyPatch
) -> tuple[ExtractionResult, _ReportingMock]:
    provider = _ReportingMock()
    monkeypatch.setattr("nfield.engine._async.from_model", lambda *_a, **_k: provider)
    engine = AsyncNField("mock/echo", _SCHEMA, config=config)
    return await engine.extract(_DOC), provider


class TestTokensOnMetadata:
    async def test_tokens_reflect_every_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result, provider = await _run(ExtractionConfig(max_retry_rounds=0), monkeypatch)
        assert provider.calls > 0
        assert result.metadata.tokens_prompt == provider.calls * _PROMPT_TOKENS_PER_CALL
        assert result.metadata.tokens_completion == provider.calls * _COMPLETION_TOKENS_PER_CALL

    async def test_cost_unset_without_pricing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result, _ = await _run(ExtractionConfig(max_retry_rounds=0), monkeypatch)
        assert result.metadata.cost is None

    async def test_cost_computed_from_pricing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result, provider = await _run(
            ExtractionConfig(max_retry_rounds=0, pricing=(1.0, 3.0)), monkeypatch
        )
        expected = (
            provider.calls * _PROMPT_TOKENS_PER_CALL * 1.0
            + provider.calls * _COMPLETION_TOKENS_PER_CALL * 3.0
        ) / 1e6
        assert result.metadata.cost == pytest.approx(expected)

    async def test_second_run_counts_only_itself(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _ReportingMock()
        monkeypatch.setattr("nfield.engine._async.from_model", lambda *_a, **_k: provider)
        engine = AsyncNField("mock/echo", _SCHEMA, config=ExtractionConfig(max_retry_rounds=0))
        first = await engine.extract(_DOC)
        calls_first = provider.calls
        second = await engine.extract(_DOC)
        calls_second = provider.calls - calls_first
        # A reused engine must not carry the first run's tally into the second.
        assert first.metadata.tokens_prompt == calls_first * _PROMPT_TOKENS_PER_CALL
        assert second.metadata.tokens_prompt == calls_second * _PROMPT_TOKENS_PER_CALL


class TestPricingValidation:
    def test_negative_price_rejected(self) -> None:
        with pytest.raises(ValueError, match="pricing"):
            ExtractionConfig(pricing=(-1.0, 2.0))

    def test_zero_prices_accepted(self) -> None:
        assert ExtractionConfig(pricing=(0.0, 0.0)).pricing == (0.0, 0.0)
