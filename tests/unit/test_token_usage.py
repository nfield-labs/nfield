"""Unit tests for TokenUsage dataclass (GROUP N — Stage 4)."""

from __future__ import annotations

from formatshield.scorer.features import TokenUsage


class TestTokenUsageBasics:
    def test_default_all_none(self) -> None:
        usage = TokenUsage()
        assert usage.input_tokens is None
        assert usage.output_tokens is None
        assert usage.cached_tokens is None
        assert usage.total_tokens is None
        assert usage.ttft_ms is None
        assert usage.forward_passes == 1

    def test_total_computed_when_both_known(self) -> None:
        usage = TokenUsage(input_tokens=150, output_tokens=42)
        assert usage.total_tokens == 192

    def test_total_not_computed_when_only_input(self) -> None:
        usage = TokenUsage(input_tokens=150)
        assert usage.total_tokens is None

    def test_total_not_computed_when_only_output(self) -> None:
        usage = TokenUsage(output_tokens=42)
        assert usage.total_tokens is None

    def test_explicit_total_not_overridden(self) -> None:
        # If caller provides total_tokens, __post_init__ should not override it
        usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=999)
        assert usage.total_tokens == 999

    def test_ttf_forward_passes(self) -> None:
        usage = TokenUsage(forward_passes=2)
        assert usage.forward_passes == 2


class TestTokenUsageToDict:
    def test_to_dict_all_fields(self) -> None:
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            cached_tokens=10,
            ttft_ms=42.5,
            forward_passes=2,
        )
        d = usage.to_dict()
        assert d["input_tokens"] == 100
        assert d["output_tokens"] == 50
        assert d["cached_tokens"] == 10
        assert d["total_tokens"] == 150
        assert d["ttft_ms"] == 42.5
        assert d["forward_passes"] == 2

    def test_to_dict_none_values(self) -> None:
        usage = TokenUsage()
        d = usage.to_dict()
        assert d["input_tokens"] is None
        assert d["total_tokens"] is None


class TestTokenUsageInGenerationResult:
    def test_generation_result_token_usage_defaults_none(self) -> None:
        from formatshield.backends.dryrun_backend import DryRunBackend
        from formatshield.core import FormatShield

        shield = FormatShield(model="dryrun/test", backend=DryRunBackend())
        result = shield.generate_sync("What is 2+2?")
        # DryRunBackend doesn't report token usage
        assert result.token_usage is None
        assert result.cost_usd is None

    def test_model_dump_includes_token_usage(self) -> None:
        from formatshield.backends.dryrun_backend import DryRunBackend
        from formatshield.core import FormatShield

        shield = FormatShield(model="dryrun/test", backend=DryRunBackend())
        result = shield.generate_sync("Hello")
        d = result.model_dump()
        assert "token_usage" in d
        assert "cost_usd" in d
        assert d["token_usage"] is None

    def test_model_dump_with_token_usage_populated(self) -> None:
        from formatshield.backends.dryrun_backend import DryRunBackend
        from formatshield.core import FormatShield

        shield = FormatShield(model="dryrun/test", backend=DryRunBackend())
        result = shield.generate_sync("Hello")
        # Manually set token_usage to simulate a backend that reports it
        object.__setattr__(result, "token_usage", TokenUsage(input_tokens=10, output_tokens=5))
        d = result.model_dump()
        assert d["token_usage"]["input_tokens"] == 10
        assert d["token_usage"]["total_tokens"] == 15
