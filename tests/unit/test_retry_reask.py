"""Unit tests for FailedAttempt, FormatShieldRetryException, build_reask_prompt,
and TTFEngine reask behaviour (GROUP G — Stage 4)."""

from __future__ import annotations

import pytest

from formatshield._retry import (
    FailedAttempt,
    FormatShieldRetryException,
    build_reask_prompt,
)


class TestFailedAttempt:
    def test_basic_fields(self) -> None:
        exc = ValueError("bad json")
        fa = FailedAttempt(attempt_number=1, exception=exc, raw_output="{bad}")
        assert fa.attempt_number == 1
        assert fa.exception is exc
        assert fa.raw_output == "{bad}"

    def test_default_reask_prompt_is_empty(self) -> None:
        fa = FailedAttempt(attempt_number=1, exception=ValueError("x"), raw_output="")
        assert fa.reask_prompt == ""

    def test_custom_reask_prompt(self) -> None:
        fa = FailedAttempt(
            attempt_number=2,
            exception=ValueError("bad"),
            raw_output="output",
            reask_prompt="corrected prompt",
        )
        assert fa.reask_prompt == "corrected prompt"

    def test_is_namedtuple(self) -> None:
        fa = FailedAttempt(attempt_number=1, exception=ValueError(), raw_output="x")
        # NamedTuples are tuples
        assert isinstance(fa, tuple)
        assert fa[0] == 1


class TestFormatShieldRetryException:
    def test_message_and_empty_attempts(self) -> None:
        exc = FormatShieldRetryException("all failed")
        assert str(exc) == "all failed"
        assert exc.failed_attempts == []

    def test_with_attempts(self) -> None:
        fa = FailedAttempt(1, ValueError("oops"), "bad output")
        exc = FormatShieldRetryException("failed", failed_attempts=[fa])
        assert len(exc.failed_attempts) == 1
        assert exc.failed_attempts[0] is fa

    def test_last_attempt_none_when_empty(self) -> None:
        exc = FormatShieldRetryException("empty")
        assert exc.last_attempt is None

    def test_last_attempt_returns_last(self) -> None:
        fa1 = FailedAttempt(1, ValueError("e1"), "out1")
        fa2 = FailedAttempt(2, ValueError("e2"), "out2")
        exc = FormatShieldRetryException("x", failed_attempts=[fa1, fa2])
        assert exc.last_attempt is fa2

    def test_total_token_usage_sums_lengths(self) -> None:
        fa1 = FailedAttempt(1, ValueError(), "hello")  # 5
        fa2 = FailedAttempt(2, ValueError(), "world!")  # 6
        exc = FormatShieldRetryException("x", failed_attempts=[fa1, fa2])
        assert exc.total_token_usage == 11

    def test_total_token_usage_empty(self) -> None:
        exc = FormatShieldRetryException("x")
        assert exc.total_token_usage == 0

    def test_is_exception(self) -> None:
        exc = FormatShieldRetryException("x")
        assert isinstance(exc, Exception)


class TestBuildReaskPrompt:
    def test_contains_previous_attempt_marker(self) -> None:
        prompt = build_reask_prompt("What is 2+2?", "not-json", ValueError("bad json"))
        assert "PREVIOUS ATTEMPT" in prompt

    def test_contains_original_prompt(self) -> None:
        prompt = build_reask_prompt("What is 2+2?", "not-json", ValueError("bad"))
        assert "What is 2+2?" in prompt

    def test_contains_failed_output(self) -> None:
        prompt = build_reask_prompt("prompt", "FAILED_OUTPUT_HERE", ValueError("x"))
        assert "FAILED_OUTPUT_HERE" in prompt

    def test_contains_validation_error(self) -> None:
        err = ValueError("field required: answer")
        prompt = build_reask_prompt("prompt", "output", err)
        assert "field required: answer" in prompt

    def test_contains_schema_when_provided(self) -> None:
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        prompt = build_reask_prompt("prompt", "output", ValueError("x"), schema=schema)
        assert "answer" in prompt  # schema serialised in prompt

    def test_no_schema_hint_when_none(self) -> None:
        prompt = build_reask_prompt("prompt", "output", ValueError("x"), schema=None)
        # Should not contain "Required output schema:" section
        assert "Required output schema" not in prompt

    def test_returns_string(self) -> None:
        result = build_reask_prompt("p", "o", ValueError("e"))
        assert isinstance(result, str)
        assert len(result) > 0


class TestTTFEngineReask:
    """Test TTFEngine reask integration via DryRunBackend."""

    def test_max_reasks_default(self) -> None:
        from formatshield.backends.dryrun_backend import DryRunBackend
        from formatshield.ttf.engine import TTFEngine

        engine = TTFEngine(backend=DryRunBackend())
        assert engine._max_reasks == TTFEngine.DEFAULT_MAX_REASKS

    def test_max_reasks_custom(self) -> None:
        from formatshield.backends.dryrun_backend import DryRunBackend
        from formatshield.ttf.engine import TTFEngine

        engine = TTFEngine(backend=DryRunBackend(), max_reasks=0)
        assert engine._max_reasks == 0

    @pytest.mark.asyncio
    async def test_generate_valid_output_no_reask(self) -> None:
        """DryRunBackend always returns valid JSON — no reask needed."""
        from pydantic import BaseModel

        from formatshield.backends.dryrun_backend import DryRunBackend
        from formatshield.ttf.engine import TTFEngine

        class MyModel(BaseModel):
            answer: str

        engine = TTFEngine(backend=DryRunBackend(), max_reasks=2)
        thinking, output = await engine.generate(
            prompt="What is 2+2?",
            schema=MyModel.model_json_schema(),
            schema_model=MyModel,
        )
        # DryRunBackend should produce valid JSON; no fallback needed
        assert isinstance(thinking, str)
        assert isinstance(output, str)

    @pytest.mark.asyncio
    async def test_generate_direct_no_reask(self) -> None:
        from formatshield.backends.dryrun_backend import DryRunBackend
        from formatshield.ttf.engine import TTFEngine

        engine = TTFEngine(backend=DryRunBackend())
        output = await engine.generate_direct(
            prompt="Hello",
            schema={"type": "object", "properties": {"reply": {"type": "string"}}},
        )
        assert isinstance(output, str)

    def test_failed_attempt_namedtuple_fields(self) -> None:
        fa = FailedAttempt(
            attempt_number=3,
            exception=RuntimeError("oops"),
            raw_output="bad",
            reask_prompt="fix it",
        )
        assert fa.attempt_number == 3
        assert str(fa.exception) == "oops"
        assert fa.raw_output == "bad"
        assert fa.reask_prompt == "fix it"
