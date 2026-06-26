"""Tests for providers._base — BaseProvider ABC."""

from __future__ import annotations

import pytest

from nfield.exceptions import ProviderError
from nfield.providers import _base
from nfield.providers._base import BaseProvider


class MockProvider(BaseProvider):
    """Mock provider for testing BaseProvider."""

    def __init__(
        self,
        model_name: str,
        *,
        max_retries: int = 3,
        backoff_base: float = 2.0,
        backoff_max: float = 30.0,
        context_window: int | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        """Initialize mock provider."""
        super().__init__(
            model_name,
            max_retries=max_retries,
            backoff_base=backoff_base,
            backoff_max=backoff_max,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
        )

    async def _raw_complete(self, messages: list[dict[str, str]], *, max_tokens: int) -> str:
        """Mock complete implementation."""
        return "mock response"

    def _get_client(self):
        """Mock client getter."""
        return object()

    @property
    def context_window(self) -> int:
        """Mock context window with fallback."""
        if self._context_window is not None:
            return self._context_window
        return 8192

    @property
    def max_output_tokens(self) -> int:
        """Mock max output tokens with fallback."""
        if self._max_output_tokens is not None:
            return self._max_output_tokens
        return 2048


class TestBaseProvider:
    """Tests for BaseProvider."""

    def test_provider_initialization(self) -> None:
        """Provider initializes with model name."""
        provider = MockProvider("test-model")
        assert provider.model_name == "test-model"

    @pytest.mark.asyncio
    async def test_complete_calls_raw_complete(self) -> None:
        """Complete method calls _raw_complete."""
        provider = MockProvider("test-model")
        result = await provider.complete([{"role": "user", "content": "hello"}], max_tokens=100)
        assert result == "mock response"

    def test_context_window_property(self) -> None:
        """Context window property is accessible."""
        provider = MockProvider("test-model")
        assert provider.context_window == 8192

    def test_max_output_tokens_property(self) -> None:
        """Max output tokens property is accessible."""
        provider = MockProvider("test-model")
        assert provider.max_output_tokens == 2048

    def test_context_window_override(self) -> None:
        """Context window can be overridden at initialization."""
        provider = MockProvider("test-model", context_window=16384)
        assert provider.context_window == 16384

    def test_max_output_tokens_override(self) -> None:
        """Max output tokens can be overridden at initialization."""
        provider = MockProvider("test-model", max_output_tokens=4096)
        assert provider.max_output_tokens == 4096

    def test_both_specs_override(self) -> None:
        """Both context_window and max_output_tokens can be overridden."""
        provider = MockProvider(
            "test-model",
            context_window=32768,
            max_output_tokens=8192,
        )
        assert provider.context_window == 32768
        assert provider.max_output_tokens == 8192

    def test_context_window_default_fallback(self) -> None:
        """Context window defaults to 8192 when not specified."""
        provider = MockProvider("test-model")
        assert provider.context_window == 8192

    def test_max_output_tokens_default_fallback(self) -> None:
        """Max output tokens defaults to 2048 when not specified."""
        provider = MockProvider("test-model")
        assert provider.max_output_tokens == 2048

    def test_default_retry_policy_survives_a_tpm_window(self) -> None:
        """Fix A: defaults must outlast a ~60s 429 window, not surrender in ~7s."""
        from nfield.providers import _base

        assert _base._DEFAULT_RETRY_ATTEMPTS >= 6
        assert _base._DEFAULT_BACKOFF_MAX >= 60.0

    def test_invalid_max_retries_raises_error(self) -> None:
        """Invalid max_retries parameter raises ValueError."""
        with pytest.raises(ValueError, match="max_retries must be > 0"):
            MockProvider("test-model", max_retries=0)
        with pytest.raises(ValueError, match="max_retries must be > 0"):
            MockProvider("test-model", max_retries=-1)

    def test_invalid_backoff_base_raises_error(self) -> None:
        """Invalid backoff_base parameter raises ValueError."""
        with pytest.raises(ValueError, match="backoff_base must be > 0"):
            MockProvider("test-model", backoff_base=0)
        with pytest.raises(ValueError, match="backoff_base must be > 0"):
            MockProvider("test-model", backoff_base=-1)

    def test_invalid_backoff_max_raises_error(self) -> None:
        """Invalid backoff_max parameter raises ValueError."""
        with pytest.raises(
            ValueError,
            match=r"backoff_max.*must be >= backoff_base",
        ):
            MockProvider("test-model", backoff_base=2.0, backoff_max=1.0)


class _FlakyProvider(MockProvider):
    """Fails the first ``fail_times`` calls with ``error``, then returns 'ok'."""

    def __init__(self, *, fail_times: int, error: ProviderError, max_retries: int = 3) -> None:
        super().__init__("flaky", max_retries=max_retries, backoff_base=2.0, backoff_max=30.0)
        self._fail_times = fail_times
        self._error = error
        self.calls = 0

    async def _raw_complete(self, messages: list[dict[str, str]], *, max_tokens: int) -> str:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._error
        return "ok"


class TestProviderErrorClassification:
    def test_408_429_5xx_are_retryable(self) -> None:
        assert ProviderError("x", status_code=408).retryable
        assert ProviderError("x", status_code=429).retryable
        assert ProviderError("x", status_code=503).retryable

    def test_permanent_4xx_not_retryable(self) -> None:
        assert not ProviderError("x", status_code=404).retryable
        assert not ProviderError("x", status_code=401).retryable

    def test_unknown_status_not_retryable_by_default(self) -> None:
        assert not ProviderError("x", status_code=None).retryable

    def test_explicit_override_wins(self) -> None:
        # A timeout: no status code, but explicitly transient.
        assert ProviderError("timeout", status_code=None, retryable=True).retryable
        assert not ProviderError("x", status_code=503, retryable=False).retryable

    def test_retry_after_is_carried(self) -> None:
        assert ProviderError("x", status_code=429, retry_after=7.5).retry_after == 7.5


class TestRetryBehavior:
    async def test_retries_retryable_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_base.asyncio, "sleep", _no_sleep())
        provider = _FlakyProvider(fail_times=2, error=ProviderError("rate", status_code=429))
        result = await provider.complete([{"role": "user", "content": "hi"}], max_tokens=10)
        assert result == "ok"
        assert provider.calls == 3

    async def test_timeout_override_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_base.asyncio, "sleep", _no_sleep())
        timeout = ProviderError("timed out", status_code=None, retryable=True)
        provider = _FlakyProvider(fail_times=1, error=timeout)
        assert await provider.complete([], max_tokens=10) == "ok"
        assert provider.calls == 2

    async def test_non_retryable_raises_immediately(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_base.asyncio, "sleep", _no_sleep())
        provider = _FlakyProvider(fail_times=3, error=ProviderError("bad", status_code=400))
        with pytest.raises(ProviderError):
            await provider.complete([], max_tokens=10)
        assert provider.calls == 1

    async def test_gives_up_after_max_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_base.asyncio, "sleep", _no_sleep())
        provider = _FlakyProvider(
            fail_times=99, error=ProviderError("rate", status_code=429), max_retries=3
        )
        with pytest.raises(ProviderError):
            await provider.complete([], max_tokens=10)
        assert provider.calls == 3

    async def test_honors_retry_after(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sleeps: list[float] = []

        async def record(seconds: float) -> None:
            sleeps.append(seconds)

        monkeypatch.setattr(_base.asyncio, "sleep", record)
        err = ProviderError("rate", status_code=429, retry_after=5.0)
        provider = _FlakyProvider(fail_times=1, error=err)
        await provider.complete([], max_tokens=10)
        # Retry-After below the cap is honored, plus a small decorrelation jitter (< 1s).
        assert len(sleeps) == 1
        assert 5.0 <= sleeps[0] < 6.0

    async def test_caps_large_retry_after(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sleeps: list[float] = []

        async def record(seconds: float) -> None:
            sleeps.append(seconds)

        monkeypatch.setattr(_base.asyncio, "sleep", record)
        # A full-window TPM Retry-After (~55s) is capped to rate_limit_backoff_max
        # (8s) + jitter — the bucket refills continuously, so we don't sleep a window.
        err = ProviderError("rate", status_code=429, retry_after=55.0)
        provider = _FlakyProvider(fail_times=1, error=err)
        await provider.complete([], max_tokens=10)
        assert len(sleeps) == 1
        assert 8.0 <= sleeps[0] < 9.0

    async def test_full_jitter_on_exponential_branch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A 5xx has no Retry-After → exponential branch, which uses FULL jitter:
        # uniform(0, ceiling) where ceiling = min(base**attempt, backoff_max). The old
        # code did additive uniform(0, 1) on top of base**attempt; capturing the jitter
        # range proves the new wait spans the whole [0, ceiling] window.
        ranges: list[tuple[float, float]] = []
        sleeps: list[float] = []

        def fake_uniform(low: float, high: float) -> float:
            ranges.append((low, high))
            return high  # deterministic: take the top of the range

        async def record(seconds: float) -> None:
            sleeps.append(seconds)

        monkeypatch.setattr(_base.random, "uniform", fake_uniform)
        monkeypatch.setattr(_base.asyncio, "sleep", record)
        provider = _FlakyProvider(fail_times=2, error=ProviderError("server", status_code=503))
        assert await provider.complete([], max_tokens=10) == "ok"
        # base=2.0 → ceilings min(2**0, 30)=1, min(2**1, 30)=2; full jitter = uniform(0, ceiling).
        assert ranges == [(0, 1.0), (0, 2.0)]
        assert sleeps == [1.0, 2.0]


def _no_sleep():
    async def _sleep(_seconds: float) -> None:
        return None

    return _sleep
