"""
Unit tests for formatshield._retry.

Covers RetryConfig.delay_for(), with_retry() success / retry / exhaustion paths,
and the pre-built DEFAULT_RETRY / API_RETRY sentinel objects.
"""

from __future__ import annotations

import pytest

from formatshield._retry import (
    API_RETRY,
    DEFAULT_RETRY,
    RetryConfig,
    with_retry,
)

# ---------------------------------------------------------------------------
# RetryConfig
# ---------------------------------------------------------------------------


def test_retry_config_defaults() -> None:
    """Default RetryConfig has expected field values."""
    cfg = RetryConfig()
    assert cfg.max_attempts == 3
    assert cfg.base_delay == 1.0
    assert cfg.max_delay == 30.0
    assert cfg.jitter is True


def test_retry_config_custom_values() -> None:
    """RetryConfig stores custom values correctly."""
    cfg = RetryConfig(max_attempts=5, base_delay=0.5, max_delay=10.0, jitter=False)
    assert cfg.max_attempts == 5
    assert cfg.base_delay == 0.5
    assert cfg.max_delay == 10.0
    assert cfg.jitter is False


def test_retry_config_frozen() -> None:
    """RetryConfig is immutable (frozen dataclass)."""
    cfg = RetryConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.max_attempts = 99  # type: ignore[misc]


def test_delay_for_attempt_0_no_jitter() -> None:
    """Attempt 0 with no jitter returns base_delay exactly."""
    cfg = RetryConfig(base_delay=1.0, max_delay=30.0, jitter=False)
    assert cfg.delay_for(0) == pytest.approx(1.0)


def test_delay_for_attempt_1_no_jitter() -> None:
    """Attempt 1 doubles the delay."""
    cfg = RetryConfig(base_delay=1.0, max_delay=30.0, jitter=False)
    assert cfg.delay_for(1) == pytest.approx(2.0)


def test_delay_for_attempt_2_no_jitter() -> None:
    """Attempt 2 quadruples the base delay."""
    cfg = RetryConfig(base_delay=1.0, max_delay=30.0, jitter=False)
    assert cfg.delay_for(2) == pytest.approx(4.0)


def test_delay_for_capped_at_max_delay() -> None:
    """Delay is capped at max_delay regardless of attempt number."""
    cfg = RetryConfig(base_delay=1.0, max_delay=5.0, jitter=False)
    # 2^10 * 1.0 = 1024 >> 5.0
    assert cfg.delay_for(10) == pytest.approx(5.0)


def test_delay_for_jitter_within_range() -> None:
    """With jitter, delay stays within [0.5 * raw, 1.5 * raw]."""
    cfg = RetryConfig(base_delay=2.0, max_delay=100.0, jitter=True)
    for _ in range(50):
        d = cfg.delay_for(0)
        assert 1.0 <= d <= 3.0, f"jitter out of range: {d}"


# ---------------------------------------------------------------------------
# Pre-built configs
# ---------------------------------------------------------------------------


def test_default_retry_is_retryconfig() -> None:
    """DEFAULT_RETRY is a RetryConfig instance."""
    assert isinstance(DEFAULT_RETRY, RetryConfig)


def test_default_retry_max_attempts() -> None:
    """DEFAULT_RETRY has 3 attempts."""
    assert DEFAULT_RETRY.max_attempts == 3


def test_api_retry_is_retryconfig() -> None:
    """API_RETRY is a RetryConfig instance."""
    assert isinstance(API_RETRY, RetryConfig)


def test_api_retry_more_attempts_than_default() -> None:
    """API_RETRY allows more attempts than DEFAULT_RETRY."""
    assert API_RETRY.max_attempts > DEFAULT_RETRY.max_attempts


def test_api_retry_longer_base_delay() -> None:
    """API_RETRY has a longer base_delay than DEFAULT_RETRY."""
    assert API_RETRY.base_delay >= DEFAULT_RETRY.base_delay


# ---------------------------------------------------------------------------
# with_retry — success paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_with_retry_succeeds_on_first_attempt() -> None:
    """with_retry returns the value immediately when the first call succeeds."""
    call_count = 0

    async def _call() -> str:
        nonlocal call_count
        call_count += 1
        return "ok"

    result = await with_retry(_call, RetryConfig(max_attempts=3, base_delay=0.001, jitter=False))
    assert result == "ok"
    assert call_count == 1


@pytest.mark.asyncio
async def test_with_retry_succeeds_on_second_attempt() -> None:
    """with_retry retries after the first failure and returns on second attempt."""
    call_count = 0

    async def _call() -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("transient")
        return "ok"

    cfg = RetryConfig(max_attempts=3, base_delay=0.001, jitter=False)
    result = await with_retry(_call, cfg, retryable=(ValueError,), operation_name="test")
    assert result == "ok"
    assert call_count == 2


@pytest.mark.asyncio
async def test_with_retry_retryable_override() -> None:
    """The retryable= parameter overrides config.retryable_exceptions."""
    call_count = 0

    async def _call() -> int:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("retry me")
        return 42

    cfg = RetryConfig(
        max_attempts=5,
        base_delay=0.001,
        jitter=False,
        retryable_exceptions=(TypeError,),  # won't match RuntimeError
    )
    result = await with_retry(_call, cfg, retryable=(RuntimeError,), operation_name="test")
    assert result == 42
    assert call_count == 3


# ---------------------------------------------------------------------------
# with_retry — exhaustion paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_with_retry_reraises_after_exhaustion() -> None:
    """with_retry re-raises the last exception when all attempts fail."""

    async def _always_fail() -> str:
        raise ConnectionError("down")

    cfg = RetryConfig(max_attempts=2, base_delay=0.001, jitter=False)
    with pytest.raises(ConnectionError, match="down"):
        await with_retry(_always_fail, cfg, retryable=(ConnectionError,))


@pytest.mark.asyncio
async def test_with_retry_call_count_matches_max_attempts() -> None:
    """with_retry calls the function exactly max_attempts times on exhaustion."""
    call_count = 0

    async def _always_fail() -> None:
        nonlocal call_count
        call_count += 1
        raise OSError("fail")

    cfg = RetryConfig(max_attempts=3, base_delay=0.001, jitter=False)
    with pytest.raises(OSError, match="fail"):
        await with_retry(_always_fail, cfg, retryable=(OSError,))

    assert call_count == 3


# ---------------------------------------------------------------------------
# with_retry — non-retryable errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_with_retry_does_not_retry_non_retryable_error() -> None:
    """Non-retryable exceptions are re-raised immediately without retry."""
    call_count = 0

    async def _call() -> str:
        nonlocal call_count
        call_count += 1
        raise TypeError("not retryable")

    cfg = RetryConfig(max_attempts=5, base_delay=0.001, jitter=False)
    with pytest.raises(TypeError):
        await with_retry(_call, cfg, retryable=(ValueError,))

    assert call_count == 1  # no retry


@pytest.mark.asyncio
async def test_with_retry_single_attempt_config() -> None:
    """max_attempts=1 means no retries at all."""
    call_count = 0

    async def _call() -> None:
        nonlocal call_count
        call_count += 1
        raise ValueError("fail")

    cfg = RetryConfig(max_attempts=1, base_delay=0.001, jitter=False)
    with pytest.raises(ValueError, match="fail"):
        await with_retry(_call, cfg, retryable=(ValueError,))

    assert call_count == 1
