"""Tests for providers._base — BaseProvider ABC."""

from __future__ import annotations

import pytest

from formatshield.providers._base import BaseProvider


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

    async def _raw_count_tokens(self, text: str) -> int:
        """Mock token counting."""
        return len(text) // 4

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

    def test_chars_per_token_caching(self) -> None:
        """Chars per token is cached."""
        provider = MockProvider("test-model")
        assert provider.chars_per_token is None
        provider.set_chars_per_token(3.5)
        assert provider.chars_per_token == 3.5

    def test_invalid_chars_per_token_raises_error(self) -> None:
        """Setting invalid chars_per_token raises ValueError."""
        provider = MockProvider("test-model")
        with pytest.raises(ValueError, match="must be positive"):
            provider.set_chars_per_token(0)
        with pytest.raises(ValueError, match="must be positive"):
            provider.set_chars_per_token(-1)

    @pytest.mark.asyncio
    async def test_complete_calls_raw_complete(self) -> None:
        """Complete method calls _raw_complete."""
        provider = MockProvider("test-model")
        result = await provider.complete([{"role": "user", "content": "hello"}], max_tokens=100)
        assert result == "mock response"

    @pytest.mark.asyncio
    async def test_count_tokens_calls_raw_count_tokens(self) -> None:
        """Count tokens calls _raw_count_tokens."""
        provider = MockProvider("test-model")
        result = await provider.count_tokens("hello world")
        assert result > 0

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
