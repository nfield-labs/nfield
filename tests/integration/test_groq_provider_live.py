"""Integration tests for GroqProvider using live API.

Tests the GroqProvider with real Groq API calls to verify:
- Client initialization and connectivity
- Message completion with retry logic
- Token counting and language detection
- Provider capabilities (context window, max output)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Load .env file before importing formatshield
env_file = Path(__file__).parent.parent.parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

from formatshield.exceptions import ProviderError  # noqa: E402
from formatshield.providers import from_model  # noqa: E402
from formatshield.providers._token_count import measure_chars_per_token  # noqa: E402


class TestGroqProviderLive:
    """Live API tests for GroqProvider."""

    @pytest.mark.asyncio
    async def test_groq_provider_complete_basic(self) -> None:
        """Test basic completion with Groq API."""
        provider = from_model("groq/llama-3.3-70b-versatile")

        messages = [
            {"role": "user", "content": "Say 'FormatShield' exactly."}
        ]

        result = await provider.complete(messages, max_tokens=10)

        # Should get a non-empty response
        assert result
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_groq_provider_count_tokens(self) -> None:
        """Test token counting with Groq."""
        provider = from_model("groq/llama-3.3-70b-versatile")

        text = "The quick brown fox jumps over the lazy dog."
        token_count = await provider.count_tokens(text)

        # Should get a positive integer
        assert isinstance(token_count, int)
        assert token_count > 0
        # Rough estimate: ~4 chars per token for English
        assert token_count < len(text)

    @pytest.mark.asyncio
    async def test_groq_provider_context_window(self) -> None:
        """Test provider context window is accessible."""
        provider = from_model("groq/llama-3.3-70b-versatile")

        # Llama 3.3 70B has 8192 context by default
        context = provider.context_window
        assert context > 0

    @pytest.mark.asyncio
    async def test_groq_provider_max_output_tokens(self) -> None:
        """Test max output tokens property."""
        provider = from_model("groq/llama-3.3-70b-versatile")

        max_out = provider.max_output_tokens
        assert max_out > 0

    @pytest.mark.asyncio
    async def test_groq_provider_model_name(self) -> None:
        """Test model name property."""
        provider = from_model("groq/llama-3.3-70b-versatile")

        assert provider.model_name == "llama-3.3-70b-versatile"

    @pytest.mark.asyncio
    async def test_groq_provider_respects_max_tokens(self) -> None:
        """Test that max_tokens parameter is respected."""
        provider = from_model("groq/llama-3.3-70b-versatile")

        messages = [
            {"role": "user", "content": "Count to 10."}
        ]

        # Request only 15 tokens
        result = await provider.complete(messages, max_tokens=15)

        # Result should be short
        assert len(result) < 300

    @pytest.mark.asyncio
    async def test_groq_provider_retry_on_transient_error(self) -> None:
        """Test retry logic on transient errors.

        Note: This test may not always trigger a 429 error,
        so it mainly verifies the infrastructure is in place.
        """
        provider = from_model("groq/llama-3.3-70b-versatile")

        # Just verify the retry max_retries is set
        assert provider._max_retries >= 1

    @pytest.mark.asyncio
    async def test_groq_provider_empty_message_handles_gracefully(self) -> None:
        """Test empty message handling."""
        provider = from_model("groq/llama-3.3-70b-versatile")

        messages = [{"role": "user", "content": ""}]

        # Should either get a response or raise a clear error
        try:
            result = await provider.complete(messages, max_tokens=10)
            # If it succeeds, should be a string
            assert isinstance(result, str)
        except ProviderError:
            # Expected - empty message often causes API error
            pass


class TestTokenCountingLive:
    """Live token counting tests."""

    @pytest.mark.asyncio
    async def test_measure_chars_per_token_english(self) -> None:
        """Test language-aware token counting for English."""
        provider = from_model("groq/llama-3.3-70b-versatile")

        ratio = await measure_chars_per_token(provider, language="en")

        # English should be around 3.5 chars per token
        assert 2.5 < ratio < 4.5

        # Cache the ratio
        provider.set_chars_per_token(ratio)
        assert provider.chars_per_token == ratio

    @pytest.mark.asyncio
    async def test_measure_chars_per_token_caching(self) -> None:
        """Test that chars_per_token can be cached after measurement."""
        provider = from_model("groq/llama-3.3-70b-versatile")

        # First call measures
        ratio1 = await measure_chars_per_token(provider, language="en")

        # Cache it
        provider.set_chars_per_token(ratio1)

        # Verify it's cached
        assert provider.chars_per_token == ratio1

    @pytest.mark.asyncio
    async def test_measure_chars_per_token_fallback(self) -> None:
        """Test fallback ratio when measurement fails."""
        # Manual fallback test: verify the fallback constants
        from formatshield.providers._token_count import (
            _FALLBACK_CHARS_PER_TOKEN_CJK,
            _FALLBACK_CHARS_PER_TOKEN_EN,
            _FALLBACK_CHARS_PER_TOKEN_MIXED,
        )

        assert _FALLBACK_CHARS_PER_TOKEN_EN == 3.5
        assert _FALLBACK_CHARS_PER_TOKEN_CJK == 1.5
        assert _FALLBACK_CHARS_PER_TOKEN_MIXED == 2.5


class TestProviderFactoryLive:
    """Live tests for provider factory."""

    @pytest.mark.asyncio
    async def test_from_model_groq_resolution(self) -> None:
        """Test that from_model resolves groq/ prefix correctly."""
        provider = from_model("groq/llama-3.3-70b-versatile")

        # Should be a valid provider
        assert hasattr(provider, "complete")
        assert hasattr(provider, "count_tokens")
        assert provider.model_name == "llama-3.3-70b-versatile"

    def test_from_model_invalid_prefix_raises_error(self) -> None:
        """Test that invalid prefix raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model string"):
            from_model("invalid_string_without_slash")

    def test_from_model_unknown_provider_raises_error(self) -> None:
        """Test that unknown provider raises ProviderError."""
        with pytest.raises(ProviderError, match="Unknown provider"):
            from_model("unknown-provider/model-name")

    @pytest.mark.asyncio
    async def test_groq_lazy_import_not_imported_until_use(self) -> None:
        """Test that groq is only imported when needed.

        This is important for keeping cold-start time low.
        """
        import sys

        # groq should not be in sys.modules yet if not imported
        had_groq_before = "groq" in sys.modules

        try:
            # Create provider (should lazy import groq)
            provider = from_model("groq/llama-3.1-8b")

            # groq should be imported now (via _get_client)
            _ = provider._get_client()
            assert "groq" in sys.modules
        finally:
            # Restore state if groq wasn't imported before
            if not had_groq_before and "groq" in sys.modules:
                del sys.modules["groq"]
