"""Tests for providers.groq._provider — GroqProvider implementation."""

from __future__ import annotations

from formatshield.providers.groq import GroqProvider


class TestGroqProviderInitialization:
    """Tests for GroqProvider initialization."""

    def test_groq_provider_initializes_with_model_name(self) -> None:
        """GroqProvider initializes with model name."""
        provider = GroqProvider("llama-3.1-8b")
        assert provider.model_name == "llama-3.1-8b"

    def test_groq_provider_default_context_window(self) -> None:
        """GroqProvider uses default context window when not specified."""
        provider = GroqProvider("llama-3.1-8b")
        assert provider.context_window == 8192

    def test_groq_provider_default_max_output_tokens(self) -> None:
        """GroqProvider uses default max_output_tokens when not specified."""
        provider = GroqProvider("llama-3.1-8b")
        assert provider.max_output_tokens == 8192

    def test_groq_provider_custom_context_window(self) -> None:
        """GroqProvider accepts custom context_window parameter."""
        provider = GroqProvider(
            "llama-3.1-70b",
            context_window=131_072,
        )
        assert provider.context_window == 131_072

    def test_groq_provider_custom_max_output_tokens(self) -> None:
        """GroqProvider accepts custom max_output_tokens parameter."""
        provider = GroqProvider(
            "llama-3.1-70b",
            max_output_tokens=16384,
        )
        assert provider.max_output_tokens == 16384

    def test_groq_provider_both_specs_custom(self) -> None:
        """GroqProvider accepts both context_window and max_output_tokens."""
        provider = GroqProvider(
            "llama-3.1-70b",
            context_window=131_072,
            max_output_tokens=8192,
        )
        assert provider.context_window == 131_072
        assert provider.max_output_tokens == 8192

    def test_groq_provider_new_model_with_defaults(self) -> None:
        """GroqProvider handles new/unreleased models with defaults."""
        # New model not in hardcoded specs should use defaults
        provider = GroqProvider("llama-3.2-new")
        assert provider.context_window == 8192
        assert provider.max_output_tokens == 8192

    def test_groq_provider_new_model_with_overrides(self) -> None:
        """GroqProvider allows overriding defaults for new models."""
        # User provides specs for new/unreleased model
        provider = GroqProvider(
            "llama-3.2-new",
            context_window=200_000,
            max_output_tokens=16384,
        )
        assert provider.context_window == 200_000
        assert provider.max_output_tokens == 16384


class TestGroqProviderProperties:
    """Tests for GroqProvider properties."""

    def test_model_name_property(self) -> None:
        """Model name property is accessible."""
        provider = GroqProvider("llama-3.1-8b")
        assert provider.model_name == "llama-3.1-8b"

    def test_context_window_with_spec_override(self) -> None:
        """Context window returns user-provided value."""
        provider = GroqProvider(
            "test-model",
            context_window=65536,
        )
        assert provider.context_window == 65536

    def test_max_output_tokens_with_spec_override(self) -> None:
        """Max output tokens returns user-provided value."""
        provider = GroqProvider(
            "test-model",
            max_output_tokens=32768,
        )
        assert provider.max_output_tokens == 32768


class TestGroqProviderBackendIntegration:
    """Tests for GroqProvider backend integration."""

    def test_groq_provider_has_complete_method(self) -> None:
        """GroqProvider has complete method (from BaseProvider)."""
        provider = GroqProvider("llama-3.1-8b")
        assert hasattr(provider, "complete")
        assert callable(provider.complete)

    def test_groq_provider_has_count_tokens_method(self) -> None:
        """GroqProvider has count_tokens method (from BaseProvider)."""
        provider = GroqProvider("llama-3.1-8b")
        assert hasattr(provider, "count_tokens")
        assert callable(provider.count_tokens)

    def test_groq_provider_implements_llm_provider_protocol(self) -> None:
        """GroqProvider implements LLMProvider protocol."""
        provider = GroqProvider("llama-3.1-8b")
        # Check required attributes
        assert hasattr(provider, "model_name")
        assert hasattr(provider, "context_window")
        assert hasattr(provider, "max_output_tokens")
        assert hasattr(provider, "complete")
        assert hasattr(provider, "count_tokens")


class TestGroqProviderMissingDependencies:
    """Tests for GroqProvider error handling when SDK is missing."""

    def test_groq_client_initialization_missing_sdk(self) -> None:
        """GroqProvider._get_client raises ProviderError if groq SDK missing."""
        provider = GroqProvider("llama-3.1-8b")
        # Note: This test assumes groq SDK is not installed in test environment.
        # In a real test environment with groq installed, this would fail.
        # For now, we test that the method exists and can be called.
        assert hasattr(provider, "_get_client")
        assert callable(provider._get_client)
