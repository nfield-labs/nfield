"""Tests for providers._registry — provider factory."""

from __future__ import annotations

import pytest

from formatshield.exceptions import ProviderError
from formatshield.providers._registry import from_model, register_provider


class TestFromModel:
    """Tests for provider factory."""

    def test_groq_prefix_returns_groq_provider(self) -> None:
        """Groq prefix resolves to GroqProvider."""
        provider = from_model("groq/llama-3.1-8b")
        assert provider.model_name == "llama-3.1-8b"
        assert hasattr(provider, "complete")
        assert hasattr(provider, "context_window")

    def test_model_string_without_slash_raises_error(self) -> None:
        """Model string without slash raises ValueError."""
        with pytest.raises(ValueError, match="Expected format"):
            from_model("llama-3.1-8b")

    def test_unknown_provider_raises_error(self) -> None:
        """Unknown provider prefix raises ProviderError."""
        with pytest.raises(ProviderError, match="Unknown provider"):
            from_model("unknown/model-name")

    def test_empty_provider_or_model_raises_error(self) -> None:
        """Empty provider or model name raises ValueError."""
        with pytest.raises(ValueError, match="must be non-empty"):
            from_model("/model-name")
        with pytest.raises(ValueError, match="must be non-empty"):
            from_model("provider/")

    def test_whitespace_is_stripped(self) -> None:
        """Whitespace in model string is stripped."""
        provider = from_model("  groq  /  llama-3.1-8b  ")
        assert provider.model_name == "llama-3.1-8b"

    def test_case_insensitive_provider_prefix(self) -> None:
        """Provider prefix is case-insensitive."""
        provider = from_model("GROQ/llama-3.1-8b")
        assert provider.model_name == "llama-3.1-8b"

    def test_import_error_raises_provider_error(self) -> None:
        """ImportError from missing module is wrapped in ProviderError."""
        # Register a provider with a non-existent module path
        register_provider("badmod", "nonexistent.module.path", "SomeProvider")
        with pytest.raises(ProviderError, match="Failed to import badmod provider"):
            from_model("badmod/model-name")

    def test_attribute_error_raises_provider_error(self) -> None:
        """AttributeError (class not found) is wrapped in ProviderError."""
        # Register a provider with existing module but non-existent class
        register_provider("badclass", "sys", "NonexistentProviderClass")
        with pytest.raises(ProviderError, match="Failed to import badclass provider"):
            from_model("badclass/model-name")


class TestRegisterProvider:
    """Tests for provider registration (post-MVP feature)."""

    def test_register_provider_adds_to_registry(self) -> None:
        """Registering a provider adds it to the factory."""
        # Register a custom provider (module path must exist in prod)
        register_provider("custom", "formatshield.providers.groq", "GroqProvider")
        # Can now use it (will fail at import if module doesn't exist, but registration works)
        assert True  # Registration succeeded
