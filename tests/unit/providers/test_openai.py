"""Tests for providers.openai._provider — OpenAIProvider implementation."""

from __future__ import annotations

import sys
import types

import pytest

from nfield.exceptions import ProviderError
from nfield.providers._registry import from_model
from nfield.providers.openai import OpenAIProvider
from nfield.providers.openai._provider import _is_transient_error, _retry_after_seconds


def _install_fake_openai(monkeypatch) -> dict:
    """Replace the openai SDK with a fake whose OpenAI() records its kwargs."""
    captured: dict = {}

    class _FakeOpenAI:
        def __init__(self, *, api_key=None, base_url=None) -> None:
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    return captured


class TestOpenAIProviderInitialization:
    """Tests for OpenAIProvider initialization."""

    def test_initializes_with_model_name(self) -> None:
        provider = OpenAIProvider("gpt-4o-mini")
        assert provider.model_name == "gpt-4o-mini"

    def test_default_context_window(self) -> None:
        assert OpenAIProvider("gpt-4o-mini").context_window == 8192

    def test_default_max_output_tokens(self) -> None:
        assert OpenAIProvider("gpt-4o-mini").max_output_tokens == 8192

    def test_custom_context_window(self) -> None:
        provider = OpenAIProvider("gpt-4o", context_window=128_000)
        assert provider.context_window == 128_000

    def test_custom_max_output_tokens(self) -> None:
        provider = OpenAIProvider("gpt-4o", max_output_tokens=16_384)
        assert provider.max_output_tokens == 16_384

    def test_both_specs_custom(self) -> None:
        provider = OpenAIProvider("gpt-4o", context_window=128_000, max_output_tokens=16_384)
        assert provider.context_window == 128_000
        assert provider.max_output_tokens == 16_384

    def test_unknown_model_uses_defaults(self) -> None:
        provider = OpenAIProvider("some-new-model")
        assert provider.context_window == 8192
        assert provider.max_output_tokens == 8192


class TestOpenAIProviderProtocol:
    """OpenAIProvider satisfies the LLMProvider protocol surface."""

    def test_has_complete(self) -> None:
        provider = OpenAIProvider("gpt-4o-mini")
        assert callable(provider.complete)

    def test_protocol_attributes_present(self) -> None:
        provider = OpenAIProvider("gpt-4o-mini")
        for attr in ("model_name", "context_window", "max_output_tokens"):
            assert hasattr(provider, attr)


class TestOpenAIProviderCredentials:
    """api_key / base_url handling and the env-fallback convention."""

    def test_credentials_stored(self) -> None:
        provider = OpenAIProvider("gpt-4o", api_key="sk-secret", base_url="https://proxy/v1")
        assert provider._api_key == "sk-secret"
        assert provider._base_url == "https://proxy/v1"

    def test_credentials_default_none(self) -> None:
        provider = OpenAIProvider("gpt-4o")
        assert provider._api_key is None
        assert provider._base_url is None

    def test_api_key_not_leaked_in_repr(self) -> None:
        provider = OpenAIProvider("gpt-4o", api_key="sk-secret")
        assert "sk-secret" not in repr(provider)

    def test_get_client_forwards_credentials(self, monkeypatch) -> None:
        captured = _install_fake_openai(monkeypatch)
        provider = OpenAIProvider("gpt-4o", api_key="sk-x", base_url="http://localhost:8000/v1")
        provider._get_client()
        assert captured == {"api_key": "sk-x", "base_url": "http://localhost:8000/v1"}

    def test_get_client_passes_none_for_env_fallback(self, monkeypatch) -> None:
        captured = _install_fake_openai(monkeypatch)
        OpenAIProvider("gpt-4o")._get_client()
        # None for both → the SDK falls back to OPENAI_API_KEY env + default URL.
        assert captured == {"api_key": None, "base_url": None}

    def test_get_client_is_cached(self, monkeypatch) -> None:
        _install_fake_openai(monkeypatch)
        provider = OpenAIProvider("gpt-4o")
        assert provider._get_client() is provider._get_client()


class TestOpenAIProviderMissingDependency:
    """Missing openai SDK raises a guiding ProviderError."""

    def test_missing_sdk_raises_provider_error(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "openai", None)
        provider = OpenAIProvider("gpt-4o")
        with pytest.raises(ProviderError, match="openai SDK not installed"):
            provider._get_client()


class TestTransientErrorClassification:
    """Transient-vs-permanent classification for retry decisions."""

    def test_named_transient_error_is_retryable(self) -> None:
        exc = type("APITimeoutError", (Exception,), {})()
        assert _is_transient_error(exc) is True

    def test_keyword_transient_error_is_retryable(self) -> None:
        assert _is_transient_error(Exception("connection reset")) is True

    def test_unknown_error_defers_to_status(self) -> None:
        assert _is_transient_error(Exception("bad request")) is None

    def test_retry_after_parsed_from_headers(self) -> None:
        exc = type("E", (Exception,), {})()
        exc.response = type("R", (), {"headers": {"retry-after": "3"}})()  # type: ignore[attr-defined]
        assert _retry_after_seconds(exc) == 3.0

    def test_retry_after_absent_returns_none(self) -> None:
        assert _retry_after_seconds(Exception("no headers")) is None

    def test_retry_after_non_numeric_returns_none(self) -> None:
        exc = type("E", (Exception,), {})()
        exc.response = type("R", (), {"headers": {"retry-after": "Wed, 21 Oct"}})()  # type: ignore[attr-defined]
        assert _retry_after_seconds(exc) is None


class TestRegistryRouting:
    """The factory routes the openai/ prefix to OpenAIProvider."""

    def test_openai_prefix_routes_to_openai_provider(self) -> None:
        provider = from_model("openai/gpt-4o-mini")
        assert isinstance(provider, OpenAIProvider)
        assert provider.model_name == "gpt-4o-mini"

    def test_base_url_forwarded_through_factory(self) -> None:
        provider = from_model("openai/llama3.2", base_url="http://localhost:11434/v1")
        assert isinstance(provider, OpenAIProvider)
        assert provider._base_url == "http://localhost:11434/v1"
