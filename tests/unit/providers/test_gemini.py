"""Tests for providers.gemini._provider - GeminiProvider implementation."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from nfield.exceptions import ProviderError
from nfield.providers._registry import from_model
from nfield.providers.gemini import GeminiProvider
from nfield.providers.gemini._provider import (
    _is_retryable,
    _is_thinking_unsupported,
)


class _FakeModels:
    """Stub for client.models: records the request and runs a behavior callback."""

    def __init__(self, behavior) -> None:
        self._behavior = behavior
        self.seen: dict[str, Any] | None = None

    def generate_content(self, **kwargs: Any) -> Any:
        self.seen = kwargs
        return self._behavior(kwargs)


class _FakeClient:
    def __init__(self, behavior) -> None:
        self.models = _FakeModels(behavior)


def _ok_response(_kwargs: dict[str, Any]) -> Any:
    usage = types.SimpleNamespace(prompt_token_count=42)
    return types.SimpleNamespace(text="vendor = Acme", usage_metadata=usage)


class TestGeminiInitialization:
    """Construction and default specs."""

    def test_initializes_with_model_name(self) -> None:
        assert GeminiProvider("gemini-2.5-flash").model_name == "gemini-2.5-flash"

    def test_default_context_window(self) -> None:
        assert GeminiProvider("gemini-2.5-flash").context_window == 1_048_576

    def test_default_max_output_tokens(self) -> None:
        assert GeminiProvider("gemini-2.5-flash").max_output_tokens == 65_536

    def test_custom_specs(self) -> None:
        provider = GeminiProvider(
            "gemini-2.5-pro", context_window=1_048_576, max_output_tokens=65_536
        )
        assert provider.context_window == 1_048_576
        assert provider.max_output_tokens == 65_536


class TestGeminiProtocol:
    """GeminiProvider satisfies the provider surface."""

    def test_protocol_attributes_present(self) -> None:
        provider = GeminiProvider("gemini-2.5-flash")
        for attr in ("model_name", "context_window", "max_output_tokens", "complete"):
            assert hasattr(provider, attr)


class TestGeminiCredentials:
    """api_key handling and the env-fallback convention."""

    def test_credentials_stored(self) -> None:
        provider = GeminiProvider("gemini-2.5-flash", api_key="k-secret", base_url="https://x")
        assert provider._api_key == "k-secret"
        assert provider._base_url == "https://x"

    def test_credentials_default_none(self) -> None:
        provider = GeminiProvider("gemini-2.5-flash")
        assert provider._api_key is None
        assert provider._base_url is None

    def test_api_key_not_leaked_in_repr(self) -> None:
        provider = GeminiProvider("gemini-2.5-flash", api_key="k-secret")
        assert "k-secret" not in repr(provider)


class TestGeminiMissingDependency:
    """Missing google-genai SDK raises a guiding ProviderError."""

    def test_missing_sdk_raises_provider_error(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "google", None)
        provider = GeminiProvider("gemini-2.5-flash")
        with pytest.raises(ProviderError, match="google-genai SDK not installed"):
            provider._get_client()


class TestGeminiRawComplete:
    """The completion call converts messages, reads usage, and returns text."""

    @pytest.mark.asyncio
    async def test_returns_text_and_records_prompt_tokens(self) -> None:
        provider = GeminiProvider("gemini-2.5-flash", api_key="k")
        provider._client = _FakeClient(_ok_response)
        messages = [
            {"role": "system", "content": "Extract fields."},
            {"role": "user", "content": "INVOICE Vendor: Acme"},
        ]
        text = await provider._raw_complete(messages, max_tokens=128)
        assert text == "vendor = Acme"
        assert provider.last_prompt_tokens == 42

    @pytest.mark.asyncio
    async def test_system_message_becomes_system_instruction(self) -> None:
        provider = GeminiProvider("gemini-2.5-flash", api_key="k")
        client = _FakeClient(_ok_response)
        provider._client = client
        await provider._raw_complete(
            [{"role": "system", "content": "sys text"}, {"role": "user", "content": "hi"}],
            max_tokens=64,
        )
        request = client.models.seen
        assert request is not None
        # System text is routed to the config, not the content turns.
        assert request["config"].system_instruction == "sys text"
        assert request["config"].max_output_tokens == 64
        assert len(request["contents"]) == 1  # only the user turn

    @pytest.mark.asyncio
    async def test_api_error_wrapped_with_status(self) -> None:
        def _raise(_kwargs: dict[str, Any]) -> Any:
            exc = RuntimeError("rate limited")
            exc.code = 429  # type: ignore[attr-defined]
            raise exc

        provider = GeminiProvider("gemini-2.5-flash", api_key="k")
        provider._client = _FakeClient(_raise)
        with pytest.raises(ProviderError) as info:
            await provider._raw_complete([{"role": "user", "content": "x"}], max_tokens=32)
        assert info.value.status_code == 429
        assert info.value.retryable is True

    @pytest.mark.asyncio
    async def test_blocked_response_raises_provider_error(self) -> None:
        class _Blocked:
            usage_metadata = types.SimpleNamespace(prompt_token_count=5)

            @property
            def text(self) -> str:
                raise ValueError("response blocked")

        provider = GeminiProvider("gemini-2.5-flash", api_key="k")
        provider._client = _FakeClient(lambda _k: _Blocked())
        with pytest.raises(ProviderError, match="no usable text"):
            await provider._raw_complete([{"role": "user", "content": "x"}], max_tokens=32)

    @pytest.mark.asyncio
    async def test_thinking_unsupported_retries_without_it(self) -> None:
        calls: list[Any] = []

        def behavior(kwargs: dict[str, Any]) -> Any:
            calls.append(kwargs)
            if len(calls) == 1:
                exc = RuntimeError("thinking_config is not supported")
                exc.code = 400  # type: ignore[attr-defined]
                raise exc
            return _ok_response(kwargs)

        provider = GeminiProvider("gemini-2.5-flash", api_key="k", reasoning_model=True)
        provider._client = _FakeClient(behavior)
        text = await provider._raw_complete([{"role": "user", "content": "x"}], max_tokens=32)
        assert text == "vendor = Acme"
        assert provider._thinking_unsupported is True
        assert len(calls) == 2
        assert calls[0]["config"].thinking_config is not None
        assert calls[1]["config"].thinking_config is None


class TestGeminiClassifiers:
    """Retry and thinking-support classification helpers."""

    def test_retryable_status_is_retryable(self) -> None:
        assert _is_retryable(Exception("x"), 503) is True

    def test_timeout_keyword_is_retryable(self) -> None:
        assert _is_retryable(Exception("connection timed out"), None) is True

    def test_unknown_defers_to_none(self) -> None:
        assert _is_retryable(Exception("bad request"), 400) is None

    def test_thinking_unsupported_detected(self) -> None:
        exc = RuntimeError("thinking_config is not supported")
        exc.code = 400  # type: ignore[attr-defined]
        assert _is_thinking_unsupported(exc) is True

    def test_non_thinking_400_not_flagged(self) -> None:
        exc = RuntimeError("invalid argument")
        exc.code = 400  # type: ignore[attr-defined]
        assert _is_thinking_unsupported(exc) is False


class TestGeminiRouting:
    """The factory routes the google/ prefix to GeminiProvider."""

    def test_google_prefix_routes_to_gemini_provider(self) -> None:
        provider = from_model("google/gemini-2.5-flash")
        assert isinstance(provider, GeminiProvider)
        assert provider.model_name == "gemini-2.5-flash"

    def test_api_key_forwarded_through_factory(self) -> None:
        provider = from_model("google/gemini-2.5-pro", api_key="k-explicit")
        assert isinstance(provider, GeminiProvider)
        assert provider._api_key == "k-explicit"
