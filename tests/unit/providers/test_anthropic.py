"""Tests for providers.anthropic._provider - AnthropicProvider implementation."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from nfield.exceptions import ProviderError
from nfield.providers._registry import from_model
from nfield.providers.anthropic import AnthropicProvider
from nfield.providers.anthropic._provider import _is_transient_error, _retry_after_seconds


class _FakeMessages:
    """Stub for client.messages: records the request and runs a behavior callback."""

    def __init__(self, behavior) -> None:
        self._behavior = behavior
        self.seen: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> Any:
        self.seen = kwargs
        return self._behavior(kwargs)


class _FakeClient:
    def __init__(self, behavior) -> None:
        self.messages = _FakeMessages(behavior)


def _ok_response(_kwargs: dict[str, Any]) -> Any:
    block = types.SimpleNamespace(type="text", text="vendor = Acme")
    usage = types.SimpleNamespace(input_tokens=37)
    return types.SimpleNamespace(content=[block], usage=usage)


def _install_fake_anthropic(monkeypatch) -> dict:
    """Replace the anthropic SDK with a fake whose Anthropic() records kwargs."""
    captured: dict = {}

    class _FakeAnthropic:
        def __init__(self, *, api_key=None, base_url=None, max_retries=None) -> None:
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            captured["max_retries"] = max_retries

    fake_module = types.ModuleType("anthropic")
    fake_module.Anthropic = _FakeAnthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    return captured


class TestAnthropicInitialization:
    """Construction and default specs."""

    def test_initializes_with_model_name(self) -> None:
        assert AnthropicProvider("claude-sonnet-4").model_name == "claude-sonnet-4"

    def test_default_context_window(self) -> None:
        assert AnthropicProvider("claude-sonnet-4").context_window == 200_000

    def test_default_max_output_tokens(self) -> None:
        assert AnthropicProvider("claude-sonnet-4").max_output_tokens == 64_000

    def test_custom_specs(self) -> None:
        provider = AnthropicProvider(
            "claude-opus-4", context_window=200_000, max_output_tokens=64_000
        )
        assert provider.context_window == 200_000
        assert provider.max_output_tokens == 64_000


class TestAnthropicCredentials:
    """api_key / base_url handling and the env-fallback convention."""

    def test_credentials_stored(self) -> None:
        provider = AnthropicProvider("claude-sonnet-4", api_key="sk-ant", base_url="https://x")
        assert provider._api_key == "sk-ant"
        assert provider._base_url == "https://x"

    def test_credentials_default_none(self) -> None:
        provider = AnthropicProvider("claude-sonnet-4")
        assert provider._api_key is None
        assert provider._base_url is None

    def test_api_key_not_leaked_in_repr(self) -> None:
        provider = AnthropicProvider("claude-sonnet-4", api_key="sk-ant-secret")
        assert "sk-ant-secret" not in repr(provider)

    def test_get_client_forwards_credentials(self, monkeypatch) -> None:
        captured = _install_fake_anthropic(monkeypatch)
        AnthropicProvider(
            "claude-sonnet-4", api_key="sk-ant", base_url="https://proxy"
        )._get_client()
        assert captured == {"api_key": "sk-ant", "base_url": "https://proxy", "max_retries": 0}

    def test_get_client_omits_base_url_when_none(self, monkeypatch) -> None:
        captured = _install_fake_anthropic(monkeypatch)
        AnthropicProvider("claude-sonnet-4", api_key="sk-ant")._get_client()
        assert captured == {"api_key": "sk-ant", "base_url": None, "max_retries": 0}


class TestAnthropicMissingDependency:
    """Missing anthropic SDK raises a guiding ProviderError."""

    def test_missing_sdk_raises_provider_error(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "anthropic", None)
        provider = AnthropicProvider("claude-sonnet-4")
        with pytest.raises(ProviderError, match="anthropic SDK not installed"):
            provider._get_client()


class TestAnthropicRawComplete:
    """The completion call maps system, reads usage, and returns joined text."""

    @pytest.mark.asyncio
    async def test_returns_text_and_records_prompt_tokens(self) -> None:
        provider = AnthropicProvider("claude-sonnet-4", api_key="k")
        provider._client = _FakeClient(_ok_response)
        text = await provider._raw_complete(
            [{"role": "system", "content": "Extract."}, {"role": "user", "content": "INVOICE"}],
            max_tokens=128,
        )
        assert text == "vendor = Acme"
        assert provider.last_prompt_tokens == 37

    @pytest.mark.asyncio
    async def test_system_message_becomes_top_level_system(self) -> None:
        provider = AnthropicProvider("claude-sonnet-4", api_key="k")
        client = _FakeClient(_ok_response)
        provider._client = client
        await provider._raw_complete(
            [{"role": "system", "content": "sys text"}, {"role": "user", "content": "hi"}],
            max_tokens=64,
        )
        request = client.messages.seen
        assert request is not None
        # System text is the top-level parameter; it is not a message turn.
        assert request["system"] == "sys text"
        assert request["max_tokens"] == 64
        assert request["messages"] == [{"role": "user", "content": "hi"}]

    @pytest.mark.asyncio
    async def test_api_error_wrapped_with_status(self) -> None:
        def _raise(_kwargs: dict[str, Any]) -> Any:
            exc = RuntimeError("overloaded")
            exc.status_code = 529  # type: ignore[attr-defined]
            raise exc

        provider = AnthropicProvider("claude-sonnet-4", api_key="k")
        provider._client = _FakeClient(_raise)
        with pytest.raises(ProviderError) as info:
            await provider._raw_complete([{"role": "user", "content": "x"}], max_tokens=32)
        assert info.value.status_code == 529

    @pytest.mark.asyncio
    async def test_joins_text_blocks_and_skips_non_text(self) -> None:
        def behavior(_kwargs: dict[str, Any]) -> Any:
            blocks = [
                types.SimpleNamespace(type="text", text="vendor = Acme"),
                types.SimpleNamespace(type="thinking", thinking="reasoning, no text attr"),
                types.SimpleNamespace(type="text", text=" | total = 10"),
            ]
            return types.SimpleNamespace(
                content=blocks, usage=types.SimpleNamespace(input_tokens=5)
            )

        provider = AnthropicProvider("claude-sonnet-4", api_key="k")
        provider._client = _FakeClient(behavior)
        text = await provider._raw_complete([{"role": "user", "content": "x"}], max_tokens=32)
        assert text == "vendor = Acme | total = 10"


class TestAnthropicClassifiers:
    """Transient-error and retry-after classification helpers."""

    def test_named_transient_error_is_retryable(self) -> None:
        exc = type("APITimeoutError", (Exception,), {})()
        assert _is_transient_error(exc) is True

    def test_keyword_transient_error_is_retryable(self) -> None:
        assert _is_transient_error(Exception("connection reset")) is True

    def test_unknown_error_defers(self) -> None:
        assert _is_transient_error(Exception("bad request")) is None

    def test_retry_after_parsed(self) -> None:
        exc = type("E", (Exception,), {})()
        exc.response = type("R", (), {"headers": {"retry-after": "4"}})()  # type: ignore[attr-defined]
        assert _retry_after_seconds(exc) == 4.0

    def test_retry_after_absent(self) -> None:
        assert _retry_after_seconds(Exception("no headers")) is None


class TestAnthropicRouting:
    """The factory routes the anthropic/ prefix to AnthropicProvider."""

    def test_anthropic_prefix_routes(self) -> None:
        provider = from_model("anthropic/claude-sonnet-4")
        assert isinstance(provider, AnthropicProvider)
        assert provider.model_name == "claude-sonnet-4"

    def test_api_key_forwarded(self) -> None:
        provider = from_model("anthropic/claude-opus-4", api_key="sk-ant-x")
        assert isinstance(provider, AnthropicProvider)
        assert provider._api_key == "sk-ant-x"
