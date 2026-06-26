"""Tests for providers._reasoning — stripping reasoning traces from output."""

from __future__ import annotations

import sys
import types

import pytest

from nfield.providers._reasoning import (
    is_unsupported_reasoning_param_error,
    reasoning_suppression_kwargs,
    strip_reasoning,
)


def _http_error(status_code: int | None, message: str) -> Exception:
    """Build an exception mimicking the SDK's status-coded API error."""
    exc = Exception(message)
    exc.status_code = status_code  # type: ignore[attr-defined]
    return exc


class TestReasoningSuppressionKwargs:
    """Endpoint-chosen request kwargs that turn thinking off."""

    def test_hosted_endpoint_uses_reasoning_effort(self) -> None:
        for base in (None, "https://api.groq.com/openai/v1", "https://api.together.xyz/v1"):
            assert reasoning_suppression_kwargs(base) == {"reasoning_effort": "none"}

    def test_local_endpoint_uses_enable_thinking(self) -> None:
        for base in (
            "http://localhost:11434/v1",
            "http://ollama:11434/v1",
            "http://localhost:8000/v1",
        ):
            assert reasoning_suppression_kwargs(base) == {
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
            }


class TestUnsupportedReasoningParamError:
    """The narrow 400 that means 'drop the thinking-off parameter and retry'."""

    def test_reasoning_effort_400_is_unsupported(self) -> None:
        exc = _http_error(400, "reasoning_effort is not supported with this model")
        assert is_unsupported_reasoning_param_error(exc) is True

    def test_enable_thinking_400_is_unsupported(self) -> None:
        exc = _http_error(400, "enable_thinking is not a valid chat_template kwarg")
        assert is_unsupported_reasoning_param_error(exc) is True

    def test_unrelated_400_is_not_swallowed(self) -> None:
        exc = _http_error(400, "context length exceeded")
        assert is_unsupported_reasoning_param_error(exc) is False

    def test_429_is_not_unsupported(self) -> None:
        exc = _http_error(429, "reasoning_effort rate limited")
        assert is_unsupported_reasoning_param_error(exc) is False

    def test_generic_exception_is_not_unsupported(self) -> None:
        assert is_unsupported_reasoning_param_error(Exception("boom")) is False


class TestStripReasoning:
    """strip_reasoning removes <think> blocks and is a no-op otherwise."""

    def test_no_tag_returns_identical_object(self) -> None:
        text = "name = Alice\nage = 30"
        assert strip_reasoning(text) is text  # true no-op, same object

    def test_empty_string(self) -> None:
        assert strip_reasoning("") == ""

    def test_single_leading_block_stripped(self) -> None:
        assert strip_reasoning("<think>the name is Alice</think>\nname = Alice") == "name = Alice"

    def test_block_then_answer_keeps_only_answer(self) -> None:
        text = "<think>reasoning here</think>\n\nname = Alice\nage = 30"
        assert strip_reasoning(text) == "name = Alice\nage = 30"

    def test_multiple_blocks_all_removed(self) -> None:
        # Each block plus its trailing newline is consumed, leaving clean lines.
        text = "<think>one</think>\nname = Alice\n<think>two</think>\nage = 30"
        assert strip_reasoning(text) == "name = Alice\nage = 30"

    def test_equals_inside_block_does_not_leak(self) -> None:
        # A reasoning line with " = " would be parsed as a false field if kept.
        assert strip_reasoning("<think>so x = 9 here</think>\nname = Alice") == "name = Alice"

    def test_case_insensitive(self) -> None:
        assert strip_reasoning("<THINK>r</THINK>\nname = Alice") == "name = Alice"

    def test_multiline_reasoning(self) -> None:
        text = "<think>line one\nline two\nline three</think>\nname = Alice"
        assert strip_reasoning(text) == "name = Alice"

    def test_unclosed_block_left_as_is(self) -> None:
        # Truncated output (no </think>): no answer to recover, leave untouched.
        text = "<think>reasoning that never closed"
        assert strip_reasoning(text) is text

    def test_whitespace_only_after_strip(self) -> None:
        assert strip_reasoning("<think>only reasoning</think>\n   \n") == ""

    def test_answer_before_and_after_block(self) -> None:
        text = "name = Alice\n<think>mid</think>\nage = 30"
        assert strip_reasoning(text) == "name = Alice\nage = 30"


# ---------------------------------------------------------------------------
# Provider-level: _raw_complete strips reasoning before returning content
# ---------------------------------------------------------------------------

_THINK_RESPONSE = "<think>x = 9</think>\nname = Alice"
_EXPECTED = "name = Alice"


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _Recorder:
    """Records each create() call; optionally fails the first param-carrying one."""

    def __init__(self, content: str, fail_when_suppressing: Exception | None) -> None:
        self.content = content
        self.fail_when_suppressing = fail_when_suppressing
        self.calls: list[dict] = []

    def create(self, **kwargs: object) -> _FakeResponse:
        self.calls.append(dict(kwargs))
        suppressing = "reasoning_effort" in kwargs or "extra_body" in kwargs
        if self.fail_when_suppressing is not None and suppressing:
            exc, self.fail_when_suppressing = self.fail_when_suppressing, None
            raise exc
        return _FakeResponse(self.content)


class _FakeChat:
    def __init__(self, recorder: _Recorder) -> None:
        self.completions = recorder


class _FakeClient:
    def __init__(self, recorder: _Recorder, **_kwargs: object) -> None:
        self.chat = _FakeChat(recorder)


def _install_fake_sdk(
    monkeypatch,
    module_name: str,
    class_name: str,
    content: str,
    *,
    fail_when_suppressing: Exception | None = None,
) -> _Recorder:
    """Install a fake SDK; return the recorder of its create() calls."""
    recorder = _Recorder(content, fail_when_suppressing)

    def _factory(**_kwargs: object) -> _FakeClient:
        return _FakeClient(recorder)

    fake_module = types.ModuleType(module_name)
    setattr(fake_module, class_name, _factory)
    monkeypatch.setitem(sys.modules, module_name, fake_module)
    return recorder


async def _complete(provider) -> str:
    return await provider._raw_complete([{"role": "user", "content": "go"}], max_tokens=64)


class TestProviderStripsReasoning:
    """Both providers strip a <think> block from a completion."""

    @pytest.mark.asyncio
    async def test_openai_strips_reasoning(self, monkeypatch) -> None:
        from nfield.providers.openai import OpenAIProvider

        _install_fake_sdk(monkeypatch, "openai", "OpenAI", _THINK_RESPONSE)
        assert await _complete(OpenAIProvider("qwen3", api_key="x")) == _EXPECTED

    @pytest.mark.asyncio
    async def test_groq_strips_reasoning(self, monkeypatch) -> None:
        from nfield.providers.groq import GroqProvider

        _install_fake_sdk(monkeypatch, "groq", "Groq", _THINK_RESPONSE)
        assert await _complete(GroqProvider("qwen3-32b", api_key="x")) == _EXPECTED

    @pytest.mark.asyncio
    async def test_openai_non_reasoning_unchanged(self, monkeypatch) -> None:
        from nfield.providers.openai import OpenAIProvider

        _install_fake_sdk(monkeypatch, "openai", "OpenAI", _EXPECTED)
        assert await _complete(OpenAIProvider("gpt-4o-mini", api_key="x")) == _EXPECTED


def _unsupported_400() -> Exception:
    exc = Exception("`reasoning_effort` is not supported with this model")
    exc.status_code = 400  # type: ignore[attr-defined]
    return exc


class TestOpenAIReasoningModelFlag:
    """reasoning_model=True disables thinking from the first call; default sends nothing."""

    @pytest.mark.asyncio
    async def test_default_sends_nothing(self, monkeypatch) -> None:
        from nfield.providers.openai import OpenAIProvider

        rec = _install_fake_sdk(monkeypatch, "openai", "OpenAI", _EXPECTED)
        provider = OpenAIProvider("gpt-4o-mini", api_key="x")
        await _complete(provider)
        await _complete(provider)
        assert all("reasoning_effort" not in c and "extra_body" not in c for c in rec.calls)

    @pytest.mark.asyncio
    async def test_flag_suppresses_from_first_call(self, monkeypatch) -> None:
        from nfield.providers.openai import OpenAIProvider

        rec = _install_fake_sdk(monkeypatch, "openai", "OpenAI", _EXPECTED)
        provider = OpenAIProvider("qwen/qwen3.6-27b", api_key="x", reasoning_model=True)
        await _complete(provider)
        assert rec.calls[0].get("reasoning_effort") == "none"

    @pytest.mark.asyncio
    async def test_flag_local_sends_enable_thinking(self, monkeypatch) -> None:
        from nfield.providers.openai import OpenAIProvider

        rec = _install_fake_sdk(monkeypatch, "openai", "OpenAI", _EXPECTED)
        provider = OpenAIProvider(
            "qwen3:4b", api_key="x", base_url="http://localhost:11434/v1", reasoning_model=True
        )
        await _complete(provider)
        assert rec.calls[0].get("extra_body") == {
            "chat_template_kwargs": {"enable_thinking": False}
        }
        assert "reasoning_effort" not in rec.calls[0]

    @pytest.mark.asyncio
    async def test_safety_net_drops_param_on_400(self, monkeypatch) -> None:
        from nfield.providers.openai import OpenAIProvider

        rec = _install_fake_sdk(
            monkeypatch, "openai", "OpenAI", _EXPECTED, fail_when_suppressing=_unsupported_400()
        )
        provider = OpenAIProvider("qwen/qwen3.6-27b", api_key="x", reasoning_model=True)
        # Call 1 sends the param (400) then retries without; call 2 sends nothing.
        assert await _complete(provider) == _EXPECTED
        assert rec.calls[0].get("reasoning_effort") == "none"
        assert "reasoning_effort" not in rec.calls[1]
        await _complete(provider)
        assert "reasoning_effort" not in rec.calls[2]
        assert len(rec.calls) == 3

    @pytest.mark.asyncio
    async def test_unrelated_400_is_not_swallowed(self, monkeypatch) -> None:
        from nfield.exceptions import ProviderError
        from nfield.providers.openai import OpenAIProvider

        other = Exception("context length exceeded")
        other.status_code = 400  # type: ignore[attr-defined]
        _install_fake_sdk(monkeypatch, "openai", "OpenAI", _EXPECTED, fail_when_suppressing=other)
        provider = OpenAIProvider("qwen/qwen3.6-27b", api_key="x", reasoning_model=True)
        with pytest.raises(ProviderError, match="context length"):
            await _complete(provider)


class TestGroqReasoningModelFlag:
    """Groq honours reasoning_model identically to the OpenAI provider."""

    @pytest.mark.asyncio
    async def test_default_sends_nothing(self, monkeypatch) -> None:
        from nfield.providers.groq import GroqProvider

        rec = _install_fake_sdk(monkeypatch, "groq", "Groq", _EXPECTED)
        await _complete(GroqProvider("qwen3-32b", api_key="x"))
        assert "reasoning_effort" not in rec.calls[0]

    @pytest.mark.asyncio
    async def test_flag_suppresses_from_first_call(self, monkeypatch) -> None:
        from nfield.providers.groq import GroqProvider

        rec = _install_fake_sdk(monkeypatch, "groq", "Groq", _EXPECTED)
        await _complete(GroqProvider("qwen3-32b", api_key="x", reasoning_model=True))
        assert rec.calls[0].get("reasoning_effort") == "none"

    @pytest.mark.asyncio
    async def test_flag_still_strips_think(self, monkeypatch) -> None:
        from nfield.providers.groq import GroqProvider

        _install_fake_sdk(monkeypatch, "groq", "Groq", _THINK_RESPONSE)
        provider = GroqProvider("qwen3-32b", api_key="x", reasoning_model=True)
        assert await _complete(provider) == _EXPECTED

    @pytest.mark.asyncio
    async def test_safety_net_drops_param_on_400(self, monkeypatch) -> None:
        from nfield.providers.groq import GroqProvider

        rec = _install_fake_sdk(
            monkeypatch, "groq", "Groq", _EXPECTED, fail_when_suppressing=_unsupported_400()
        )
        provider = GroqProvider("qwen3-32b", api_key="x", reasoning_model=True)
        assert await _complete(provider) == _EXPECTED
        assert rec.calls[0].get("reasoning_effort") == "none"
        assert "reasoning_effort" not in rec.calls[1]
