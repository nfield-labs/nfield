"""Unit tests for CohereBackend, MistralBackend, and TogetherBackend.

All three are optional dependencies not installed in CI.  Tests use
``patch.dict(sys.modules, ...)`` to simulate the libraries being present,
or verify the correct ``ImportError`` / ``ValueError`` is raised when absent.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from formatshield.backends.cohere_backend import CohereBackend
from formatshield.backends.mistral_backend import MistralBackend
from formatshield.backends.together_backend import TogetherBackend

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _async_context_manager(iterable):
    """Wrap an iterable as an async context-manager that iterates async."""

    class _CM:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        def __aiter__(self):
            return _AIter(iterable)

    class _AIter:
        def __init__(self, it):
            self._it = iter(it)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._it)
            except StopIteration:
                raise StopAsyncIteration from None

    return _CM()


# ===========================================================================
# CohereBackend
# ===========================================================================


def _make_cohere_mock(text: str = "hello cohere") -> MagicMock:
    """Return a minimal ``cohere`` SDK mock."""
    mock_cohere = MagicMock()

    # cohere.AsyncClientV2(api_key=...) → client
    mock_client = MagicMock()
    mock_cohere.AsyncClientV2.return_value = mock_client

    # client.chat(**kwargs) → response  (awaitable)
    mock_content_item = MagicMock()
    mock_content_item.text = text
    mock_response = MagicMock()
    mock_response.message.content = [mock_content_item]
    mock_client.chat = AsyncMock(return_value=mock_response)

    # client.chat_stream(**kwargs) → awaitable that returns async iterable
    mock_event = MagicMock()
    mock_event.type = "content-delta"
    mock_event.delta.message.content.text = text

    class _AsyncIter:
        def __init__(self, items):
            self._items = iter(items)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._items)
            except StopIteration:
                raise StopAsyncIteration from None

    mock_client.chat_stream = AsyncMock(return_value=_AsyncIter([mock_event]))

    return mock_cohere


def _cohere_patch(mock_cohere: MagicMock):
    return patch.dict("sys.modules", {"cohere": mock_cohere})


def _remove_cohere():
    return patch.dict("sys.modules", {"cohere": None})


class TestCohereBackendInit:
    def test_default_name_and_model(self) -> None:
        with _cohere_patch(_make_cohere_mock()):
            b = CohereBackend(api_key="test-key")
        assert b.name == "cohere"
        assert b.model == "command-r-plus"

    def test_custom_model_strips_prefix(self) -> None:
        with _cohere_patch(_make_cohere_mock()):
            b = CohereBackend(api_key="test-key", model="cohere/command-r")
        assert b.model == "command-r"

    def test_supports_kv_cache_reuse_false(self) -> None:
        with _cohere_patch(_make_cohere_mock()):
            b = CohereBackend(api_key="test-key")
        assert b.supports_kv_cache_reuse is False

    def test_accuracy_loss_baseline(self) -> None:
        with _cohere_patch(_make_cohere_mock()):
            b = CohereBackend(api_key="test-key")
        assert b.accuracy_loss_baseline == pytest.approx(0.12)

    def test_missing_api_key_raises(self) -> None:
        with _cohere_patch(_make_cohere_mock()):
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(ValueError, match="COHERE_API_KEY"):
                    CohereBackend()

    def test_missing_cohere_package_raises(self) -> None:
        with _remove_cohere():
            with pytest.raises(ImportError, match="formatshield\\[cohere\\]"):
                CohereBackend(api_key="test-key")


class TestCohereBackendGenerate:
    @pytest.mark.asyncio
    async def test_generate_returns_text(self) -> None:
        mock_cohere = _make_cohere_mock(text="cohere output")
        with _cohere_patch(mock_cohere):
            b = CohereBackend(api_key="test-key")
            result = await b.generate("prompt")
        assert result == "cohere output"

    @pytest.mark.asyncio
    async def test_generate_with_json_mode(self) -> None:
        mock_cohere = _make_cohere_mock(text='{"x": 1}')
        with _cohere_patch(mock_cohere):
            b = CohereBackend(api_key="test-key")
            result = await b.generate("prompt", schema={"type": "object"}, constraints="json")
        assert result == '{"x": 1}'
        call_kwargs = mock_cohere.AsyncClientV2.return_value.chat.call_args.kwargs
        assert call_kwargs.get("response_format") == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_generate_kv_cache_prefix_ignored(self) -> None:
        mock_cohere = _make_cohere_mock(text="ok")
        with _cohere_patch(mock_cohere):
            b = CohereBackend(api_key="test-key")
            result = await b.generate("prompt", kv_cache_prefix="prefix")
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_generate_schema_in_system_prompt(self) -> None:
        """When schema supplied without json constraints, it goes in system message."""
        mock_cohere = _make_cohere_mock(text='{"answer": "yes"}')
        with _cohere_patch(mock_cohere):
            b = CohereBackend(api_key="test-key")
            await b.generate("prompt", schema={"type": "object"})
        call_kwargs = mock_cohere.AsyncClientV2.return_value.chat.call_args.kwargs
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert "JSON schema" in messages[0]["content"]


class TestCohereBackendStream:
    @pytest.mark.asyncio
    async def test_stream_yields_events(self) -> None:
        mock_cohere = _make_cohere_mock(text="streamed")
        with _cohere_patch(mock_cohere):
            b = CohereBackend(api_key="test-key")
            events = [e async for e in await b.stream("prompt")]
        output_evts = [e for e in events if e.type == "output"]
        complete_evts = [e for e in events if e.type == "complete"]
        assert len(output_evts) >= 1
        assert len(complete_evts) == 1
        assert complete_evts[0].backend == "cohere"

    @pytest.mark.asyncio
    async def test_stream_complete_event_has_content(self) -> None:
        mock_cohere = _make_cohere_mock(text="full text")
        with _cohere_patch(mock_cohere):
            b = CohereBackend(api_key="test-key")
            events = [e async for e in await b.stream("prompt")]
        complete = next(e for e in events if e.type == "complete")
        assert complete.content == "full text"


# ===========================================================================
# MistralBackend
# ===========================================================================


def _make_mistral_mock(text: str = "hello mistral") -> MagicMock:
    """Return a minimal ``mistralai`` SDK mock."""
    mock_mistralai = MagicMock()

    # Mistral(api_key=...) → client
    mock_client = MagicMock()
    mock_mistralai.Mistral.return_value = mock_client

    # client.chat.complete_async(**kwargs) → response (awaitable)
    mock_choice = MagicMock()
    mock_choice.message.content = text
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.complete_async = AsyncMock(return_value=mock_response)

    # client.chat.stream_async(**kwargs) → awaitable that returns async iterable
    mock_chunk = MagicMock()
    mock_chunk.data.choices = [MagicMock()]
    mock_chunk.data.choices[0].delta.content = text

    class _AsyncIter:
        def __init__(self, items):
            self._items = iter(items)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._items)
            except StopIteration:
                raise StopAsyncIteration from None

    mock_client.chat.stream_async = AsyncMock(return_value=_AsyncIter([mock_chunk]))

    return mock_mistralai


def _mistral_patch(mock_mistralai: MagicMock):
    return patch.dict("sys.modules", {"mistralai": mock_mistralai})


def _remove_mistral():
    return patch.dict("sys.modules", {"mistralai": None})


class TestMistralBackendInit:
    def test_default_name_and_model(self) -> None:
        with _mistral_patch(_make_mistral_mock()):
            b = MistralBackend(api_key="test-key")
        assert b.name == "mistral"
        assert b.model == "mistral-large-latest"

    def test_custom_model_strips_prefix(self) -> None:
        with _mistral_patch(_make_mistral_mock()):
            b = MistralBackend(api_key="test-key", model="mistral/mistral-medium")
        assert b.model == "mistral-medium"

    def test_supports_kv_cache_reuse_false(self) -> None:
        with _mistral_patch(_make_mistral_mock()):
            b = MistralBackend(api_key="test-key")
        assert b.supports_kv_cache_reuse is False

    def test_accuracy_loss_baseline(self) -> None:
        with _mistral_patch(_make_mistral_mock()):
            b = MistralBackend(api_key="test-key")
        assert b.accuracy_loss_baseline == pytest.approx(0.14)

    def test_missing_api_key_raises(self) -> None:
        with _mistral_patch(_make_mistral_mock()):
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
                    MistralBackend()

    def test_missing_mistralai_package_raises(self) -> None:
        with _remove_mistral():
            with pytest.raises(ImportError, match="formatshield\\[mistral\\]"):
                MistralBackend(api_key="test-key")


class TestMistralBackendGenerate:
    @pytest.mark.asyncio
    async def test_generate_returns_text(self) -> None:
        mock_mistralai = _make_mistral_mock(text="mistral output")
        with _mistral_patch(mock_mistralai):
            b = MistralBackend(api_key="test-key")
            result = await b.generate("prompt")
        assert result == "mistral output"

    @pytest.mark.asyncio
    async def test_generate_with_json_mode(self) -> None:
        mock_mistralai = _make_mistral_mock(text='{"y": 2}')
        with _mistral_patch(mock_mistralai):
            b = MistralBackend(api_key="test-key")
            result = await b.generate("prompt", schema={"type": "object"}, constraints="json")
        assert result == '{"y": 2}'
        call_kwargs = mock_mistralai.Mistral.return_value.chat.complete_async.call_args.kwargs
        assert call_kwargs.get("response_format") == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_generate_kv_cache_prefix_ignored(self) -> None:
        mock_mistralai = _make_mistral_mock(text="ok")
        with _mistral_patch(mock_mistralai):
            b = MistralBackend(api_key="test-key")
            result = await b.generate("prompt", kv_cache_prefix="some-prefix")
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_generate_schema_in_system_prompt(self) -> None:
        mock_mistralai = _make_mistral_mock(text='{"z": true}')
        with _mistral_patch(mock_mistralai):
            b = MistralBackend(api_key="test-key")
            await b.generate("prompt", schema={"type": "object"})
        call_kwargs = mock_mistralai.Mistral.return_value.chat.complete_async.call_args.kwargs
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert "JSON schema" in messages[0]["content"]


class TestMistralBackendStream:
    @pytest.mark.asyncio
    async def test_stream_yields_events(self) -> None:
        mock_mistralai = _make_mistral_mock(text="streamed")
        with _mistral_patch(mock_mistralai):
            b = MistralBackend(api_key="test-key")
            events = [e async for e in await b.stream("prompt")]
        output_evts = [e for e in events if e.type == "output"]
        complete_evts = [e for e in events if e.type == "complete"]
        assert len(output_evts) >= 1
        assert len(complete_evts) == 1
        assert complete_evts[0].backend == "mistral"

    @pytest.mark.asyncio
    async def test_stream_complete_event_has_content(self) -> None:
        mock_mistralai = _make_mistral_mock(text="full text")
        with _mistral_patch(mock_mistralai):
            b = MistralBackend(api_key="test-key")
            events = [e async for e in await b.stream("prompt")]
        complete = next(e for e in events if e.type == "complete")
        assert complete.content == "full text"


# ===========================================================================
# TogetherBackend
# ===========================================================================


def _make_openai_mock(text: str = "hello together") -> MagicMock:
    """Return a minimal ``openai`` SDK mock for TogetherBackend."""
    mock_openai = MagicMock()

    # AsyncOpenAI(api_key=..., base_url=...) → client
    mock_client = MagicMock()
    mock_openai.AsyncOpenAI.return_value = mock_client

    # client.chat.completions.create(**kwargs) → response (awaitable)
    mock_choice = MagicMock()
    mock_choice.message.content = text
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    # Streaming: create(**kwargs, stream=True) → async context manager yielding chunks
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta.content = text

    mock_stream_cm = MagicMock()
    mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_stream_cm)
    mock_stream_cm.__aexit__ = AsyncMock(return_value=False)
    mock_stream_cm.__aiter__ = MagicMock(return_value=iter([mock_chunk]))

    async def _aiter_stream():
        yield mock_chunk

    mock_stream_cm.__aiter__ = lambda self: _aiter_stream().__aiter__()
    mock_client.chat.completions.create_stream = AsyncMock(return_value=mock_stream_cm)

    # Expose exception classes on the mock
    mock_openai.RateLimitError = type("RateLimitError", (Exception,), {})
    mock_openai.InternalServerError = type("InternalServerError", (Exception,), {})
    mock_openai.APIError = type("APIError", (Exception,), {})

    return mock_openai


def _together_patch(mock_openai: MagicMock):
    return patch.dict("sys.modules", {"openai": mock_openai})


def _remove_openai():
    return patch.dict("sys.modules", {"openai": None})


class TestTogetherBackendInit:
    def test_default_name_and_model(self) -> None:
        with _together_patch(_make_openai_mock()):
            b = TogetherBackend(api_key="test-key")
        assert b.name == "together"
        assert b.model == "meta-llama/Llama-3-70b-chat-hf"

    def test_custom_model_strips_prefix(self) -> None:
        with _together_patch(_make_openai_mock()):
            b = TogetherBackend(api_key="test-key", model="together/Qwen/Qwen2-72B-Instruct")
        assert b.model == "Qwen/Qwen2-72B-Instruct"

    def test_supports_kv_cache_reuse_false(self) -> None:
        with _together_patch(_make_openai_mock()):
            b = TogetherBackend(api_key="test-key")
        assert b.supports_kv_cache_reuse is False

    def test_accuracy_loss_baseline(self) -> None:
        with _together_patch(_make_openai_mock()):
            b = TogetherBackend(api_key="test-key")
        assert b.accuracy_loss_baseline == pytest.approx(0.16)

    def test_missing_api_key_raises(self) -> None:
        with _together_patch(_make_openai_mock()):
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(ValueError, match="TOGETHER_API_KEY"):
                    TogetherBackend()

    def test_missing_openai_package_raises(self) -> None:
        with _remove_openai():
            with pytest.raises(ImportError, match="formatshield\\[together\\]"):
                TogetherBackend(api_key="test-key")


class TestTogetherBackendGenerate:
    @pytest.mark.asyncio
    async def test_generate_returns_text(self) -> None:
        mock_openai = _make_openai_mock(text="together output")
        with _together_patch(mock_openai):
            b = TogetherBackend(api_key="test-key")
            result = await b.generate("prompt")
        assert result == "together output"

    @pytest.mark.asyncio
    async def test_generate_with_json_mode(self) -> None:
        mock_openai = _make_openai_mock(text='{"a": 42}')
        with _together_patch(mock_openai):
            b = TogetherBackend(api_key="test-key")
            result = await b.generate("prompt", schema={"type": "object"}, constraints="json")
        assert result == '{"a": 42}'
        call_kwargs = mock_openai.AsyncOpenAI.return_value.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("response_format") == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_generate_kv_cache_prefix_ignored(self) -> None:
        mock_openai = _make_openai_mock(text="ok")
        with _together_patch(mock_openai):
            b = TogetherBackend(api_key="test-key")
            result = await b.generate("prompt", kv_cache_prefix="prefix")
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_generate_schema_in_system_prompt(self) -> None:
        mock_openai = _make_openai_mock(text='{"b": "val"}')
        with _together_patch(mock_openai):
            b = TogetherBackend(api_key="test-key")
            await b.generate("prompt", schema={"type": "object"})
        call_kwargs = mock_openai.AsyncOpenAI.return_value.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert "JSON schema" in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_generate_api_error_wraps_runtime_error(self) -> None:
        mock_openai = _make_openai_mock()
        api_error_cls = type("APIError", (Exception,), {})
        mock_openai.APIError = api_error_cls
        mock_openai.RateLimitError = type("RateLimitError", (Exception,), {})
        mock_openai.InternalServerError = type("InternalServerError", (Exception,), {})
        mock_openai.AsyncOpenAI.return_value.chat.completions.create = AsyncMock(
            side_effect=api_error_cls("bad request")
        )
        with _together_patch(mock_openai):
            b = TogetherBackend(api_key="test-key")
            with pytest.raises(RuntimeError, match="Together AI API error"):
                await b.generate("prompt")


# ===========================================================================
# Protocol routing tests — verify BackendName/prefix registration
# ===========================================================================


class TestBackendNameRegistration:
    def test_cohere_prefix_maps_to_cohere(self) -> None:
        from formatshield.backends.protocol import get_backend_name_from_model

        assert get_backend_name_from_model("cohere/command-r-plus") == "cohere"

    def test_mistral_prefix_maps_to_mistral(self) -> None:
        from formatshield.backends.protocol import get_backend_name_from_model

        assert get_backend_name_from_model("mistral/mistral-large-latest") == "mistral"

    def test_together_prefix_maps_to_together(self) -> None:
        from formatshield.backends.protocol import get_backend_name_from_model

        assert get_backend_name_from_model("together/meta-llama/Llama-3-70b") == "together"

    def test_openai_prefix_maps_to_openai(self) -> None:
        from formatshield.backends.protocol import get_backend_name_from_model

        assert get_backend_name_from_model("openai/gpt-4o") == "openai"

    def test_anthropic_prefix_maps_to_anthropic(self) -> None:
        from formatshield.backends.protocol import get_backend_name_from_model

        assert get_backend_name_from_model("anthropic/claude-3-5-sonnet") == "anthropic"
