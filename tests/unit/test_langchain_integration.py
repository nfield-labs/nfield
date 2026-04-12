"""Tests for formatshield.integrations.langchain.FormatShieldLLM."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from formatshield.integrations.langchain import FormatShieldLLM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_result(output: str = "mocked output") -> MagicMock:
    result = MagicMock()
    result.output = output
    return result


def _make_llm(**kwargs) -> FormatShieldLLM:
    """Instantiate FormatShieldLLM with the backend build patched out."""
    mock_backend = MagicMock()
    with patch("formatshield.core._build_backend", return_value=mock_backend):
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            return FormatShieldLLM(model="groq/llama-3.1-70b-versatile", **kwargs)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


def test_formatshield_llm_init() -> None:
    llm = _make_llm()
    assert llm.model == "groq/llama-3.1-70b-versatile"


def test_formatshield_llm_stores_model_attribute() -> None:
    llm = _make_llm()
    assert hasattr(llm, "model")
    assert hasattr(llm, "_shield")


# ---------------------------------------------------------------------------
# invoke (sync)
# ---------------------------------------------------------------------------


def test_invoke_returns_string() -> None:
    llm = _make_llm()
    mock_result = _make_mock_result("The answer is 42")

    with patch.object(llm._shield, "generate_sync", return_value=mock_result):
        output = llm.invoke("What is 6 * 7?")

    assert output == "The answer is 42"


def test_invoke_with_dict_input_extracts_prompt() -> None:
    llm = _make_llm()
    mock_result = _make_mock_result("dict answer")

    with patch.object(llm._shield, "generate_sync", return_value=mock_result):
        output = llm.invoke({"input": "What is 2 + 2?"})

    assert output == "dict answer"


def test_invoke_with_text_key_in_dict() -> None:
    llm = _make_llm()
    mock_result = _make_mock_result("text answer")

    with patch.object(llm._shield, "generate_sync", return_value=mock_result):
        output = llm.invoke({"text": "Summarise this."})

    assert output == "text answer"


def test_invoke_with_content_key_in_dict() -> None:
    llm = _make_llm()
    mock_result = _make_mock_result("content answer")

    with patch.object(llm._shield, "generate_sync", return_value=mock_result):
        output = llm.invoke({"content": "Explain entropy."})

    assert output == "content answer"


# ---------------------------------------------------------------------------
# ainvoke (async)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ainvoke_returns_string() -> None:
    llm = _make_llm()
    mock_result = _make_mock_result("async answer")

    with patch.object(
        llm._shield,
        "generate",
        new_callable=AsyncMock,
        return_value=mock_result,
    ):
        output = await llm.ainvoke("What is the capital of France?")

    assert output == "async answer"


@pytest.mark.asyncio
async def test_ainvoke_with_dict_input() -> None:
    llm = _make_llm()
    mock_result = _make_mock_result("async dict answer")

    with patch.object(
        llm._shield,
        "generate",
        new_callable=AsyncMock,
        return_value=mock_result,
    ):
        output = await llm.ainvoke({"question": "What is AI?"})

    assert output == "async dict answer"


# ---------------------------------------------------------------------------
# stream
# ---------------------------------------------------------------------------


def test_stream_yields_strings() -> None:
    llm = _make_llm()

    event1 = MagicMock()
    event1.type = "output"
    event1.token = "Hello"

    event2 = MagicMock()
    event2.type = "output"
    event2.token = " world"

    event_complete = MagicMock()
    event_complete.type = "complete"
    event_complete.token = None

    async def _fake_stream(*args: object, **kwargs: object):  # type: ignore[return]
        for ev in [event1, event2, event_complete]:
            yield ev

    with patch.object(llm._shield, "stream", new=_fake_stream):
        tokens = list(llm.stream("Say hello"))

    assert tokens == ["Hello", " world"]


def test_stream_skips_non_output_events() -> None:
    llm = _make_llm()

    thinking_event = MagicMock()
    thinking_event.type = "thinking"
    thinking_event.token = None

    output_event = MagicMock()
    output_event.type = "output"
    output_event.token = "result"

    complete_event = MagicMock()
    complete_event.type = "complete"
    complete_event.token = None

    async def _fake_stream(*args: object, **kwargs: object):  # type: ignore[return]
        for ev in [thinking_event, output_event, complete_event]:
            yield ev

    with patch.object(llm._shield, "stream", new=_fake_stream):
        tokens = list(llm.stream("Think then answer"))

    assert tokens == ["result"]


# ---------------------------------------------------------------------------
# _extract_prompt
# ---------------------------------------------------------------------------


def test_extract_prompt_from_string() -> None:
    llm = _make_llm()
    assert llm._extract_prompt("hello world") == "hello world"


def test_extract_prompt_from_dict_input_key() -> None:
    llm = _make_llm()
    assert llm._extract_prompt({"input": "my prompt"}) == "my prompt"


def test_extract_prompt_from_dict_prompt_key() -> None:
    llm = _make_llm()
    assert llm._extract_prompt({"prompt": "ask me"}) == "ask me"


def test_extract_prompt_from_dict_fallback_joins_values() -> None:
    llm = _make_llm()
    result = llm._extract_prompt({"foo": "bar", "baz": "qux"})
    assert "bar" in result
    assert "qux" in result


def test_extract_prompt_from_non_string_non_dict() -> None:
    llm = _make_llm()
    result = llm._extract_prompt(42)  # type: ignore[arg-type]
    assert result == "42"


# ---------------------------------------------------------------------------
# __or__ pipe operator
# ---------------------------------------------------------------------------


def test_pipe_operator_without_langchain_raises_import_error() -> None:
    llm = _make_llm()

    with patch.dict("sys.modules", {"langchain_core": None, "langchain_core.runnables": None}):
        with pytest.raises(ImportError):
            llm | MagicMock()
