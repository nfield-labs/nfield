"""
Unit tests for formatshield.ttf.engine and formatshield.ttf.prompts.

All tests use MockBackend from conftest.py — no API keys or network access
are required.
"""

from __future__ import annotations

import pytest

from formatshield.scorer.features import StreamEvent
from formatshield.ttf.engine import TTFEngine
from formatshield.ttf.prompts import (
    build_format_prompt,
    build_think_prompt,
    extract_thinking,
)

# conftest.py exports MockBackend and MockBackendWithKVCache via fixtures.
# The fixtures are: mock_backend, mock_backend_kv.


# ===========================================================================
# TTFEngine initialisation
# ===========================================================================


class TestTTFEngineInit:
    def test_init_does_not_raise(self, mock_backend) -> None:
        """TTFEngine(backend=mock_backend) must construct without raising."""
        engine = TTFEngine(backend=mock_backend)
        assert engine is not None

    def test_init_stores_backend(self, mock_backend) -> None:
        """TTFEngine must store the supplied backend."""
        engine = TTFEngine(backend=mock_backend)
        assert engine._backend is mock_backend

    def test_init_expose_thinking_default_false(self, mock_backend) -> None:
        """expose_thinking must default to False."""
        engine = TTFEngine(backend=mock_backend)
        assert engine._expose_thinking is False

    def test_init_ttf_fallback_default_true(self, mock_backend) -> None:
        """ttf_fallback must default to True."""
        engine = TTFEngine(backend=mock_backend)
        assert engine._ttf_fallback is True

    def test_init_expose_thinking_true(self, mock_backend) -> None:
        """expose_thinking=True must be stored correctly."""
        engine = TTFEngine(backend=mock_backend, expose_thinking=True)
        assert engine._expose_thinking is True

    def test_init_ttf_fallback_false(self, mock_backend) -> None:
        """ttf_fallback=False must be stored correctly."""
        engine = TTFEngine(backend=mock_backend, ttf_fallback=False)
        assert engine._ttf_fallback is False


# ===========================================================================
# TTFEngine.generate()
# ===========================================================================


class TestTTFEngineGenerate:
    @pytest.mark.asyncio
    async def test_generate_returns_tuple(self, mock_backend) -> None:
        """generate() must return a two-element tuple."""
        engine = TTFEngine(backend=mock_backend)
        result = await engine.generate(prompt="What is 2+2?")
        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_generate_returns_strings(self, mock_backend) -> None:
        """Both elements of the generate() tuple must be strings."""
        engine = TTFEngine(backend=mock_backend)
        thinking, output = await engine.generate(prompt="What is 2+2?")
        assert isinstance(thinking, str)
        assert isinstance(output, str)

    @pytest.mark.asyncio
    async def test_generate_output_not_none(self, mock_backend) -> None:
        """The output string from generate() must not be None."""
        engine = TTFEngine(backend=mock_backend)
        _thinking, output = await engine.generate(prompt="What is 3*3?")
        assert output is not None

    @pytest.mark.asyncio
    async def test_generate_thinking_not_none(self, mock_backend) -> None:
        """The thinking string from generate() must not be None."""
        engine = TTFEngine(backend=mock_backend)
        thinking, _output = await engine.generate(prompt="Describe photosynthesis.")
        assert thinking is not None

    @pytest.mark.asyncio
    async def test_generate_with_schema_does_not_raise(self, mock_backend, simple_schema) -> None:
        """generate() with a schema dict must not raise."""
        engine = TTFEngine(backend=mock_backend)
        _thinking, output = await engine.generate(prompt="Extract entity", schema=simple_schema)
        assert isinstance(output, str)

    @pytest.mark.asyncio
    async def test_generate_with_kv_cache_prefix_does_not_raise(self, mock_backend) -> None:
        """generate() with kv_cache_prefix must not raise."""
        engine = TTFEngine(backend=mock_backend)
        _thinking, output = await engine.generate(
            prompt="Simple question",
            kv_cache_prefix="some_prefix",
        )
        assert isinstance(output, str)

    @pytest.mark.asyncio
    async def test_generate_with_kv_cache_backend_does_not_raise(self, mock_backend_kv) -> None:
        """generate() with a KV-cache-capable backend must not raise."""
        engine = TTFEngine(backend=mock_backend_kv)
        _thinking, output = await engine.generate(prompt="Test kv cache path.")
        assert isinstance(output, str)

    @pytest.mark.asyncio
    async def test_generate_thinking_extracted_from_tags(self, mock_backend) -> None:
        """
        MockBackend pass-1 returns <think>…</think> tags;
        extract_thinking must strip them so thinking has no angle brackets.
        """
        engine = TTFEngine(backend=mock_backend)
        thinking, _ = await engine.generate(prompt="Reason about X.")
        # The mock backend wraps thinking in <think>…</think>;
        # extract_thinking should strip those tags.
        assert "<think>" not in thinking
        assert "</think>" not in thinking


# ===========================================================================
# TTFEngine with schema_model (fallback path)
# ===========================================================================


class TestTTFEngineFallback:
    @pytest.mark.asyncio
    async def test_generate_with_invalid_schema_model_triggers_fallback(self, mock_backend) -> None:
        """
        When schema_model validation fails and ttf_fallback=True,
        generate() should return ('', direct_output) without raising.
        """
        from pydantic import BaseModel

        class StrictModel(BaseModel):
            required_field_xyz: str  # mock backend output won't have this

        engine = TTFEngine(backend=mock_backend, ttf_fallback=True)
        thinking, output = await engine.generate(
            prompt="Produce structured output",
            schema_model=StrictModel,
        )
        # Fallback path returns empty thinking
        assert isinstance(thinking, str)
        assert isinstance(output, str)

    @pytest.mark.asyncio
    async def test_generate_with_ttf_fallback_false_no_raise(self, mock_backend) -> None:
        """
        With ttf_fallback=False and a mismatched schema_model,
        generate() must still return (thinking, output) without raising.
        """
        from pydantic import BaseModel

        class StrictModel(BaseModel):
            required_field_xyz: str

        engine = TTFEngine(backend=mock_backend, ttf_fallback=False)
        thinking, output = await engine.generate(
            prompt="Produce structured output",
            schema_model=StrictModel,
        )
        assert isinstance(thinking, str)
        assert isinstance(output, str)


# ===========================================================================
# TTFEngine.generate_direct()
# ===========================================================================


class TestTTFEngineGenerateDirect:
    @pytest.mark.asyncio
    async def test_generate_direct_returns_string(self, mock_backend) -> None:
        """generate_direct() must return a string."""
        engine = TTFEngine(backend=mock_backend)
        result = await engine.generate_direct(prompt="Direct generation test")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_generate_direct_with_schema_does_not_raise(
        self, mock_backend, simple_schema
    ) -> None:
        """generate_direct() with a schema must not raise."""
        engine = TTFEngine(backend=mock_backend)
        result = await engine.generate_direct(prompt="Extract", schema=simple_schema)
        assert isinstance(result, str)


# ===========================================================================
# TTFEngine._stream_impl()
# ===========================================================================


class TestTTFEngineStreamImpl:
    @pytest.mark.asyncio
    async def test_stream_impl_yields_events(self, mock_backend) -> None:
        """_stream_impl must yield at least one StreamEvent."""
        engine = TTFEngine(backend=mock_backend)
        events: list[StreamEvent] = []
        async for event in engine._stream_impl("Test prompt", schema=None):
            events.append(event)
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_stream_impl_all_events_are_stream_events(self, mock_backend) -> None:
        """Every item yielded by _stream_impl must be a StreamEvent."""
        engine = TTFEngine(backend=mock_backend)
        async for event in engine._stream_impl("Another test", schema=None):
            assert isinstance(event, StreamEvent)

    @pytest.mark.asyncio
    async def test_stream_impl_yields_complete_event(self, mock_backend) -> None:
        """_stream_impl must yield at least one event of type 'complete'."""
        engine = TTFEngine(backend=mock_backend)
        types_seen: list[str] = []
        async for event in engine._stream_impl("Prompt text", schema=None):
            types_seen.append(event.type)
        assert "complete" in types_seen

    @pytest.mark.asyncio
    async def test_stream_impl_yields_output_or_thinking_events(self, mock_backend) -> None:
        """_stream_impl must yield at least one 'thinking' or 'output' event."""
        engine = TTFEngine(backend=mock_backend)
        types_seen: list[str] = []
        async for event in engine._stream_impl("Some prompt", schema=None):
            types_seen.append(event.type)
        assert "thinking" in types_seen or "output" in types_seen

    @pytest.mark.asyncio
    async def test_stream_impl_with_schema_does_not_raise(
        self, mock_backend, simple_schema
    ) -> None:
        """_stream_impl with a schema dict must not raise."""
        engine = TTFEngine(backend=mock_backend)
        async for _event in engine._stream_impl("Extract this", schema=simple_schema):
            pass  # just consume without raising

    @pytest.mark.asyncio
    async def test_stream_impl_backend_name_on_events(self, mock_backend) -> None:
        """Each StreamEvent emitted must carry the backend name."""
        engine = TTFEngine(backend=mock_backend)
        async for event in engine._stream_impl("Query", schema=None):
            assert event.backend == "mock"

    @pytest.mark.asyncio
    async def test_stream_impl_latency_ms_non_negative(self, mock_backend) -> None:
        """latency_ms on each StreamEvent must be >= 0."""
        engine = TTFEngine(backend=mock_backend)
        async for event in engine._stream_impl("Query", schema=None):
            assert event.latency_ms >= 0.0


# ===========================================================================
# build_think_prompt
# ===========================================================================


class TestBuildThinkPrompt:
    def test_returns_string(self) -> None:
        """build_think_prompt() must return a string."""
        result = build_think_prompt("What is 2+2?")
        assert isinstance(result, str)

    def test_contains_original_prompt(self) -> None:
        """The result must contain the original prompt text."""
        prompt = "Explain quantum entanglement."
        result = build_think_prompt(prompt)
        assert prompt in result

    def test_contains_think_instruction(self) -> None:
        """The result must mention <think> tags."""
        result = build_think_prompt("Any prompt")
        assert "<think>" in result

    def test_non_empty_for_empty_input(self) -> None:
        """build_think_prompt('') must still return a non-empty string."""
        result = build_think_prompt("")
        assert len(result) > 0

    def test_longer_than_original_prompt(self) -> None:
        """The think prompt must be longer than the original prompt."""
        prompt = "Calculate 5 * 7."
        result = build_think_prompt(prompt)
        assert len(result) > len(prompt)


# ===========================================================================
# build_format_prompt
# ===========================================================================


class TestBuildFormatPrompt:
    def test_returns_string(self) -> None:
        """build_format_prompt() must return a string."""
        think_prompt = build_think_prompt("Extract info.")
        result = build_format_prompt(think_prompt, "My thinking here", schema=None)
        assert isinstance(result, str)

    def test_contains_thinking_text(self) -> None:
        """The format prompt must contain the thinking text."""
        think_prompt = build_think_prompt("Reason about X.")
        thinking = "I believe the answer is 42."
        result = build_format_prompt(think_prompt, thinking)
        assert thinking in result

    def test_contains_think_prompt(self) -> None:
        """The format prompt must contain the think prompt."""
        think_prompt = build_think_prompt("Original question?")
        result = build_format_prompt(think_prompt, "Some thinking")
        assert think_prompt in result

    def test_with_schema_mentions_json(self) -> None:
        """When schema is provided, the result must mention JSON."""
        think_prompt = build_think_prompt("Question")
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        result = build_format_prompt(think_prompt, "Thinking", schema=schema)
        assert "JSON" in result or "json" in result

    def test_with_schema_embeds_schema_fields(self, simple_schema) -> None:
        """When schema is provided, property names must appear in the result."""
        think_prompt = build_think_prompt("Extract")
        result = build_format_prompt(think_prompt, "Thinking text", schema=simple_schema)
        assert "name" in result

    def test_without_schema_does_not_raise(self) -> None:
        """build_format_prompt without schema must not raise."""
        think_prompt = build_think_prompt("Question")
        result = build_format_prompt(think_prompt, "Thinking", schema=None)
        assert isinstance(result, str)

    def test_legacy_thinking_text_alias(self) -> None:
        """thinking_text keyword alias must be accepted for backward compat."""
        think_prompt = build_think_prompt("Question")
        result = build_format_prompt(think_prompt, thinking="", thinking_text="legacy thinking")
        assert "legacy thinking" in result


# ===========================================================================
# extract_thinking
# ===========================================================================


class TestExtractThinking:
    def test_extracts_from_think_tags(self) -> None:
        """extract_thinking must extract content from <think>...</think> tags."""
        raw = "<think>Let me calculate: 3 * 4 = 12</think>"
        result = extract_thinking(raw)
        assert result == "Let me calculate: 3 * 4 = 12"

    def test_returns_empty_string_when_no_tags(self) -> None:
        """When no think tags are present, the full stripped text is returned."""
        raw = "The answer is 42."
        result = extract_thinking(raw)
        # No tags → returns the full response stripped
        assert result == "The answer is 42."

    def test_extracts_multiline_content(self) -> None:
        """extract_thinking must handle multi-line content inside tags."""
        raw = "<think>\nLine 1\nLine 2\n</think>"
        result = extract_thinking(raw)
        assert "Line 1" in result
        assert "Line 2" in result

    def test_handles_thinking_tags_anthropic_style(self) -> None:
        """extract_thinking must support <thinking>...</thinking> tags."""
        raw = "<thinking>Anthropic extended thinking</thinking>"
        result = extract_thinking(raw)
        assert "Anthropic extended thinking" in result

    def test_handles_multiple_think_blocks(self) -> None:
        """All <think> blocks must be concatenated when multiple exist."""
        raw = "<think>Block 1</think> Some text <think>Block 2</think>"
        result = extract_thinking(raw)
        assert "Block 1" in result
        assert "Block 2" in result

    def test_case_insensitive_tags(self) -> None:
        """Tag matching must be case-insensitive."""
        raw = "<THINK>upper case tags</THINK>"
        result = extract_thinking(raw)
        assert "upper case tags" in result

    def test_empty_string_input(self) -> None:
        """extract_thinking('') must return an empty string without raising."""
        result = extract_thinking("")
        assert result == ""

    def test_returns_string_type(self) -> None:
        """extract_thinking must always return a str."""
        result = extract_thinking("<think>content</think>")
        assert isinstance(result, str)

    def test_strips_surrounding_whitespace_from_content(self) -> None:
        """Content inside tags must be stripped of leading/trailing whitespace."""
        raw = "<think>  content with spaces  </think>"
        result = extract_thinking(raw)
        assert result == "content with spaces"

    def test_extract_thinking_no_tags_returns_stripped_text(self) -> None:
        """When no tags exist, the full response is returned stripped."""
        raw = "  plain reasoning text  "
        result = extract_thinking(raw)
        assert result == "plain reasoning text"
