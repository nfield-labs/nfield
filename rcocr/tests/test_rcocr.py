"""Tests for the RCOCR standalone package."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rcocr import RCOCREngine, __version__
from rcocr.engine import (
    _build_format_prompt,
    _build_think_prompt,
    _extract_thinking,
    _self_consistency_pass1,
)
from rcocr.protocol import RCOCRBackend

# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------


class MockRCOCRBackend:
    name = "mock"

    async def generate(
        self,
        prompt: str,
        constraints: str | None = None,
        **kwargs: Any,
    ) -> str:
        if constraints == "json":
            return json.dumps({"result": "extracted", "value": 42})
        return "<think>I need to analyse the prompt carefully and extract the fields.</think>"


# ---------------------------------------------------------------------------
# Module-level tests
# ---------------------------------------------------------------------------


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_engine_importable() -> None:
    assert RCOCREngine is not None


def test_backend_protocol_satisfied() -> None:
    backend = MockRCOCRBackend()
    assert isinstance(backend, RCOCRBackend)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


class TestBuildThinkPrompt:
    def test_contains_original_prompt(self) -> None:
        result = _build_think_prompt("Extract order details.")
        assert "Extract order details." in result

    def test_contains_think_instruction(self) -> None:
        result = _build_think_prompt("test prompt")
        assert "think" in result.lower()

    def test_no_json_in_think_instruction(self) -> None:
        result = _build_think_prompt("test")
        assert "Do NOT produce any JSON" in result


class TestBuildFormatPrompt:
    def test_contains_thinking(self) -> None:
        result = _build_format_prompt("prompt", "my reasoning here", None)
        assert "my reasoning here" in result

    def test_contains_original_prompt(self) -> None:
        result = _build_format_prompt("Extract order.", "thinking", None)
        assert "Extract order." in result

    def test_schema_embedded_when_provided(self) -> None:
        schema = {"type": "object", "properties": {"id": {"type": "string"}}}
        result = _build_format_prompt("prompt", "thinking", schema)
        assert "id" in result
        assert "string" in result

    def test_no_schema_omits_schema_block(self) -> None:
        result = _build_format_prompt("prompt", "thinking", None)
        # No schema JSON → schema field names won't appear
        assert "properties" not in result

    def test_format_instruction_present(self) -> None:
        result = _build_format_prompt("prompt", "thinking", None)
        assert "JSON" in result


class TestExtractThinking:
    def test_extracts_think_tags(self) -> None:
        raw = "<think>Step 1: analyse. Step 2: conclude.</think> done"
        assert _extract_thinking(raw) == "Step 1: analyse. Step 2: conclude."

    def test_empty_string_returns_empty(self) -> None:
        assert _extract_thinking("") == ""

    def test_no_tags_falls_back_to_pre_brace(self) -> None:
        raw = "some preamble here { 'key': 'val' }"
        result = _extract_thinking(raw)
        assert "preamble" in result

    def test_no_tags_no_brace_returns_stripped(self) -> None:
        raw = "  plain text  "
        assert _extract_thinking(raw) == "plain text"

    def test_multiline_thinking(self) -> None:
        raw = "<think>\nline1\nline2\n</think>"
        result = _extract_thinking(raw)
        assert "line1" in result
        assert "line2" in result


# ---------------------------------------------------------------------------
# Self-consistency pass
# ---------------------------------------------------------------------------


class TestSelfConsistencyPass1:
    @pytest.mark.asyncio
    async def test_k1_single_call(self) -> None:
        backend = MockRCOCRBackend()
        thinking, raw = await _self_consistency_pass1(backend, "test prompt", k=1)
        assert isinstance(thinking, str)
        assert isinstance(raw, str)

    @pytest.mark.asyncio
    async def test_k3_returns_best_trace(self) -> None:
        call_count = {"n": 0}

        class VariableBackend:
            name = "variable"

            async def generate(
                self, prompt: str, constraints: str | None = None, **kwargs: Any
            ) -> str:
                call_count["n"] += 1
                n = call_count["n"]
                # Return traces of different lengths
                if n == 1:
                    return "<think>short</think>"
                if n == 2:
                    return "<think>much longer reasoning with many details about the fields</think>"
                return "<think>medium length trace here</think>"

        backend = VariableBackend()
        thinking, _ = await _self_consistency_pass1(backend, "prompt", k=3)
        # Should pick the longest trace
        assert "much longer" in thinking

    @pytest.mark.asyncio
    async def test_k0_treated_as_k1(self) -> None:
        backend = MockRCOCRBackend()
        thinking, _raw = await _self_consistency_pass1(backend, "test", k=0)
        assert isinstance(thinking, str)


# ---------------------------------------------------------------------------
# RCOCREngine
# ---------------------------------------------------------------------------


class TestRCOCREngine:
    @pytest.mark.asyncio
    async def test_generate_returns_tuple(self) -> None:
        engine = RCOCREngine(backend=MockRCOCRBackend())
        result = await engine.generate("Extract data.")
        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_generate_thinking_is_string(self) -> None:
        engine = RCOCREngine(backend=MockRCOCRBackend())
        thinking, _ = await engine.generate("Extract data.")
        assert isinstance(thinking, str)

    @pytest.mark.asyncio
    async def test_generate_output_is_string(self) -> None:
        engine = RCOCREngine(backend=MockRCOCRBackend())
        _, output = await engine.generate("Extract data.")
        assert isinstance(output, str)

    @pytest.mark.asyncio
    async def test_generate_with_schema(self) -> None:
        engine = RCOCREngine(backend=MockRCOCRBackend())
        schema = {"type": "object", "properties": {"result": {"type": "string"}}}
        _, output = await engine.generate("Extract.", schema=schema)
        parsed = json.loads(output)
        assert "result" in parsed

    @pytest.mark.asyncio
    async def test_pass1_gets_constraints_none(self) -> None:
        captured: dict[str, Any] = {}

        class CapBackend:
            name = "cap"

            async def generate(
                self, prompt: str, constraints: str | None = None, **kwargs: Any
            ) -> str:
                if constraints is None:
                    captured["pass1_constraints"] = constraints
                    return "<think>reasoning</think>"
                return '{"ok": true}'

        engine = RCOCREngine(backend=CapBackend())
        await engine.generate("test")
        assert captured.get("pass1_constraints") is None

    @pytest.mark.asyncio
    async def test_pass2_gets_constraints_json(self) -> None:
        captured: dict[str, Any] = {}

        class CapBackend:
            name = "cap"

            async def generate(
                self, prompt: str, constraints: str | None = None, **kwargs: Any
            ) -> str:
                if constraints == "json":
                    captured["pass2_constraints"] = constraints
                    return '{"ok": true}'
                return "<think>reasoning</think>"

        engine = RCOCREngine(backend=CapBackend())
        await engine.generate("test")
        assert captured.get("pass2_constraints") == "json"

    @pytest.mark.asyncio
    async def test_self_consistency_k3(self) -> None:
        call_count = {"n": 0}

        class CountingBackend:
            name = "counting"

            async def generate(
                self, prompt: str, constraints: str | None = None, **kwargs: Any
            ) -> str:
                if constraints is None:
                    call_count["n"] += 1
                    return "<think>reasoning trace</think>"
                return '{"result": "done"}'

        engine = RCOCREngine(backend=CountingBackend(), self_consistency_k=3)
        await engine.generate("test")
        assert call_count["n"] == 3

    @pytest.mark.asyncio
    async def test_self_consistency_k1_single_pass(self) -> None:
        call_count = {"n": 0}

        class CountingBackend:
            name = "counting"

            async def generate(
                self, prompt: str, constraints: str | None = None, **kwargs: Any
            ) -> str:
                if constraints is None:
                    call_count["n"] += 1
                    return "<think>reasoning</think>"
                return '{"result": "done"}'

        engine = RCOCREngine(backend=CountingBackend(), self_consistency_k=1)
        await engine.generate("test")
        assert call_count["n"] == 1

    @pytest.mark.asyncio
    async def test_temperature_forwarded(self) -> None:
        captured: dict[str, Any] = {}

        class TempBackend:
            name = "temp"

            async def generate(
                self, prompt: str, constraints: str | None = None, **kwargs: Any
            ) -> str:
                captured["temperature"] = kwargs.get("temperature")
                if constraints == "json":
                    return '{"x": 1}'
                return "<think>ok</think>"

        engine = RCOCREngine(backend=TempBackend())
        await engine.generate("test", temperature=0.7)
        assert captured.get("temperature") == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_engine_default_k_is_1(self) -> None:
        engine = RCOCREngine(backend=MockRCOCRBackend())
        assert engine._k == 1

    def test_engine_clamps_k_min_to_1(self) -> None:
        engine = RCOCREngine(backend=MockRCOCRBackend(), self_consistency_k=0)
        assert engine._k == 1
