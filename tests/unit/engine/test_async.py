"""Tests for AsyncFormatShield: context manager, schema caching, call forms."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from formatshield import AsyncFormatShield, nfield_async
from formatshield.config import ExtractionConfig
from formatshield.engine._async import (
    _MAX_SCHEMA_DEPTH,
    _dataclass_to_json_schema,
)
from formatshield.exceptions import SchemaError


# Module-scope dataclasses so get_type_hints can resolve the nested type — this
# is how real users define schemas (top level), and what the converter targets.
@dataclass
class _Addr:
    city: str


@dataclass
class _Person:
    home: _Addr
    work: _Addr


# Self-referential dataclass: the converter must reject this with a clean
# SchemaError instead of recursing into a RecursionError.
@dataclass
class _Tree:
    value: int
    children: list[_Tree]


_DOC = "Name: Alice. Age: 30."
_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
}
_ECHO = "name = Alice\nage = 30"


class TestAsyncEngine:
    async def test_async_context_manager(self, install_provider):
        install_provider(_ECHO)
        async with AsyncFormatShield("mock/echo", _SCHEMA) as fs:
            result = await fs.extract(_DOC)
        assert result.data["name"] == "Alice"

    async def test_call_alias(self, install_provider):
        install_provider(_ECHO)
        engine = AsyncFormatShield("mock/echo", _SCHEMA)
        result = await engine(_DOC)
        assert result.data["age"] == 30

    async def test_cached_schema_reused_across_calls(self, install_provider):
        install_provider(_ECHO)
        engine = AsyncFormatShield(
            "mock/echo", _SCHEMA, config=ExtractionConfig(max_retry_rounds=0)
        )
        first = await engine.extract("doc one")
        second = await engine.extract("doc two")
        assert first.metadata.fields_total == 2
        assert second.metadata.fields_total == 2

    async def test_per_call_schema_overrides_cached(self, install_provider):
        install_provider("name = Bob")
        engine = AsyncFormatShield("mock/echo", _SCHEMA)
        override = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = await engine.extract(_DOC, schema=override)
        assert result.metadata.fields_total == 1
        assert result.data["name"] == "Bob"

    async def test_missing_schema_raises(self, install_provider):
        install_provider(_ECHO)
        engine = AsyncFormatShield("mock/echo")
        with pytest.raises(SchemaError):
            await engine.extract(_DOC)

    async def test_nfield_async_one_shot(self, install_provider):
        install_provider(_ECHO)
        result = await nfield_async(_DOC, _SCHEMA, "mock/echo")
        assert result.data["name"] == "Alice"

    def test_model_property(self, install_provider):
        install_provider(_ECHO)
        engine = AsyncFormatShield("mock/echo", _SCHEMA)
        assert engine.model == "mock/echo"

    def test_model_specs_reach_provider(self):
        # No fixture: build a real Groq provider (no API call at construction)
        # and confirm the engine forwarded the caller-supplied model specs.
        engine = AsyncFormatShield(
            "groq/llama-3.1-8b-instant",
            _SCHEMA,
            context_window=131_072,
            max_output_tokens=32_768,
        )
        assert engine._provider.context_window == 131_072
        assert engine._provider.max_output_tokens == 32_768

    async def test_calibration_runs_once_across_calls(self, install_provider):
        # Stage 0 calibration must be measured once and cached; reusing the
        # engine across documents must not re-measure chars_per_token.
        provider = install_provider(_ECHO)
        engine = AsyncFormatShield(
            "mock/echo", _SCHEMA, config=ExtractionConfig(max_retry_rounds=0)
        )
        await engine.extract("doc one")
        await engine.extract("doc two")
        await engine.extract("doc three")
        assert provider.token_calls == 1, (
            f"calibration ran {provider.token_calls} times; expected once (cached)"
        )


class TestNestedSchemaForms:
    """Lock the diamond-schema fix at the public engine boundary."""

    async def test_pydantic_reused_submodel_keeps_all_fields(self, install_provider):
        pydantic = pytest.importorskip("pydantic")

        class Addr(pydantic.BaseModel):
            city: str

        class Person(pydantic.BaseModel):
            home: Addr
            work: Addr

        install_provider("home.city = Paris\nwork.city = Lyon")
        engine = AsyncFormatShield(
            "mock/echo", Person, config=ExtractionConfig(max_retry_rounds=0)
        )
        result = await engine.extract("doc")
        # Both reused-submodel branches must survive (regression: work.* dropped).
        assert result.metadata.fields_total == 2
        assert result.data["home"]["city"] == "Paris"
        assert result.data["work"]["city"] == "Lyon"

    async def test_nested_dataclass_expands(self, install_provider):
        install_provider("home.city = Paris\nwork.city = Lyon")
        engine = AsyncFormatShield(
            "mock/echo", _Person, config=ExtractionConfig(max_retry_rounds=0)
        )
        result = await engine.extract("doc")
        assert result.metadata.fields_total == 2
        assert result.data["home"]["city"] == "Paris"
        assert result.data["work"]["city"] == "Lyon"


class TestSchemaDepthGuard:
    """A self-referential / pathologically deep dataclass fails cleanly."""

    def test_self_referential_dataclass_raises_schema_error(self):
        # Without the guard this recurses forever → RecursionError.
        with pytest.raises(SchemaError, match="nests deeper"):
            _dataclass_to_json_schema(_Tree)

    def test_self_referential_via_engine_construction(self, install_provider):
        install_provider(_ECHO)
        with pytest.raises(SchemaError, match="nests deeper"):
            AsyncFormatShield("mock/echo", _Tree)

    def test_normal_nesting_still_converts(self):
        # _Person nests one level (_Addr); well under the cap → no error.
        node = _dataclass_to_json_schema(_Person)
        assert node["properties"]["home"]["properties"]["city"]["type"] == "string"

    def test_depth_cap_is_a_named_constant(self):
        assert _MAX_SCHEMA_DEPTH > 0


class TestConcurrentCalibration:
    """Concurrent first-time extracts calibrate exactly once (LOW-1 lock)."""

    async def test_concurrent_extracts_calibrate_once(self, install_provider):
        provider = install_provider(_ECHO)
        engine = AsyncFormatShield(
            "mock/echo", _SCHEMA, config=ExtractionConfig(max_retry_rounds=0)
        )
        # Fire several extracts at once on a fresh engine: the calibration lock
        # must keep Stage 0 (the single count_tokens probe) from running per call.
        await asyncio.gather(
            engine.extract("doc one"),
            engine.extract("doc two"),
            engine.extract("doc three"),
        )
        assert provider.token_calls == 1, (
            f"calibration ran {provider.token_calls} times under concurrency; expected once (lock)"
        )
