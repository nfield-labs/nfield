"""Tests for AsyncFormatShield: context manager, schema caching, call forms."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from formatshield import AsyncFormatShield, nfield_async
from formatshield.config import ExtractionConfig
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
