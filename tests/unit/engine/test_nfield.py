"""Contract tests for the public ``nfield`` entry point.

Covers the 2/3/4-parameter call forms, the model-resolution fallback chain,
the three accepted schema shapes (dict / Pydantic / dataclass), and the shape
of the returned ExtractionResult. All runs use the mock provider, so no network
access occurs.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from formatshield import ExtractionResult, ExtractionStatus, nfield
from formatshield.config import ExtractionConfig
from formatshield.exceptions import SchemaError

_DOC = "Name: Alice. Age: 30."

_DICT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
}

_ECHO = "name = Alice\nage = 30"


class TestNfieldContract:
    def test_three_param_returns_extraction_result(self, install_provider):
        install_provider(_ECHO)
        result = nfield(_DOC, _DICT_SCHEMA, "mock/echo")
        assert isinstance(result, ExtractionResult)
        assert isinstance(result.status, ExtractionStatus)
        assert result.data["name"] == "Alice"
        assert result.data["age"] == 30

    def test_four_param_accepts_config(self, install_provider):
        install_provider(_ECHO)
        config = ExtractionConfig(max_retry_rounds=0)
        result = nfield(_DOC, _DICT_SCHEMA, "mock/echo", config=config)
        assert result.metadata.fields_total == 2

    def test_two_param_falls_back_to_env_var(self, install_provider, monkeypatch):
        install_provider(_ECHO)
        monkeypatch.setenv("FORMATSHIELD_MODEL", "mock/echo")
        result = nfield(_DOC, _DICT_SCHEMA)
        assert result.data["name"] == "Alice"

    def test_two_param_falls_back_to_config_default_model(self, install_provider):
        install_provider(_ECHO)
        config = ExtractionConfig(default_model="mock/echo")
        result = nfield(_DOC, _DICT_SCHEMA, config=config)
        assert result.data["age"] == 30

    def test_no_model_anywhere_raises(self, install_provider, monkeypatch):
        install_provider(_ECHO)
        monkeypatch.delenv("FORMATSHIELD_MODEL", raising=False)
        with pytest.raises(SchemaError):
            nfield(_DOC, _DICT_SCHEMA)

    def test_result_metadata_shape(self, install_provider):
        install_provider(_ECHO)
        meta = nfield(_DOC, _DICT_SCHEMA, "mock/echo").metadata
        assert meta.fields_total == 2
        assert meta.K >= 1
        assert meta.K_min >= 1


class TestSchemaInputForms:
    def test_dataclass_schema(self, install_provider):
        @dataclass
        class Person:
            name: str
            age: int

        install_provider(_ECHO)
        result = nfield(_DOC, Person, "mock/echo")
        assert result.data["name"] == "Alice"
        assert result.data["age"] == 30

    def test_dataclass_instance_schema(self, install_provider):
        @dataclass
        class Person:
            name: str
            age: int

        install_provider(_ECHO)
        result = nfield(_DOC, Person(name="x", age=0), "mock/echo")
        assert result.metadata.fields_total == 2

    def test_pydantic_schema(self, install_provider):
        pydantic = pytest.importorskip("pydantic")

        class Person(pydantic.BaseModel):
            name: str
            age: int

        install_provider(_ECHO)
        result = nfield(_DOC, Person, "mock/echo")
        assert result.data["name"] == "Alice"
        assert result.data["age"] == 30

    def test_unsupported_schema_raises(self, install_provider):
        install_provider(_ECHO)
        with pytest.raises(SchemaError):
            nfield(_DOC, "not a schema", "mock/echo")
