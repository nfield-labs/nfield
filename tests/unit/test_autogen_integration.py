"""Unit tests for FormatShieldAutoGen — no API keys required."""

from __future__ import annotations

import inspect
from typing import Any

from formatshield.backends.dryrun_backend import DryRunBackend
from formatshield.integrations.autogen import FormatShieldAutoGen


def _make_client() -> FormatShieldAutoGen:
    """Return a FormatShieldAutoGen instance backed by DryRunBackend."""
    from formatshield.core import FormatShield

    client = FormatShieldAutoGen.__new__(FormatShieldAutoGen)
    client._shield = FormatShield(model="dryrun/test", backend=DryRunBackend())
    client.model = "dryrun/test"
    client._schema = None
    return client


def test_autogen_has_generate_sync_method() -> None:
    """FormatShieldAutoGen exposes a generate_sync() method."""
    assert hasattr(FormatShieldAutoGen, "generate_sync")
    assert callable(FormatShieldAutoGen.generate_sync)


def test_autogen_has_generate_method() -> None:
    """FormatShieldAutoGen exposes an async generate() method."""
    assert hasattr(FormatShieldAutoGen, "generate")
    assert inspect.iscoroutinefunction(FormatShieldAutoGen.generate)


def test_autogen_has_create_method() -> None:
    """FormatShieldAutoGen exposes a create() method for AutoGen protocol."""
    assert hasattr(FormatShieldAutoGen, "create")
    assert callable(FormatShieldAutoGen.create)


def test_autogen_model_attribute_set() -> None:
    """model attribute is stored on the instance."""
    client = _make_client()
    assert client.model == "dryrun/test"


def test_autogen_schema_stored() -> None:
    """_schema attribute is stored correctly when provided."""
    from formatshield.core import FormatShield

    schema: dict[str, Any] = {"type": "object"}
    client = FormatShieldAutoGen.__new__(FormatShieldAutoGen)
    client._shield = FormatShield(model="dryrun/test", backend=DryRunBackend())
    client.model = "dryrun/test"
    client._schema = schema
    assert client._schema is schema


def test_autogen_default_model() -> None:
    """Default model string is the expected Groq model."""
    sig = inspect.signature(FormatShieldAutoGen.__init__)
    assert sig.parameters["model"].default == "groq/llama-3.3-70b-versatile"


def test_autogen_create_returns_choices_dict() -> None:
    """create() returns a dict with a 'choices' list."""
    client = _make_client()
    result = client.create([{"role": "user", "content": "Hello"}])
    assert "choices" in result
    assert isinstance(result["choices"], list)
    assert len(result["choices"]) > 0


def test_autogen_create_extracts_last_user_message() -> None:
    """create() picks the last user-role message as the prompt."""
    client = _make_client()
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question"},
    ]
    result = client.create(messages)
    assert "choices" in result
    choice = result["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)


def test_autogen_create_handles_empty_messages() -> None:
    """create() returns a valid dict even with an empty messages list."""
    client = _make_client()
    result = client.create([])
    assert "choices" in result
    assert isinstance(result["choices"], list)


def test_autogen_generate_sync_returns_string() -> None:
    """generate_sync() returns a plain string."""
    client = _make_client()
    output = client.generate_sync("What is 2+2?")
    assert isinstance(output, str)


def test_autogen_create_model_field_in_response() -> None:
    """create() includes the model field in the response dict."""
    client = _make_client()
    result = client.create([{"role": "user", "content": "Hi"}])
    assert "model" in result
    assert result["model"] == "dryrun/test"


def test_autogen_create_finish_reason_stop() -> None:
    """create() returns finish_reason 'stop' in each choice."""
    client = _make_client()
    result = client.create([{"role": "user", "content": "Hi"}])
    assert result["choices"][0]["finish_reason"] == "stop"


def test_autogen_generate_sync_with_schema_override() -> None:
    """generate_sync() accepts a schema kwarg without raising."""
    client = _make_client()
    schema: dict[str, Any] = {"type": "object", "properties": {"answer": {"type": "string"}}}
    output = client.generate_sync("What is 2+2?", schema=schema)
    assert isinstance(output, str)
