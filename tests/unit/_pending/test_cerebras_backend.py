"""Unit tests for CerebrasBackend — no API calls or live credentials required."""

from __future__ import annotations

import inspect
import os

import pytest

from formatshield.backends.cerebras_backend import (
    CEREBRAS_BASE_URL,
    DEFAULT_CEREBRAS_MODEL,
    CerebrasBackend,
)


def test_cerebras_backend_name() -> None:
    """Backend identifier must be 'cerebras'."""
    backend = CerebrasBackend(api_key="fake-key")
    assert backend.name == "cerebras"


def test_cerebras_supports_kv_cache_reuse_is_false() -> None:
    """Cerebras does not expose KV-cache prefix reuse."""
    backend = CerebrasBackend(api_key="fake-key")
    assert backend.supports_kv_cache_reuse is False


def test_cerebras_accuracy_loss_baseline() -> None:
    """Baseline accuracy loss must be 0.15."""
    backend = CerebrasBackend(api_key="fake-key")
    assert backend.accuracy_loss_baseline == 0.15


def test_cerebras_model_prefix_stripped() -> None:
    """The 'cerebras/' prefix must be stripped from the model name."""
    backend = CerebrasBackend(api_key="fake-key", model="cerebras/llama3.1-70b")
    assert backend.model == "llama3.1-70b"


def test_cerebras_no_api_key_raises_value_error() -> None:
    """ValueError must be raised when no API key is available."""
    env_backup = os.environ.pop("CEREBRAS_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="CEREBRAS_API_KEY"):
            CerebrasBackend()
    finally:
        if env_backup is not None:
            os.environ["CEREBRAS_API_KEY"] = env_backup


def test_cerebras_has_generate_method() -> None:
    """CerebrasBackend must expose an async generate() method."""
    backend = CerebrasBackend(api_key="fake-key")
    assert hasattr(backend, "generate")
    assert inspect.iscoroutinefunction(backend.generate)


def test_cerebras_has_stream_method() -> None:
    """CerebrasBackend must expose a stream() async generator method."""
    backend = CerebrasBackend(api_key="fake-key")
    assert hasattr(backend, "stream")
    assert inspect.isasyncgenfunction(backend.stream)


def test_cerebras_default_model() -> None:
    """Default model must match DEFAULT_CEREBRAS_MODEL constant."""
    backend = CerebrasBackend(api_key="fake-key")
    assert backend.model == DEFAULT_CEREBRAS_MODEL


def test_cerebras_base_url_constant() -> None:
    """CEREBRAS_BASE_URL must point to the Cerebras API endpoint."""
    assert CEREBRAS_BASE_URL == "https://api.cerebras.ai/v1"


def test_cerebras_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """API key must be resolved from the CEREBRAS_API_KEY environment variable."""
    monkeypatch.setenv("CEREBRAS_API_KEY", "env-key")
    backend = CerebrasBackend()
    assert backend._api_key == "env-key"


def test_cerebras_build_messages_with_schema_adds_system_message() -> None:
    """A schema must cause a system message to be prepended."""
    backend = CerebrasBackend(api_key="fake-key")
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    messages = backend._build_messages("Test prompt", schema=schema, constraints=None)
    assert messages[0]["role"] == "system"
    assert '"answer"' in messages[0]["content"]
    assert messages[-1]["role"] == "user"
