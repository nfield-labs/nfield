"""Unit tests for FireworksBackend — no API calls or live credentials required."""

from __future__ import annotations

import inspect
import os

import pytest

from formatshield.backends.fireworks_backend import (
    DEFAULT_FIREWORKS_MODEL,
    FIREWORKS_BASE_URL,
    FireworksBackend,
)


def test_fireworks_backend_name() -> None:
    """Backend identifier must be 'fireworks'."""
    backend = FireworksBackend(api_key="fake-key")
    assert backend.name == "fireworks"


def test_fireworks_supports_kv_cache_reuse_is_false() -> None:
    """Fireworks AI does not expose KV-cache prefix reuse."""
    backend = FireworksBackend(api_key="fake-key")
    assert backend.supports_kv_cache_reuse is False


def test_fireworks_accuracy_loss_baseline() -> None:
    """Baseline accuracy loss must be 0.13."""
    backend = FireworksBackend(api_key="fake-key")
    assert backend.accuracy_loss_baseline == 0.13


def test_fireworks_model_prefix_stripped() -> None:
    """The 'fireworks/' prefix must be stripped from the model name."""
    backend = FireworksBackend(
        api_key="fake-key",
        model="fireworks/accounts/fireworks/models/llama-v3p1-70b-instruct",
    )
    assert backend.model == "accounts/fireworks/models/llama-v3p1-70b-instruct"


def test_fireworks_no_api_key_raises_value_error() -> None:
    """ValueError must be raised when no API key is available."""
    env_backup = os.environ.pop("FIREWORKS_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="FIREWORKS_API_KEY"):
            FireworksBackend()
    finally:
        if env_backup is not None:
            os.environ["FIREWORKS_API_KEY"] = env_backup


def test_fireworks_has_generate_method() -> None:
    """FireworksBackend must expose an async generate() method."""
    backend = FireworksBackend(api_key="fake-key")
    assert hasattr(backend, "generate")
    assert inspect.iscoroutinefunction(backend.generate)


def test_fireworks_has_stream_method() -> None:
    """FireworksBackend must expose a stream() async generator method."""
    backend = FireworksBackend(api_key="fake-key")
    assert hasattr(backend, "stream")
    assert inspect.isasyncgenfunction(backend.stream)


def test_fireworks_default_model() -> None:
    """Default model must match DEFAULT_FIREWORKS_MODEL constant."""
    backend = FireworksBackend(api_key="fake-key")
    assert backend.model == DEFAULT_FIREWORKS_MODEL


def test_fireworks_base_url_constant() -> None:
    """FIREWORKS_BASE_URL must point to the Fireworks AI inference endpoint."""
    assert FIREWORKS_BASE_URL == "https://api.fireworks.ai/inference/v1"


def test_fireworks_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """API key must be resolved from the FIREWORKS_API_KEY environment variable."""
    monkeypatch.setenv("FIREWORKS_API_KEY", "env-key")
    backend = FireworksBackend()
    assert backend._api_key == "env-key"


def test_fireworks_build_messages_with_schema_adds_system_message() -> None:
    """A schema must cause a system message to be prepended."""
    backend = FireworksBackend(api_key="fake-key")
    schema = {"type": "object", "properties": {"result": {"type": "number"}}}
    messages = backend._build_messages("Test prompt", schema=schema, constraints=None)
    assert messages[0]["role"] == "system"
    assert '"result"' in messages[0]["content"]
    assert messages[-1]["role"] == "user"
