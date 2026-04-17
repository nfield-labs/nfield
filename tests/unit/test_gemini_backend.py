"""Unit tests for formatshield.backends.gemini_backend.GeminiBackend.

Covers construction, property values, and interface checks — no real API
calls are made.  ``google-generativeai`` is an optional dependency not
present in the test environment; tests that do not call generate/stream
do not require it.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from formatshield.backends.gemini_backend import GeminiBackend

# ---------------------------------------------------------------------------
# Construction / properties
# ---------------------------------------------------------------------------


def test_gemini_backend_name() -> None:
    """GeminiBackend.name must be 'gemini'."""
    backend = GeminiBackend(api_key="fake-key-for-testing")
    assert backend.name == "gemini"


def test_gemini_backend_attributes() -> None:
    """GeminiBackend must have supports_kv_cache_reuse and accuracy_loss_baseline."""
    backend = GeminiBackend(api_key="fake-key-for-testing")
    assert hasattr(backend, "supports_kv_cache_reuse")
    assert hasattr(backend, "accuracy_loss_baseline")


def test_gemini_kv_cache_false() -> None:
    """GeminiBackend.supports_kv_cache_reuse must be False."""
    backend = GeminiBackend(api_key="fake-key-for-testing")
    assert backend.supports_kv_cache_reuse is False


def test_gemini_accuracy_loss_baseline() -> None:
    """GeminiBackend.accuracy_loss_baseline must be 0.14."""
    backend = GeminiBackend(api_key="fake-key-for-testing")
    assert backend.accuracy_loss_baseline == 0.14


def test_gemini_model_prefix_stripped() -> None:
    """'gemini/' prefix is stripped from the model name."""
    backend = GeminiBackend(api_key="fake-key-for-testing", model="gemini/gemini-2.0-flash")
    assert backend._model_name == "gemini-2.0-flash"


def test_gemini_plain_model_name_unchanged() -> None:
    """Model name without 'gemini/' prefix is kept as-is."""
    backend = GeminiBackend(api_key="fake-key-for-testing", model="gemini-2.0-flash")
    assert backend._model_name == "gemini-2.0-flash"


def test_gemini_no_api_key_raises() -> None:
    """GeminiBackend raises ValueError when no API key is available."""
    with patch.dict("os.environ", {}, clear=True):
        os.environ.pop("GEMINI_API_KEY", None)
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            GeminiBackend(api_key=None)


def test_gemini_has_generate_method() -> None:
    """GeminiBackend must have a generate method."""
    backend = GeminiBackend(api_key="fake-key-for-testing")
    assert hasattr(backend, "generate")
    assert callable(backend.generate)


def test_gemini_has_stream_method() -> None:
    """GeminiBackend must have a stream method."""
    backend = GeminiBackend(api_key="fake-key-for-testing")
    assert hasattr(backend, "stream")
    assert callable(backend.stream)


def test_gemini_default_model() -> None:
    """GeminiBackend default model is 'gemini-2.0-flash'."""
    backend = GeminiBackend(api_key="fake-key-for-testing")
    assert backend._model_name == "gemini-2.0-flash"


def test_gemini_accuracy_loss_baseline_in_range() -> None:
    """accuracy_loss_baseline must be a float in (0.0, 1.0)."""
    backend = GeminiBackend(api_key="fake-key-for-testing")
    baseline = backend.accuracy_loss_baseline
    assert baseline is not None
    assert 0.0 < baseline < 1.0


def test_gemini_api_key_env_var() -> None:
    """GeminiBackend reads the GEMINI_API_KEY environment variable."""
    with patch.dict("os.environ", {"GEMINI_API_KEY": "env-test-key"}):
        backend = GeminiBackend()
        assert backend._api_key == "env-test-key"


def test_gemini_explicit_api_key_takes_precedence() -> None:
    """Explicit api_key takes precedence over the GEMINI_API_KEY env var."""
    with patch.dict("os.environ", {"GEMINI_API_KEY": "env-key"}):
        backend = GeminiBackend(api_key="explicit-key")
        assert backend._api_key == "explicit-key"


def test_gemini_build_prompt_without_schema() -> None:
    """_build_prompt with no schema returns the raw prompt."""
    backend = GeminiBackend(api_key="fake-key-for-testing")
    result = backend._build_prompt("Hello world", None)
    assert result == "Hello world"


def test_gemini_build_prompt_with_schema_embeds_instructions() -> None:
    """_build_prompt with schema prepends JSON formatting instructions."""
    backend = GeminiBackend(api_key="fake-key-for-testing")
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    result = backend._build_prompt("Extract the name", schema)
    assert "JSON schema" in result
    assert "Extract the name" in result
    assert "name" in result


def test_gemini_has_required_attributes() -> None:
    """GeminiBackend has all required Backend protocol attributes."""
    backend = GeminiBackend(api_key="fake-key-for-testing")
    assert hasattr(backend, "name")
    assert hasattr(backend, "supports_kv_cache_reuse")
    assert hasattr(backend, "accuracy_loss_baseline")
