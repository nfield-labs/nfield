"""Unit tests for VertexAIBackend — no Google credentials or SDK calls required."""

from __future__ import annotations

import inspect

from formatshield.backends.vertexai_backend import VertexAIBackend


def test_vertexai_backend_name() -> None:
    """Backend identifier must be 'vertexai'."""
    backend = VertexAIBackend()
    assert backend.name == "vertexai"


def test_vertexai_supports_kv_cache_reuse_is_false() -> None:
    """Vertex AI does not expose KV-cache prefix reuse."""
    backend = VertexAIBackend()
    assert backend.supports_kv_cache_reuse is False


def test_vertexai_accuracy_loss_baseline() -> None:
    """Baseline accuracy loss must be 0.14."""
    backend = VertexAIBackend()
    assert backend.accuracy_loss_baseline == 0.14


def test_vertexai_model_prefix_stripped() -> None:
    """The 'vertexai/' prefix must be stripped from the model name."""
    backend = VertexAIBackend(model="vertexai/gemini-2.0-flash-001")
    assert backend._model_name == "gemini-2.0-flash-001"


def test_vertexai_default_location() -> None:
    """Default location must be 'us-central1'."""
    backend = VertexAIBackend()
    assert backend._location == "us-central1"


def test_vertexai_custom_project_accepted() -> None:
    """A custom project ID passed explicitly must be stored."""
    backend = VertexAIBackend(project="my-gcp-project")
    assert backend._project == "my-gcp-project"


def test_vertexai_has_generate_method() -> None:
    """VertexAIBackend must expose an async generate() method."""
    backend = VertexAIBackend()
    assert hasattr(backend, "generate")
    assert inspect.iscoroutinefunction(backend.generate)


def test_vertexai_has_stream_method() -> None:
    """VertexAIBackend must expose a stream() async generator method."""
    backend = VertexAIBackend()
    assert hasattr(backend, "stream")
    assert inspect.isasyncgenfunction(backend.stream)


def test_vertexai_default_model_name() -> None:
    """Default model name must be 'gemini-2.0-flash-001'."""
    backend = VertexAIBackend()
    assert backend._model_name == "gemini-2.0-flash-001"


def test_vertexai_custom_location_accepted() -> None:
    """A custom location passed explicitly must be stored."""
    backend = VertexAIBackend(location="europe-west4")
    assert backend._location == "europe-west4"


def test_vertexai_model_prefix_not_double_stripped() -> None:
    """A model without a 'vertexai/' prefix must be stored as-is."""
    backend = VertexAIBackend(model="gemini-1.5-pro-002")
    assert backend._model_name == "gemini-1.5-pro-002"
