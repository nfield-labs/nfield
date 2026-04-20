"""Unit tests for TransformersBackend — no API keys or GPU required."""

from __future__ import annotations

from formatshield.backends.transformers_backend import TransformersBackend


def test_transformers_backend_name() -> None:
    """Backend identifier must be 'transformers'."""
    backend = TransformersBackend()
    assert backend.name == "transformers"


def test_transformers_kv_cache_false() -> None:
    """TransformersBackend does not support KV-cache reuse."""
    backend = TransformersBackend()
    assert backend.supports_kv_cache_reuse is False


def test_transformers_accuracy_loss_baseline() -> None:
    """Baseline accuracy loss should be 0.10."""
    backend = TransformersBackend()
    assert backend.accuracy_loss_baseline == 0.10


def test_transformers_model_prefix_stripped() -> None:
    """'transformers/' prefix is removed from the model name."""
    backend = TransformersBackend(model="transformers/llama")
    assert backend._model_name == "llama"


def test_transformers_hf_prefix_stripped() -> None:
    """'hf/' prefix is removed from the model name."""
    backend = TransformersBackend(model="hf/mistral")
    assert backend._model_name == "mistral"


def test_transformers_has_generate_method() -> None:
    """Backend exposes a generate() coroutine."""
    backend = TransformersBackend()
    assert hasattr(backend, "generate")
    import asyncio

    assert asyncio.iscoroutinefunction(backend.generate)


def test_transformers_has_stream_method() -> None:
    """Backend exposes a stream() async generator method."""
    backend = TransformersBackend()
    assert hasattr(backend, "stream")


def test_transformers_default_model() -> None:
    """Default model should be the Llama-3.1-8B-Instruct Hub ID."""
    backend = TransformersBackend()
    assert backend._model_name == "meta-llama/Llama-3.1-8B-Instruct"


def test_transformers_device_stored() -> None:
    """Device parameter is stored correctly."""
    backend = TransformersBackend(device="cuda")
    assert backend._device == "cuda"


def test_transformers_pipeline_initially_none() -> None:
    """Pipeline is not loaded until first generate() call (lazy loading)."""
    backend = TransformersBackend()
    assert backend._pipeline is None


def test_transformers_build_prompt_no_schema() -> None:
    """Without a schema the prompt is returned unchanged."""
    backend = TransformersBackend()
    prompt = "Hello, world!"
    assert backend._build_prompt(prompt, None) == prompt


def test_transformers_build_prompt_with_schema() -> None:
    """With a schema the prompt is augmented with schema instructions."""
    import json

    backend = TransformersBackend()
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    result = backend._build_prompt("What is 2+2?", schema)
    assert json.dumps(schema, indent=2) in result
    assert "What is 2+2?" in result
    assert "JSON" in result


def test_transformers_import_error_on_missing_library() -> None:
    """_get_pipeline raises ImportError with install instructions when transformers is absent."""
    import sys

    import pytest

    backend = TransformersBackend()
    # Remove transformers from sys.modules so the import inside _get_pipeline fails.
    original = sys.modules.pop("transformers", None)
    # Insert a sentinel that causes `from transformers import pipeline` to raise.
    sys.modules["transformers"] = None  # type: ignore[assignment]
    try:
        backend._pipeline = None
        with pytest.raises(ImportError, match="formatshield\\[transformers\\]"):
            backend._get_pipeline()
    finally:
        if original is not None:
            sys.modules["transformers"] = original
        else:
            sys.modules.pop("transformers", None)
