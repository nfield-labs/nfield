"""Unit tests for LlamaCppBackend — no API keys, GPU, or GGUF files required."""

from __future__ import annotations

from formatshield.backends.llamacpp_backend import LlamaCppBackend


def test_llamacpp_backend_name() -> None:
    """Backend identifier must be 'llamacpp'."""
    backend = LlamaCppBackend()
    assert backend.name == "llamacpp"


def test_llamacpp_kv_cache_false() -> None:
    """LlamaCppBackend does not support KV-cache reuse."""
    backend = LlamaCppBackend()
    assert backend.supports_kv_cache_reuse is False


def test_llamacpp_accuracy_loss_baseline() -> None:
    """Baseline accuracy loss should be 0.12."""
    backend = LlamaCppBackend()
    assert backend.accuracy_loss_baseline == 0.12


def test_llamacpp_model_prefix_stripped() -> None:
    """'llamacpp/' prefix is removed from the model path."""
    backend = LlamaCppBackend(model="llamacpp/model.gguf")
    assert backend._model_path == "model.gguf"


def test_llamacpp_has_generate_method() -> None:
    """Backend exposes a generate() coroutine."""
    import asyncio

    backend = LlamaCppBackend()
    assert hasattr(backend, "generate")
    assert asyncio.iscoroutinefunction(backend.generate)


def test_llamacpp_has_stream_method() -> None:
    """Backend exposes a stream() async generator method."""
    backend = LlamaCppBackend()
    assert hasattr(backend, "stream")


def test_llamacpp_default_model() -> None:
    """Default model path should point to the bundled GGUF placeholder."""
    backend = LlamaCppBackend()
    assert backend._model_path == "models/llama-3.1-8b.gguf"


def test_llamacpp_custom_n_ctx() -> None:
    """Custom n_ctx value is stored on the instance."""
    backend = LlamaCppBackend(n_ctx=8192)
    assert backend._n_ctx == 8192


def test_llamacpp_llm_initially_none() -> None:
    """Llama instance is not created until first generate() call (lazy loading)."""
    backend = LlamaCppBackend()
    assert backend._llm is None


def test_llamacpp_n_gpu_layers_stored() -> None:
    """n_gpu_layers is stored correctly."""
    backend = LlamaCppBackend(n_gpu_layers=32)
    assert backend._n_gpu_layers == 32


def test_llamacpp_verbose_stored() -> None:
    """verbose flag is stored correctly."""
    backend = LlamaCppBackend(verbose=True)
    assert backend._verbose is True


def test_llamacpp_import_error_on_missing_library() -> None:
    """_get_llm raises ImportError with install instructions when llama-cpp-python is absent."""
    import sys

    import pytest

    backend = LlamaCppBackend()
    # llama_cpp is not installed in the test environment; confirm it's absent.
    original = sys.modules.pop("llama_cpp", None)
    # Insert a sentinel that causes `from llama_cpp import Llama` to raise.
    sys.modules["llama_cpp"] = None  # type: ignore[assignment]
    try:
        backend._llm = None
        with pytest.raises(ImportError, match="formatshield\\[llamacpp\\]"):
            backend._get_llm()
    finally:
        if original is not None:
            sys.modules["llama_cpp"] = original
        else:
            sys.modules.pop("llama_cpp", None)


def test_llamacpp_build_user_content_no_schema() -> None:
    """Without a schema the prompt is returned unchanged."""
    backend = LlamaCppBackend()
    prompt = "Hello"
    assert backend._build_user_content(prompt, None, use_grammar_mode=False) == prompt


def test_llamacpp_build_user_content_with_schema_grammar_mode() -> None:
    """With a schema and grammar mode the prompt includes schema instructions."""
    import json

    backend = LlamaCppBackend()
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    result = backend._build_user_content("Who are you?", schema, use_grammar_mode=True)
    assert json.dumps(schema, indent=2) in result
    assert "Who are you?" in result
