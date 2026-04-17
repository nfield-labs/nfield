"""Unit tests for formatshield.backends.sglang_backend.SGLangBackend.

Covers construction, property values, and interface checks — no real API
calls are made and no API key is required (SGLang is a local server).
"""

from __future__ import annotations

from formatshield.backends.sglang_backend import _DEFAULT_BASE_URL, _DEFAULT_MODEL, SGLangBackend

# ---------------------------------------------------------------------------
# Construction / properties
# ---------------------------------------------------------------------------


def test_sglang_backend_name() -> None:
    """SGLangBackend.name must be 'sglang'."""
    backend = SGLangBackend()
    assert backend.name == "sglang"


def test_sglang_kv_cache_reuse_true() -> None:
    """SGLangBackend.supports_kv_cache_reuse must be True (RadixAttention)."""
    backend = SGLangBackend()
    assert backend.supports_kv_cache_reuse is True


def test_sglang_accuracy_loss_baseline() -> None:
    """SGLangBackend.accuracy_loss_baseline must be 0.20."""
    backend = SGLangBackend()
    assert backend.accuracy_loss_baseline == 0.20


def test_sglang_model_prefix_stripped() -> None:
    """'sglang/' prefix is stripped from the model name."""
    backend = SGLangBackend(model="sglang/meta-llama/Llama-3.1-8B-Instruct")
    assert backend.model == "meta-llama/Llama-3.1-8B-Instruct"


def test_sglang_plain_model_name_unchanged() -> None:
    """Model name without 'sglang/' prefix is kept as-is."""
    backend = SGLangBackend(model="meta-llama/Llama-3.1-8B-Instruct")
    assert backend.model == "meta-llama/Llama-3.1-8B-Instruct"


def test_sglang_default_base_url() -> None:
    """SGLangBackend default base_url is 'http://localhost:30000/v1'."""
    backend = SGLangBackend()
    assert backend._base_url == "http://localhost:30000/v1"
    assert backend._base_url == _DEFAULT_BASE_URL


def test_sglang_custom_base_url() -> None:
    """SGLangBackend accepts a custom base_url."""
    custom_url = "http://my-sglang-server:8080/v1"
    backend = SGLangBackend(base_url=custom_url)
    assert backend._base_url == custom_url


def test_sglang_has_generate_method() -> None:
    """SGLangBackend must have a generate method."""
    backend = SGLangBackend()
    assert hasattr(backend, "generate")
    assert callable(backend.generate)


def test_sglang_has_stream_method() -> None:
    """SGLangBackend must have a stream method."""
    backend = SGLangBackend()
    assert hasattr(backend, "stream")
    assert callable(backend.stream)


def test_sglang_default_model() -> None:
    """SGLangBackend default model is 'meta-llama/Llama-3.1-8B-Instruct'."""
    backend = SGLangBackend()
    assert backend.model == _DEFAULT_MODEL


def test_sglang_accuracy_loss_baseline_in_range() -> None:
    """accuracy_loss_baseline must be a float in (0.0, 1.0)."""
    backend = SGLangBackend()
    baseline = backend.accuracy_loss_baseline
    assert baseline is not None
    assert 0.0 < baseline < 1.0


def test_sglang_no_api_key_required() -> None:
    """SGLangBackend can be constructed without any API key."""
    backend = SGLangBackend()
    assert backend is not None


def test_sglang_custom_api_key() -> None:
    """SGLangBackend accepts a custom api_key."""
    backend = SGLangBackend(api_key="my-sglang-key")
    assert backend is not None


def test_sglang_build_messages_no_schema() -> None:
    """_build_messages with no schema and no constraints returns only user message."""
    backend = SGLangBackend()
    messages = backend._build_messages("Hello", None, None)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello"


def test_sglang_build_messages_with_schema() -> None:
    """_build_messages with schema prepends a system message."""
    backend = SGLangBackend()
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    messages = backend._build_messages("Extract the name", schema, None)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "JSON schema" in messages[0]["content"]
    assert messages[1]["role"] == "user"


def test_sglang_build_messages_json_constraint() -> None:
    """_build_messages with constraints='json' prepends a JSON system message."""
    backend = SGLangBackend()
    messages = backend._build_messages("Return JSON", None, "json")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "JSON" in messages[0]["content"]
    assert messages[1]["role"] == "user"


def test_sglang_has_required_attributes() -> None:
    """SGLangBackend has all required Backend protocol attributes."""
    backend = SGLangBackend()
    assert hasattr(backend, "name")
    assert hasattr(backend, "supports_kv_cache_reuse")
    assert hasattr(backend, "accuracy_loss_baseline")
