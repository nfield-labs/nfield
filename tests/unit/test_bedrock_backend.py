"""Unit tests for BedrockBackend — no AWS credentials or boto3 calls required."""

from __future__ import annotations

import inspect

from formatshield.backends.bedrock_backend import BedrockBackend


def test_bedrock_backend_name() -> None:
    """Backend identifier must be 'bedrock'."""
    backend = BedrockBackend()
    assert backend.name == "bedrock"


def test_bedrock_supports_kv_cache_reuse_is_false() -> None:
    """Bedrock does not expose KV-cache prefix reuse."""
    backend = BedrockBackend()
    assert backend.supports_kv_cache_reuse is False


def test_bedrock_accuracy_loss_baseline() -> None:
    """Baseline accuracy loss must be 0.16."""
    backend = BedrockBackend()
    assert backend.accuracy_loss_baseline == 0.16


def test_bedrock_model_prefix_stripped() -> None:
    """The 'bedrock/' prefix must be stripped from the model ID."""
    backend = BedrockBackend(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")
    assert backend._model_id == "anthropic.claude-3-5-sonnet-20241022-v2:0"


def test_bedrock_default_region() -> None:
    """Default region must be 'us-east-1' when AWS_DEFAULT_REGION is not set."""
    import os

    env_backup = os.environ.pop("AWS_DEFAULT_REGION", None)
    try:
        backend = BedrockBackend()
        assert backend._region == "us-east-1"
    finally:
        if env_backup is not None:
            os.environ["AWS_DEFAULT_REGION"] = env_backup


def test_bedrock_custom_region_accepted() -> None:
    """A custom region passed explicitly must be stored."""
    backend = BedrockBackend(region="eu-west-1")
    assert backend._region == "eu-west-1"


def test_bedrock_has_generate_method() -> None:
    """BedrockBackend must expose an async generate() method."""
    backend = BedrockBackend()
    assert hasattr(backend, "generate")
    assert inspect.iscoroutinefunction(backend.generate)


def test_bedrock_has_stream_method() -> None:
    """BedrockBackend must expose a stream() async generator method."""
    backend = BedrockBackend()
    assert hasattr(backend, "stream")
    assert inspect.isasyncgenfunction(backend.stream)


def test_bedrock_default_model_id() -> None:
    """Default model ID must be the Claude 3.5 Sonnet v2 Bedrock model."""
    backend = BedrockBackend()
    assert backend._model_id == "anthropic.claude-3-5-sonnet-20241022-v2:0"


def test_bedrock_build_messages_without_schema() -> None:
    """Without a schema, the prompt is passed through unchanged."""
    backend = BedrockBackend()
    messages = backend._build_messages("Hello", schema=None)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "Hello" in messages[0]["content"][0]["text"]


def test_bedrock_build_messages_with_schema_embeds_schema() -> None:
    """With a schema, the schema JSON must appear in the user message."""
    backend = BedrockBackend()
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    messages = backend._build_messages("Extract name", schema=schema)
    text = messages[0]["content"][0]["text"]
    assert '"name"' in text
    assert "Extract name" in text
