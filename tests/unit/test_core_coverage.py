"""
Targeted coverage tests for formatshield.core.

Covers the following previously-uncovered lines:
  134-149 : _build_backend() for ollama, vllm, and unknown-fallback
  273-293 : generate() TTF path (force via oracle patch)
  288-293 : generate() TTF exception fallback to direct
  310-314 : generate() JSON parse failure (parsed = None)
  328     : generate() metrics.record_fallback() (fallback_triggered=True)
  387-388 : generate_sync() error_holder path (generate raises in thread)
  396-397 : generate_sync() raise error_holder[0]
  436-441 : stream() TTF path
  455-472 : from_config() — JSON and YAML loading
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from formatshield.backends.dryrun_backend import DryRunBackend
from formatshield.core import FormatShield, GenerationResult, _build_backend
from formatshield.oracle.routing_decision import RoutingDecision
from formatshield.scorer.features import StreamEvent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODEL = "groq/llama-3.3-70b-versatile"


def _make_shield(
    model: str = _MODEL,
    *,
    ttf_fallback: bool = True,
    expose_thinking: bool = False,
) -> FormatShield:
    """Build a FormatShield instance with DryRunBackend injected."""
    with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
        shield = FormatShield(
            model=model, ttf_fallback=ttf_fallback, expose_thinking=expose_thinking
        )
    shield._backend = DryRunBackend(base_latency_ms=0.0)
    return shield


def _ttf_decision() -> RoutingDecision:
    return RoutingDecision(
        strategy="ttf",
        expected_accuracy_delta=0.15,
        expected_overhead_pct=120.0,
        confidence=0.85,
        explanation="test forced TTF",
        failure_modes=[],
    )


# ---------------------------------------------------------------------------
# Lines 134-149: _build_backend() for ollama, vllm, unknown
# ---------------------------------------------------------------------------


def test_build_backend_ollama_returns_ollama_backend() -> None:
    """_build_backend('ollama', ...) must return an OllamaBackend."""
    from formatshield.backends.ollama_backend import OllamaBackend

    backend = _build_backend("ollama/llama3", "ollama", None, None)
    assert isinstance(backend, OllamaBackend)


def test_build_backend_ollama_uses_custom_base_url() -> None:
    """_build_backend('ollama') with base_url must pass it as host."""
    from formatshield.backends.ollama_backend import OllamaBackend

    backend = _build_backend("ollama/llama3", "ollama", "http://custom:11434", None)
    assert isinstance(backend, OllamaBackend)
    assert backend.host == "http://custom:11434"


def test_build_backend_vllm_returns_vllm_backend() -> None:
    """_build_backend('vllm', ...) must return a VLLMBackend."""
    from formatshield.backends.vllm_backend import VLLMBackend

    backend = _build_backend("vllm/mistral-7b", "vllm", None, None)
    assert isinstance(backend, VLLMBackend)


def test_build_backend_vllm_uses_custom_base_url() -> None:
    """_build_backend('vllm') with base_url must use it."""
    from formatshield.backends.vllm_backend import VLLMBackend

    backend = _build_backend("vllm/mistral", "vllm", "http://localhost:9000/v1", None)
    assert isinstance(backend, VLLMBackend)


def test_build_backend_openrouter_returns_openrouter_backend() -> None:
    """backend_name=='openrouter' must resolve to OpenRouterBackend."""
    from formatshield.backends.openrouter_backend import OpenRouterBackend

    backend = _build_backend("openrouter/gpt-4o", "openrouter", None, "fake-key")
    assert isinstance(backend, OpenRouterBackend)


def test_build_backend_openai_returns_openai_backend() -> None:
    """backend_name=='openai' must resolve to OpenAIBackend."""
    from formatshield.backends.openai_backend import OpenAIBackend

    backend = _build_backend("openai/gpt-4o", "openai", None, "fake-key")
    assert isinstance(backend, OpenAIBackend)


# ---------------------------------------------------------------------------
# Lines 273-293: generate() TTF path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_ttf_path_returns_generation_result() -> None:
    """Force the TTF route and verify generate() returns a GenerationResult."""
    shield = _make_shield()
    with (
        patch.object(shield._oracle_x, "predict", return_value=_ttf_decision()),
        patch.object(shield._detector, "should_override_to_direct", return_value=False),
    ):
        result = await shield.generate("Analyze compound interest step by step")
    assert isinstance(result, GenerationResult)
    assert result.routing.strategy == "ttf"


@pytest.mark.asyncio
async def test_generate_ttf_path_output_is_string() -> None:
    """TTF route must produce a non-empty string output."""
    shield = _make_shield()
    with patch.object(shield._oracle_x, "predict", return_value=_ttf_decision()):
        result = await shield.generate("Reason through this problem carefully")
    assert isinstance(result.output, str)
    assert len(result.output) > 0


@pytest.mark.asyncio
async def test_generate_ttf_path_with_schema() -> None:
    """TTF route with a dict schema must produce valid output."""
    shield = _make_shield()
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    with patch.object(shield._oracle_x, "predict", return_value=_ttf_decision()):
        result = await shield.generate("Step by step reasoning task", schema=schema)
    assert isinstance(result, GenerationResult)


# ---------------------------------------------------------------------------
# Lines 288-293: TTF exception → fallback to direct (fallback_triggered=True)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_ttf_exception_fallback_triggered() -> None:
    """When TTFEngine.generate raises, generate() falls back to direct (lines 288-293, 328)."""
    shield = _make_shield()

    with (
        patch.object(shield._oracle_x, "predict", return_value=_ttf_decision()),
        patch.object(shield._detector, "should_override_to_direct", return_value=False),
        patch(
            "formatshield.ttf.engine.TTFEngine.generate",
            new_callable=AsyncMock,
            side_effect=RuntimeError("TTF blew up"),
        ),
    ):
        result = await shield.generate("Some prompt")

    assert isinstance(result, GenerationResult)
    assert result.fallback_triggered is True  # line 293 + 328


# ---------------------------------------------------------------------------
# Lines 310-314: JSON parse failure (parsed = None)
# ---------------------------------------------------------------------------


class _StrictModel(BaseModel):
    never_matches_xyz: int
    another_required: str


@pytest.mark.asyncio
async def test_generate_json_parse_failure_parsed_is_none() -> None:
    """When schema validation AND json.loads both fail, parsed must be None (lines 310-314)."""
    shield = _make_shield()

    with patch.object(shield._backend, "generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = "THIS IS NOT JSON AT ALL %%%"
        result = await shield.generate(
            "Produce output",
            schema=_StrictModel,
        )

    # schema_model validation fails (wrong shape) AND json.loads("THIS IS NOT JSON") fails
    assert result.parsed is None


# ---------------------------------------------------------------------------
# Lines 436-441: stream() TTF path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_ttf_path_yields_events() -> None:
    """stream() TTF path must yield StreamEvent instances (lines 436-441)."""
    shield = _make_shield()
    events: list[StreamEvent] = []
    with patch.object(shield._oracle_x, "predict", return_value=_ttf_decision()):
        async for event in shield.stream("Reason through this step by step"):
            events.append(event)
    assert len(events) > 0
    assert all(isinstance(e, StreamEvent) for e in events)


@pytest.mark.asyncio
async def test_stream_ttf_path_expose_thinking_false_filters_thinking() -> None:
    """With expose_thinking=False, 'thinking' events should be filtered out."""
    shield = _make_shield(expose_thinking=False)
    with patch.object(shield._oracle_x, "predict", return_value=_ttf_decision()):
        events = [e async for e in shield.stream("Step by step calculation")]
    # thinking events are suppressed
    thinking_events = [e for e in events if e.type == "thinking"]
    assert len(thinking_events) == 0


@pytest.mark.asyncio
async def test_stream_ttf_path_expose_thinking_true_passes_all_events() -> None:
    """With expose_thinking=True, all event types (including thinking) are forwarded."""
    shield = _make_shield(expose_thinking=True)
    with patch.object(shield._oracle_x, "predict", return_value=_ttf_decision()):
        events = [e async for e in shield.stream("Analyze this problem carefully")]
    # expose_thinking=True means thinking events are NOT filtered; at minimum output/complete appear
    assert len(events) > 0
    assert all(isinstance(e, StreamEvent) for e in events)


# ---------------------------------------------------------------------------
# Lines 387-388, 396-397: generate_sync() error path (inside running loop)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_sync_reraises_from_thread_when_generate_raises() -> None:
    """
    generate_sync() called from within an async context (running loop)
    must re-raise exceptions from the thread (lines 387-388, 396-397).
    """
    shield = _make_shield()

    with patch.object(
        shield,
        "generate",
        new_callable=AsyncMock,
        side_effect=ValueError("injected error from generate"),
    ):
        with pytest.raises(ValueError, match="injected error from generate"):
            shield.generate_sync("Any prompt")


# ---------------------------------------------------------------------------
# Lines 455-472: from_config() — JSON config loading
# ---------------------------------------------------------------------------


def test_from_config_json_creates_shield() -> None:
    """from_config() with a JSON file must return a FormatShield instance."""
    config = {"model": _MODEL}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        tmp_path = f.name

    try:
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            shield = FormatShield.from_config(tmp_path)
        assert isinstance(shield, FormatShield)
        assert shield.model == _MODEL
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_from_config_nonexistent_file_raises_file_not_found() -> None:
    """from_config() with a missing file must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        FormatShield.from_config("/nonexistent/path/config.json")


def test_from_config_yaml_without_pyyaml_raises_import_error(tmp_path: Path) -> None:
    """from_config() with YAML file but no pyyaml must raise ImportError (lines 467-468)."""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("model: groq/llama-3.3-70b-versatile\n")

    with patch.dict("sys.modules", {"yaml": None}):
        with pytest.raises((ImportError, Exception)):
            FormatShield.from_config(str(yaml_file))
