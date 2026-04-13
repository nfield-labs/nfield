"""Unit tests for OutlinesBackend and GuidanceBackend.

Both ``outlines`` and ``guidance`` are optional dependencies that are NOT
installed in the test environment.  Every test either:

* Patches ``sys.modules`` to simulate the library being available, or
* Verifies that the correct ``ImportError`` is raised when the module is absent.

The backend classes themselves import cleanly at module level because they use
lazy imports (inside each method body).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from formatshield.backends.guidance_backend import GuidanceBackend
from formatshield.backends.outlines_backend import OutlinesBackend
from formatshield.scorer.features import StreamEvent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_outlines_mock(text_result: str = "hello world") -> MagicMock:
    """Return a fully wired ``outlines`` mock that returns *text_result*."""
    mock_outlines = MagicMock()

    # outlines.models.transformers(model_name, device=...) -> model object
    mock_model = MagicMock()
    mock_outlines.models.transformers.return_value = mock_model

    # outlines.generate.text(model) -> callable that returns the text
    mock_text_gen = MagicMock(return_value=text_result)
    mock_outlines.generate.text.return_value = mock_text_gen

    # outlines.generate.json(model, schema) -> callable that returns the text
    mock_json_gen = MagicMock(return_value=text_result)
    mock_outlines.generate.json.return_value = mock_json_gen

    return mock_outlines


def _make_guidance_mock(text_result: str = "hello world") -> MagicMock:
    """Return a fully wired ``guidance`` mock that returns *text_result*."""
    mock_guidance = MagicMock()

    # guidance.models.Transformers(model_name) -> lm object
    mock_lm = MagicMock()
    # lm + prompt + guidance.json(...) and lm + prompt must both produce an
    # object whose str() is text_result.
    mock_program = MagicMock()
    mock_program.__str__ = MagicMock(return_value=text_result)
    mock_lm.__add__ = MagicMock(return_value=mock_program)
    mock_program.__add__ = MagicMock(return_value=mock_program)
    mock_guidance.models.Transformers.return_value = mock_lm

    # guidance.json(schema=...) -> some constraint object (opaque to us)
    mock_guidance.json.return_value = MagicMock()

    return mock_guidance


def _outlines_patch(mock_outlines: MagicMock):
    """Return a ``patch.dict`` context that makes ``outlines`` importable."""
    return patch.dict(
        "sys.modules",
        {
            "outlines": mock_outlines,
            "outlines.generate": mock_outlines.generate,
            "outlines.models": mock_outlines.models,
        },
    )


def _guidance_patch(mock_guidance: MagicMock):
    """Return a ``patch.dict`` context that makes ``guidance`` importable."""
    return patch.dict(
        "sys.modules",
        {
            "guidance": mock_guidance,
            "guidance.models": mock_guidance.models,
        },
    )


def _remove_outlines_from_modules():
    """Return a ``patch.dict`` that hides ``outlines`` from ``sys.modules``."""
    return patch.dict(
        "sys.modules",
        {"outlines": None, "outlines.generate": None, "outlines.models": None},
    )


def _remove_guidance_from_modules():
    """Return a ``patch.dict`` that hides ``guidance`` from ``sys.modules``."""
    return patch.dict(
        "sys.modules",
        {"guidance": None, "guidance.models": None},
    )


# ===========================================================================
# OutlinesBackend tests
# ===========================================================================


class TestOutlinesBackendInit:
    """Tests 1-2: __init__ and default / custom attribute values."""

    def test_outlines_init_defaults(self) -> None:
        backend = OutlinesBackend()
        assert backend.model_name == "microsoft/Phi-3-mini-4k-instruct"
        assert backend.device == "cpu"
        assert backend.name == "outlines"

    def test_outlines_init_custom(self) -> None:
        backend = OutlinesBackend(model_name="mistralai/Mistral-7B-v0.1", device="cuda")
        assert backend.model_name == "mistralai/Mistral-7B-v0.1"
        assert backend.device == "cuda"


class TestOutlinesBackendCapabilities:
    """Tests 3-4: capability properties."""

    def test_outlines_supports_kv_cache_reuse_false(self) -> None:
        backend = OutlinesBackend()
        assert backend.supports_kv_cache_reuse is False

    def test_outlines_accuracy_loss_baseline_zero(self) -> None:
        backend = OutlinesBackend()
        assert backend.accuracy_loss_baseline == pytest.approx(0.0)


class TestOutlinesBackendGenerate:
    """Tests 5-8: generate() method."""

    @pytest.mark.asyncio
    async def test_outlines_generate_no_schema(self) -> None:
        """generate() without a schema uses the text generator."""
        mock_outlines = _make_outlines_mock(text_result="free text response")
        backend = OutlinesBackend(model_name="some/model", device="cpu")

        with _outlines_patch(mock_outlines):
            result = await backend.generate("Say hello")

        assert result == "free text response"
        mock_outlines.generate.text.assert_called_once()
        mock_outlines.generate.json.assert_not_called()

    @pytest.mark.asyncio
    async def test_outlines_generate_with_schema(self) -> None:
        """generate() with a schema dict uses the json generator."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        mock_outlines = _make_outlines_mock(text_result='{"name": "Alice"}')
        backend = OutlinesBackend()

        with _outlines_patch(mock_outlines):
            result = await backend.generate("Give me a name", schema=schema)

        assert result == '{"name": "Alice"}'
        mock_outlines.generate.json.assert_called_once_with(
            mock_outlines.models.transformers.return_value, schema
        )
        mock_outlines.generate.text.assert_not_called()

    @pytest.mark.asyncio
    async def test_outlines_generate_import_error(self) -> None:
        """generate() raises ImportError with install hint when outlines is absent."""
        backend = OutlinesBackend()

        with _remove_outlines_from_modules():
            with pytest.raises(ImportError, match="pip install outlines"):
                await backend.generate("prompt")

    @pytest.mark.asyncio
    async def test_outlines_generate_runtime_error(self) -> None:
        """generate() wraps generator exceptions as RuntimeError."""
        mock_outlines = _make_outlines_mock()
        # Make the text generator callable raise an exception
        mock_outlines.generate.text.return_value.side_effect = ValueError("GPU OOM")
        backend = OutlinesBackend(model_name="test/model")

        with _outlines_patch(mock_outlines):
            with pytest.raises(RuntimeError, match="OutlinesBackend generation error"):
                await backend.generate("prompt")


class TestOutlinesBackendStream:
    """Tests 9-10: stream() async generator."""

    @pytest.mark.asyncio
    async def test_outlines_stream_yields_output_events(self) -> None:
        """stream() yields StreamEvent objects: one per word + final complete."""
        mock_outlines = _make_outlines_mock(text_result="hello world foo")
        backend = OutlinesBackend()

        with _outlines_patch(mock_outlines):
            events = [e async for e in backend.stream("prompt")]

        # "hello world foo" splits into 3 words → 3 output events + 1 complete
        assert len(events) == 4

        output_events = [e for e in events if e.type == "output"]
        complete_events = [e for e in events if e.type == "complete"]

        assert len(output_events) == 3
        assert len(complete_events) == 1

        # All events must be StreamEvent instances with the correct backend tag
        for event in events:
            assert isinstance(event, StreamEvent)
            assert event.backend == "outlines"

        # The complete event must carry the full generated text
        assert complete_events[0].content == "hello world foo"

        # output events must carry individual word tokens
        tokens = [e.token for e in output_events]
        assert "hello " in tokens
        assert "world " in tokens
        assert "foo" in tokens  # last word has no trailing space

    @pytest.mark.asyncio
    async def test_outlines_stream_import_error(self) -> None:
        """stream() raises ImportError when outlines is missing."""
        backend = OutlinesBackend()

        with _remove_outlines_from_modules():
            with pytest.raises(ImportError, match="pip install outlines"):
                async for _ in backend.stream("prompt"):
                    pass


# ===========================================================================
# GuidanceBackend tests
# ===========================================================================


class TestGuidanceBackendInit:
    """Tests 11-12: __init__ and default / custom attribute values."""

    def test_guidance_init_defaults(self) -> None:
        backend = GuidanceBackend()
        assert backend.model_name == "gpt2"
        assert backend.backend_type == "transformers"
        assert backend.name == "guidance"

    def test_guidance_init_custom(self) -> None:
        backend = GuidanceBackend(model_name="mistralai/Mistral-7B-v0.1", backend_type="llamacpp")
        assert backend.model_name == "mistralai/Mistral-7B-v0.1"
        assert backend.backend_type == "llamacpp"


class TestGuidanceBackendCapabilities:
    """Tests 13-14: capability properties."""

    def test_guidance_supports_kv_cache_reuse_false(self) -> None:
        backend = GuidanceBackend()
        assert backend.supports_kv_cache_reuse is False

    def test_guidance_accuracy_loss_baseline(self) -> None:
        """GuidanceBackend reports a 5 % accuracy loss baseline."""
        backend = GuidanceBackend()
        assert backend.accuracy_loss_baseline == pytest.approx(0.05)


class TestGuidanceBackendGenerate:
    """Tests 15-18: generate() method."""

    @pytest.mark.asyncio
    async def test_guidance_generate_no_schema(self) -> None:
        """generate() without schema concatenates lm + prompt (no json constraint)."""
        mock_guidance = _make_guidance_mock(text_result="unconstrained output")
        backend = GuidanceBackend()

        with _guidance_patch(mock_guidance):
            result = await backend.generate("Say something")

        assert result == "unconstrained output"
        # guidance.json should NOT have been called because schema=None
        mock_guidance.json.assert_not_called()

    @pytest.mark.asyncio
    async def test_guidance_generate_with_schema(self) -> None:
        """generate() with schema calls guidance.json(schema=...) as a constraint."""
        schema = {"type": "object", "properties": {"answer": {"type": "integer"}}}
        mock_guidance = _make_guidance_mock(text_result='{"answer": 42}')
        backend = GuidanceBackend()

        with _guidance_patch(mock_guidance):
            result = await backend.generate("Give me a number", schema=schema)

        assert result == '{"answer": 42}'
        mock_guidance.json.assert_called_once_with(schema=schema)

    @pytest.mark.asyncio
    async def test_guidance_generate_import_error(self) -> None:
        """generate() raises ImportError with install hint when guidance is absent."""
        backend = GuidanceBackend()

        with _remove_guidance_from_modules():
            with pytest.raises(ImportError, match="pip install guidance"):
                await backend.generate("prompt")

    @pytest.mark.asyncio
    async def test_guidance_generate_runtime_error(self) -> None:
        """generate() wraps generation exceptions as RuntimeError."""
        mock_guidance = _make_guidance_mock()
        # Force an exception when str() is called on the program object
        mock_lm = mock_guidance.models.Transformers.return_value
        mock_program = MagicMock()
        mock_program.__str__ = MagicMock(side_effect=RuntimeError("CUDA error"))
        mock_lm.__add__ = MagicMock(return_value=mock_program)
        mock_program.__add__ = MagicMock(return_value=mock_program)
        backend = GuidanceBackend(model_name="test/model")

        with _guidance_patch(mock_guidance):
            with pytest.raises(RuntimeError, match="GuidanceBackend generation error"):
                await backend.generate("prompt")


class TestGuidanceBackendStream:
    """Tests 19-20: stream() async generator."""

    @pytest.mark.asyncio
    async def test_guidance_stream_yields_output_events(self) -> None:
        """stream() yields one StreamEvent per word plus a final complete event."""
        mock_guidance = _make_guidance_mock(text_result="one two three")
        backend = GuidanceBackend()

        with _guidance_patch(mock_guidance):
            events = [e async for e in backend.stream("prompt")]

        # "one two three" splits into 3 words → 3 output + 1 complete
        assert len(events) == 4

        output_events = [e for e in events if e.type == "output"]
        complete_events = [e for e in events if e.type == "complete"]

        assert len(output_events) == 3
        assert len(complete_events) == 1

        for event in events:
            assert isinstance(event, StreamEvent)
            assert event.backend == "guidance"

        assert complete_events[0].content == "one two three"

        tokens = [e.token for e in output_events]
        assert "one " in tokens
        assert "two " in tokens
        assert "three" in tokens  # last word has no trailing space

    @pytest.mark.asyncio
    async def test_guidance_stream_import_error(self) -> None:
        """stream() raises ImportError when guidance is missing."""
        backend = GuidanceBackend()

        with _remove_guidance_from_modules():
            with pytest.raises(ImportError, match="pip install guidance"):
                async for _ in backend.stream("prompt"):
                    pass


# ---------------------------------------------------------------------------
# Bonus edge-case tests
# ---------------------------------------------------------------------------


class TestOutlinesBackendStreamSingleWord:
    """Extra: stream with a single-word response produces 1 output + 1 complete."""

    @pytest.mark.asyncio
    async def test_outlines_stream_single_word(self) -> None:
        mock_outlines = _make_outlines_mock(text_result="yes")
        backend = OutlinesBackend()

        with _outlines_patch(mock_outlines):
            events = [e async for e in backend.stream("prompt")]

        assert len(events) == 2  # 1 output + 1 complete
        assert events[0].type == "output"
        assert events[0].token == "yes"  # last (and only) word — no trailing space
        assert events[1].type == "complete"
        assert events[1].content == "yes"


class TestGuidanceBackendStreamSingleWord:
    """Extra: stream with a single-word response produces 1 output + 1 complete."""

    @pytest.mark.asyncio
    async def test_guidance_stream_single_word(self) -> None:
        mock_guidance = _make_guidance_mock(text_result="no")
        backend = GuidanceBackend()

        with _guidance_patch(mock_guidance):
            events = [e async for e in backend.stream("prompt")]

        assert len(events) == 2
        assert events[0].type == "output"
        assert events[0].token == "no"
        assert events[1].type == "complete"
        assert events[1].content == "no"


class TestOutlinesBackendGenerateKvCacheIgnored:
    """Extra: kv_cache_prefix argument is silently ignored (no error raised)."""

    @pytest.mark.asyncio
    async def test_outlines_generate_kv_cache_prefix_ignored(self) -> None:
        mock_outlines = _make_outlines_mock(text_result="result")
        backend = OutlinesBackend()

        with _outlines_patch(mock_outlines):
            result = await backend.generate("prompt", kv_cache_prefix="sys:prefix")

        assert result == "result"


class TestGuidanceBackendGenerateKvCacheIgnored:
    """Extra: kv_cache_prefix argument is silently ignored (no error raised)."""

    @pytest.mark.asyncio
    async def test_guidance_generate_kv_cache_prefix_ignored(self) -> None:
        mock_guidance = _make_guidance_mock(text_result="result")
        backend = GuidanceBackend()

        with _guidance_patch(mock_guidance):
            result = await backend.generate("prompt", kv_cache_prefix="sys:prefix")

        assert result == "result"


class TestOutlinesBackendStreamWithSchema:
    """Extra: stream() uses the json generator when a schema is supplied."""

    @pytest.mark.asyncio
    async def test_outlines_stream_with_schema_uses_json_generator(self) -> None:
        schema = {"type": "object"}
        mock_outlines = _make_outlines_mock(text_result='{"ok": true}')
        backend = OutlinesBackend()

        with _outlines_patch(mock_outlines):
            events = [e async for e in backend.stream("prompt", schema=schema)]

        mock_outlines.generate.json.assert_called_once_with(
            mock_outlines.models.transformers.return_value, schema
        )
        mock_outlines.generate.text.assert_not_called()
        # Complete event should carry the json-constrained text
        complete = next(e for e in events if e.type == "complete")
        assert complete.content == '{"ok": true}'


class TestGuidanceBackendStreamWithSchema:
    """Extra: stream() calls guidance.json() when a schema is supplied."""

    @pytest.mark.asyncio
    async def test_guidance_stream_with_schema_calls_guidance_json(self) -> None:
        schema = {"type": "object", "properties": {"x": {"type": "number"}}}
        mock_guidance = _make_guidance_mock(text_result='{"x": 3.14}')
        backend = GuidanceBackend()

        with _guidance_patch(mock_guidance):
            events = [e async for e in backend.stream("prompt", schema=schema)]

        mock_guidance.json.assert_called_once_with(schema=schema)
        complete = next(e for e in events if e.type == "complete")
        assert complete.content == '{"x": 3.14}'
