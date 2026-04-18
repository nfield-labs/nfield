"""Tests for formatshield.cli — covers the Typer app commands."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from formatshield.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# version command
# ---------------------------------------------------------------------------


def test_version_command_exits_zero() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0


def test_version_command_prints_version() -> None:
    result = runner.invoke(app, ["version"])
    assert "FormatShield" in result.output


# ---------------------------------------------------------------------------
# --help flags
# ---------------------------------------------------------------------------


def test_app_help_exits_zero() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0



def test_generate_help_exits_zero() -> None:
    result = runner.invoke(app, ["generate", "--help"])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# generate — invalid schema
# ---------------------------------------------------------------------------


def test_generate_invalid_schema_exits_nonzero() -> None:
    result = runner.invoke(app, ["generate", "hello", "--schema", "not-json"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# generate — mocked FormatShield.generate
# ---------------------------------------------------------------------------


def _make_mock_result() -> MagicMock:
    mock_result = MagicMock()
    mock_result.output = '{"answer": "test"}'
    mock_result.routing.strategy = "direct"
    mock_result.complexity_score = 0.3
    mock_result.routing.expected_accuracy_delta = 0.0
    mock_result.routing.confidence = 0.9
    mock_result.routing.explanation = "simple"
    mock_result.latency_ms = 100.0
    mock_result.thinking = None
    return mock_result


def _invoke_generate_mocked(args: list[str], thinking: str | None = None) -> object:
    """Invoke the generate CLI command with FormatShield fully mocked out."""
    mock_result = _make_mock_result()
    if thinking is not None:
        mock_result.thinking = thinking

    mock_backend = MagicMock()
    with patch("formatshield.core._build_backend", return_value=mock_backend):
        with patch(
            "formatshield.core.FormatShield.generate",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
                return runner.invoke(app, args)


def test_generate_command_exits_zero_with_mock() -> None:
    result = _invoke_generate_mocked(
        ["generate", "What is 2+2?", "--model", "groq/llama-3.1-70b-versatile"]
    )
    assert result.exit_code == 0


def test_generate_command_prints_output_with_mock() -> None:
    result = _invoke_generate_mocked(
        ["generate", "What is 2+2?", "--model", "groq/llama-3.1-70b-versatile"]
    )
    assert result.exit_code == 0
    assert "answer" in result.output or "Output" in result.output


def test_generate_debug_flag_shows_routing_trace() -> None:
    result = _invoke_generate_mocked(
        [
            "generate",
            "What is 2+2?",
            "--model",
            "groq/llama-3.1-70b-versatile",
            "--debug",
        ]
    )
    assert result.exit_code == 0
    # Debug flag should cause routing trace to appear
    assert "Route" in result.output or "Routing" in result.output


def test_generate_expose_thinking_with_thinking_content() -> None:
    result = _invoke_generate_mocked(
        [
            "generate",
            "What is 2+2?",
            "--model",
            "groq/llama-3.1-70b-versatile",
            "--expose-thinking",
        ],
        thinking="Let me think about this...",
    )
    assert result.exit_code == 0
    assert "think" in result.output.lower() or "Thinking" in result.output


def test_generate_expose_thinking_without_thinking_content() -> None:
    result = _invoke_generate_mocked(
        [
            "generate",
            "What is 2+2?",
            "--model",
            "groq/llama-3.1-70b-versatile",
            "--expose-thinking",
        ],
        thinking=None,
    )
    # Should not crash when thinking is None
    assert result.exit_code == 0


def test_generate_with_valid_json_schema() -> None:
    result = _invoke_generate_mocked(
        [
            "generate",
            "What is 2+2?",
            "--model",
            "groq/llama-3.1-70b-versatile",
            "--schema",
            '{"type": "object", "properties": {"answer": {"type": "string"}}}',
        ]
    )
    assert result.exit_code == 0



def test_main_function_importable() -> None:
    from formatshield.cli import main

    assert callable(main)


def test_main_function_calls_app() -> None:
    from formatshield.cli import main

    with patch("formatshield.cli.app") as mock_app:
        main()
        mock_app.assert_called_once()
