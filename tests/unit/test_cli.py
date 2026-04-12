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


def test_benchmark_help_exits_zero() -> None:
    result = runner.invoke(app, ["benchmark", "--help"])
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


# ---------------------------------------------------------------------------
# benchmark command — mocked BenchmarkHarness
# ---------------------------------------------------------------------------


def _make_benchmark_result(backend: str = "groq") -> MagicMock:
    r = MagicMock()
    r.backend = backend
    r.task = "gsm"
    r.direct_accuracy = 0.60
    r.ttf_accuracy = 0.75
    r.accuracy_delta = 0.15
    r.overhead_pct = 125.0
    return r


def _invoke_benchmark_mocked(args: list[str], tmp_path: str | None = None) -> object:
    """Invoke the benchmark CLI command with BenchmarkHarness fully mocked."""
    mock_result = _make_benchmark_result()
    mock_artifact_path = MagicMock()
    mock_artifact_path.__str__ = lambda self: "heatmap.png"

    mock_harness = MagicMock()
    mock_harness.run = AsyncMock(return_value=[mock_result])
    mock_harness.generate_artifacts = MagicMock(return_value={"heatmap": mock_artifact_path})

    # Use a temp dir if provided, otherwise let typer use "benchmark_results"
    base_args = args if tmp_path is None else [*args, "--output", str(tmp_path)]

    with patch(
        "formatshield.benchmark.harness.BenchmarkHarness",
        return_value=mock_harness,
    ):
        return runner.invoke(app, base_args)


def test_benchmark_command_exits_zero(tmp_path) -> None:
    result = _invoke_benchmark_mocked(
        ["benchmark", "--tasks", "gsm", "--backends", "groq"],
        tmp_path=str(tmp_path),
    )
    assert result.exit_code == 0


def test_benchmark_command_prints_results_table(tmp_path) -> None:
    result = _invoke_benchmark_mocked(
        ["benchmark", "--tasks", "gsm", "--backends", "groq"],
        tmp_path=str(tmp_path),
    )
    assert result.exit_code == 0
    # Should contain backend name in output
    assert "groq" in result.output


def test_benchmark_command_quick_flag(tmp_path) -> None:
    result = _invoke_benchmark_mocked(
        ["benchmark", "--tasks", "gsm", "--backends", "groq", "--quick"],
        tmp_path=str(tmp_path),
    )
    assert result.exit_code == 0


def test_benchmark_command_reproduce_paper_flag(tmp_path) -> None:
    result = _invoke_benchmark_mocked(
        ["benchmark", "--reproduce-paper"],
        tmp_path=str(tmp_path),
    )
    assert result.exit_code == 0
    assert "paper" in result.output.lower() or "reproduction" in result.output.lower()


def test_benchmark_command_multiple_tasks_and_backends(tmp_path) -> None:
    result = _invoke_benchmark_mocked(
        [
            "benchmark",
            "--tasks",
            "gsm,medical_ner",
            "--backends",
            "groq,ollama",
        ],
        tmp_path=str(tmp_path),
    )
    assert result.exit_code == 0


def test_benchmark_command_shows_artifacts(tmp_path) -> None:
    result = _invoke_benchmark_mocked(
        ["benchmark", "--tasks", "gsm", "--backends", "groq"],
        tmp_path=str(tmp_path),
    )
    assert result.exit_code == 0
    # Artifacts line should appear
    assert "Artifacts" in result.output or "heatmap" in result.output


def test_main_function_importable() -> None:
    from formatshield.cli import main

    assert callable(main)


def test_main_function_calls_app() -> None:
    from formatshield.cli import main

    with patch("formatshield.cli.app") as mock_app:
        main()
        mock_app.assert_called_once()
