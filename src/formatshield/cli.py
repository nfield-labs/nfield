"""FormatShield CLI — formatshield generate / formatshield benchmark"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="formatshield",
    help="Route LLM outputs intelligently. Measure what structured generation costs you.",
    rich_markup_mode="rich",
    add_completion=False,
)
console = Console()


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="The prompt to send to the model"),
    model: str = typer.Option(
        "groq/llama-3.1-70b-versatile",
        "--model",
        "-m",
        help="Model in 'provider/model' format",
    ),
    schema: str | None = typer.Option(
        None,
        "--schema",
        "-s",
        help="JSON schema string for structured output",
    ),
    debug: bool = typer.Option(False, "--debug", "-d", help="Show routing trace"),
    expose_thinking: bool = typer.Option(False, "--expose-thinking", help="Show thinking text"),
) -> None:
    """Generate structured output with automatic TTF routing."""
    from formatshield.core import FormatShield

    schema_dict = None
    if schema:
        try:
            schema_dict = json.loads(schema)
        except json.JSONDecodeError as exc:
            console.print(f"[red]Error: invalid JSON schema: {schema}[/red]")
            raise typer.Exit(1) from exc

    shield = FormatShield(
        model=model,
        debug=debug,
        expose_thinking=expose_thinking,
    )

    async def _run() -> None:
        result = await shield.generate(prompt, schema=schema_dict)

        if debug:
            console.print(
                Panel(
                    f"[bold]Route:[/bold] {result.routing.strategy}\n"
                    f"[bold]Complexity:[/bold] {result.complexity_score:.3f}\n"
                    f"[bold]Expected delta:[/bold] {result.routing.expected_accuracy_delta:+.3f}\n"
                    f"[bold]Confidence:[/bold] {result.routing.confidence:.2f}\n"
                    f"[bold]Latency:[/bold] {result.latency_ms:.0f}ms\n"
                    f"[bold]Explanation:[/bold] {result.routing.explanation}",
                    title="[yellow]FormatShield Routing Trace[/yellow]",
                    border_style="yellow",
                )
            )

        if expose_thinking and result.thinking:
            console.print(
                Panel(
                    result.thinking,
                    title="[blue]Thinking (Pass 1)[/blue]",
                    border_style="blue",
                )
            )

        console.print(
            Panel(
                result.output,
                title="[green]Output[/green]",
                border_style="green",
            )
        )

    asyncio.run(_run())


@app.command()
def benchmark(
    tasks: str = typer.Option("gsm", "--tasks", "-t", help="Comma-separated task names"),
    backends: str = typer.Option("groq", "--backends", "-b", help="Comma-separated backend names"),
    quick: bool = typer.Option(False, "--quick", "-q", help="Quick run (fewer examples)"),
    output: str = typer.Option("benchmark_results", "--output", "-o", help="Output directory"),
    reproduce_paper: bool = typer.Option(
        False, "--reproduce-paper", help="Reproduce paper results"
    ),
) -> None:
    """Run FormatShield benchmark across backends and generate paper artifacts."""
    from formatshield.benchmark.harness import BenchmarkHarness

    task_list = [t.strip() for t in tasks.split(",")]
    backend_list = [b.strip() for b in backends.split(",")]

    if reproduce_paper:
        task_list = ["gsm", "medical_ner", "template_fill"]
        backend_list = ["groq", "ollama"]
        console.print("[yellow]Running paper reproduction benchmark...[/yellow]")

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]FormatShield Benchmark[/bold]")
    console.print(f"Tasks: {', '.join(task_list)}")
    console.print(f"Backends: {', '.join(backend_list)}")
    console.print(f"Quick mode: {quick}")
    console.print(f"Output: {output_dir.absolute()}")
    console.print()

    harness = BenchmarkHarness(output_dir=output_dir)

    default_models = {
        "groq": "groq/llama-3.1-70b-versatile",
        "ollama": "ollama/llama3.1:70b",
        "openrouter": "openrouter/meta-llama/llama-3.1-70b-instruct",
        "vllm": "vllm/meta-llama/Llama-3-70b-Instruct",
    }
    models = {b: default_models.get(b, f"{b}/default") for b in backend_list}

    async def _run() -> None:
        results = await harness.run(
            tasks=task_list,
            backends=backend_list,
            models=models,
            quick=quick,
        )
        artifacts = harness.generate_artifacts(results)

        # Print summary table
        table = Table(title="Benchmark Results Summary")
        table.add_column("Backend", style="cyan")
        table.add_column("Task", style="magenta")
        table.add_column("Direct Acc", justify="right")
        table.add_column("TTF Acc", justify="right")
        table.add_column("Delta", justify="right")
        table.add_column("Overhead", justify="right")

        for r in results:
            delta_color = "green" if r.accuracy_delta > 0 else "red"
            table.add_row(
                r.backend,
                r.task,
                f"{r.direct_accuracy:.3f}",
                f"{r.ttf_accuracy:.3f}",
                f"[{delta_color}]{r.accuracy_delta:+.3f}[/{delta_color}]",
                f"{r.overhead_pct:.1f}%",
            )

        console.print(table)
        console.print(f"\n[green]Artifacts written to {output_dir.absolute()}[/green]")
        for name, path in artifacts.items():
            console.print(f"  [dim]{name}:[/dim] {path}")

    asyncio.run(_run())


@app.command()
def version() -> None:
    """Print FormatShield version."""
    import formatshield

    console.print(f"FormatShield {formatshield.__version__}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
