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
        "groq/llama-3.3-70b-versatile",
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
        "groq": "groq/llama-3.3-70b-versatile",
        "ollama": "ollama/llama3.1:70b",
        "openrouter": "openrouter/meta-llama/llama-3.1-70b-instruct",
        "vllm": "vllm/meta-llama/Llama-3-70b-Instruct",
    }
    models = {b: default_models.get(b, f"{b}/default") for b in backend_list}

    # Build real backend instances from available API keys
    import os

    def _strip_prefix(model_str: str) -> str:
        return model_str.split("/", 1)[1] if "/" in model_str else model_str

    backend_objects: dict[str, object] = {}
    for backend_name in backend_list:
        if backend_name == "groq":
            api_key = os.environ.get("GROQ_API_KEY", "")
            if api_key:
                from formatshield.backends.groq_backend import GroqBackend

                backend_objects["groq"] = GroqBackend(
                    api_key=api_key, model=_strip_prefix(models[backend_name])
                )
                console.print(f"[green]Using real Groq backend: {models[backend_name]}[/green]")
            else:
                console.print(
                    "[yellow]GROQ_API_KEY not set — using DryRunBackend for groq[/yellow]"
                )
        elif backend_name == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if api_key:
                from formatshield.backends.openai_backend import OpenAIBackend

                backend_objects["openai"] = OpenAIBackend(
                    api_key=api_key, model=_strip_prefix(models[backend_name])
                )
        elif backend_name == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if api_key:
                from formatshield.backends.anthropic_backend import AnthropicBackend

                backend_objects["anthropic"] = AnthropicBackend(
                    api_key=api_key, model=_strip_prefix(models[backend_name])
                )
        elif backend_name == "dryrun":
            from formatshield.backends.dryrun_backend import DryRunBackend

            backend_objects["dryrun"] = DryRunBackend()

    async def _run() -> None:
        results = await harness.run(
            tasks=task_list,
            backends=backend_list,
            models=models,
            quick=quick,
            backend_objects=backend_objects if backend_objects else None,
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
def score(
    prompt: str = typer.Argument(..., help="Prompt to score"),
    model: str = typer.Option("groq/llama-3.3-70b-versatile", "--model", "-m"),
    schema: str | None = typer.Option(None, "--schema", "-s", help="JSON schema string"),
) -> None:
    """Show the complexity score and routing decision for a prompt."""
    from formatshield.core import FormatShield

    schema_dict = None
    if schema:
        try:
            schema_dict = json.loads(schema)
        except json.JSONDecodeError as exc:
            console.print("[red]Error: invalid JSON schema[/red]")
            raise typer.Exit(1) from exc

    shield = FormatShield(model=model)
    features = shield._scorer.score(prompt, schema=schema_dict, model_id=model)
    complexity = shield._scorer.compute_score(features)
    decision = shield._oracle.predict(features, shield.backend_name, model)

    console.print(
        Panel(
            f"[bold]Complexity Score:[/bold] {complexity:.3f}\n"
            f"[bold]Route:[/bold] {decision.strategy}\n"
            f"[bold]Expected Delta:[/bold] {decision.expected_accuracy_delta:+.3f}\n"
            f"[bold]Confidence:[/bold] {decision.confidence:.2f}\n"
            f"[bold]Schema Depth:[/bold] {features.schema_depth}\n"
            f"[bold]Reasoning Ops:[/bold] {features.required_reasoning_ops}\n"
            f"[bold]Length Bucket:[/bold] {features.prompt_length_bucket}\n"
            f"[bold]Explanation:[/bold] {decision.explanation}",
            title="[cyan]FormatShield Complexity Score[/cyan]",
            border_style="cyan",
        )
    )


# ---------------------------------------------------------------------------
# Batch command group
# ---------------------------------------------------------------------------

batch_app = typer.Typer(
    name="batch",
    help="Submit and manage large-scale batch generation jobs.",
    add_completion=False,
)
app.add_typer(batch_app, name="batch")


@batch_app.command("submit")
def batch_submit(
    prompts: list[str] = typer.Argument(..., help="One or more prompt strings"),  # noqa: B008
    model: str = typer.Option(
        "groq/llama-3.3-70b-versatile",
        "--model",
        "-m",
        help="Model in 'provider/model' format",
    ),
    schema: str | None = typer.Option(None, "--schema", "-s", help="JSON schema string"),
    output: str | None = typer.Option(None, "--output", "-o", help="Save results to JSON file"),
    concurrency: int = typer.Option(5, "--concurrency", "-c", help="Max concurrent requests"),
) -> None:
    """Submit a batch of prompts for generation."""
    from formatshield.batch.processor import BatchProcessor

    schema_dict: dict | None = None
    if schema:
        try:
            schema_dict = json.loads(schema)
        except json.JSONDecodeError as exc:
            console.print("[red]Error: invalid JSON schema[/red]")
            raise typer.Exit(1) from exc

    processor = BatchProcessor(
        model=model,
        response_model=schema_dict,
        max_concurrency=concurrency,
    )

    async def _run() -> None:
        console.print(f"[bold]Submitting {len(prompts)} prompt(s)...[/bold]")
        job = await processor.submit(prompts)
        results = await processor.results(job.job_id)

        console.print(
            Panel(
                f"[bold]Job ID:[/bold] {job.job_id}\n"
                f"[bold]Status:[/bold] {job.status.value}\n"
                f"[bold]Total:[/bold] {job.request_count}\n"
                f"[bold]Completed:[/bold] {job.completed_count}\n"
                f"[bold]Failed:[/bold] {job.failed_count}",
                title="[green]Batch Job Complete[/green]",
                border_style="green",
            )
        )

        from formatshield.batch.processor import BatchError as _BatchError
        from formatshield.batch.processor import BatchSuccess as _BatchSuccess

        data: list[dict] = []
        for r in results:
            if isinstance(r, _BatchSuccess):
                data.append({"custom_id": r.custom_id, "result": r.result, "success": True})
            elif isinstance(r, _BatchError):
                data.append({"custom_id": r.custom_id, "error": r.error_message, "success": False})

        if output:
            Path(output).write_text(json.dumps(data, indent=2))
            console.print(f"[dim]Results written to {output}[/dim]")
        else:
            console.print(json.dumps(data, indent=2))

    asyncio.run(_run())


@batch_app.command("status")
def batch_status(
    job_id: str = typer.Argument(..., help="Job ID returned by batch submit"),
) -> None:
    """Check the status of a batch job (in-memory only)."""
    msg = f"[yellow]Job status lookup for '{job_id}' requires a live processor instance.[/yellow]"
    console.print(msg)
    console.print("[dim]Tip: Use batch submit --output results.json to persist results.[/dim]")


@batch_app.command("results")
def batch_results(
    results_file: str = typer.Argument(..., help="Path to results JSON file from batch submit"),
) -> None:
    """Display results from a saved batch results file."""
    path = Path(results_file)
    if not path.exists():
        console.print(f"[red]File not found: {results_file}[/red]")
        raise typer.Exit(1)

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        console.print(f"[red]Invalid JSON in {results_file}[/red]")
        raise typer.Exit(1) from exc

    table = Table(title=f"Batch Results — {results_file}")
    table.add_column("ID", style="cyan")
    table.add_column("Success", justify="center")
    table.add_column("Result / Error", style="white")

    for item in data:
        success = item.get("success", False)
        status_str = "[green]✓[/green]" if success else "[red]✗[/red]"
        content = item.get("result", item.get("error", ""))
        if isinstance(content, str) and len(content) > 60:
            content = content[:57] + "..."
        table.add_row(str(item.get("custom_id", "")), status_str, str(content))

    console.print(table)


@app.command()
def version() -> None:
    """Print FormatShield version."""
    import formatshield

    console.print(f"FormatShield {formatshield.__version__}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
