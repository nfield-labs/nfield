"""FormatShield CLI — formatshield generate / score / batch"""

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
    policy_default: bool = typer.Option(
        False,
        "--policy-default",
        help="Enable built-in default policy engine",
    ),
    policy_max_prompt_chars: int | None = typer.Option(
        None,
        "--policy-max-prompt-chars",
        help="Block prompts longer than this length when policy-default is enabled",
    ),
    policy_deny: list[str] | None = typer.Option(  # noqa: B008
        None,
        "--policy-deny",
        help="Deny prompt containing keyword (repeatable, policy-default only)",
    ),
    policy_profile: str = typer.Option(
        "balanced",
        "--policy-profile",
        help="Policy profile: strict, balanced, permissive",
    ),
    adaptive_confidence: bool = typer.Option(
        False,
        "--adaptive-confidence",
        help="Escalate low-confidence direct routes to TTF",
    ),
    adaptive_confidence_threshold: float = typer.Option(
        0.55,
        "--adaptive-confidence-threshold",
        help="Threshold below which direct routes escalate when adaptive confidence is enabled",
    ),
    audit_file: str | None = typer.Option(
        None,
        "--audit-file",
        help="Write tamper-evident audit events to this NDJSON file",
    ),
) -> None:
    """Generate structured output with automatic TTF routing."""
    from formatshield.core import FormatShield
    from formatshield.governance.policy import DefaultPolicyEngine
    from formatshield.observability.audit_log import FileAuditLogger

    schema_dict = None
    if schema:
        try:
            schema_dict = json.loads(schema)
        except json.JSONDecodeError as exc:
            console.print(f"[red]Error: invalid JSON schema: {schema}[/red]")
            raise typer.Exit(1) from exc

    audit_logger = FileAuditLogger(audit_file) if audit_file else None

    shield = FormatShield(
        model=model,
        debug=debug,
        expose_thinking=expose_thinking,
        policy_engine=(
            None
            if not policy_default
            else DefaultPolicyEngine.from_profile(
                policy_profile,
                max_prompt_chars=policy_max_prompt_chars,
                deny_keywords=tuple(policy_deny or ()),
            )
        ),
        adaptive_confidence=adaptive_confidence,
        adaptive_confidence_threshold=adaptive_confidence_threshold,
        audit_logger=audit_logger,
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

audit_app = typer.Typer(
    name="audit",
    help="Inspect and verify tamper-evident audit logs.",
    add_completion=False,
)
app.add_typer(audit_app, name="audit")


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


@audit_app.command("verify")
def audit_verify(
    audit_file: str = typer.Argument(..., help="Path to audit NDJSON file"),
) -> None:
    """Verify the hash-chain integrity of an audit file."""
    from formatshield.observability.audit_log import FileAuditLogger

    logger = FileAuditLogger(audit_file)
    events = logger.events()
    chain_valid = logger.verify_chain()

    event_type_counts: dict[str, int] = {}
    for event in events:
        event_type_counts[event.event_type] = event_type_counts.get(event.event_type, 0) + 1

    type_rows = "\n".join(
        f"  - {event_type}: {count}"
        for event_type, count in sorted(event_type_counts.items(), key=lambda item: item[0])
    )
    if not type_rows:
        type_rows = "  - (no events)"

    summary = (
        f"[bold]Audit file:[/bold] {audit_file}\n"
        f"[bold]Event count:[/bold] {len(events)}\n"
        f"[bold]Chain valid:[/bold] {'yes' if chain_valid else 'no'}\n"
        f"[bold]Last hash:[/bold] {events[-1].event_hash if events else 'GENESIS'}\n"
        f"[bold]Event types:[/bold]\n{type_rows}"
    )

    border_style = "green" if chain_valid else "red"
    title = "[green]Audit Verification[/green]" if chain_valid else "[red]Audit Verification[/red]"
    console.print(Panel(summary, title=title, border_style=border_style))
    if not chain_valid:
        raise typer.Exit(1)


@audit_app.command("manifest")
def audit_manifest(
    audit_file: str = typer.Argument(..., help="Path to audit NDJSON file"),
    output: str = typer.Option(
        "audit-manifest.json",
        "--output",
        "-o",
        help="Path to write manifest JSON",
    ),
    signing_key: str | None = typer.Option(
        None,
        "--signing-key",
        help="Optional HMAC signing key for manifest integrity",
        envvar="FORMATSHIELD_AUDIT_SIGNING_KEY",
    ),
    signing_key_id: str | None = typer.Option(
        None,
        "--signing-key-id",
        help="Optional identifier for the signing key used in the manifest",
        envvar="FORMATSHIELD_AUDIT_SIGNING_KEY_ID",
    ),
) -> None:
    """Create a portable integrity manifest for an audit file."""
    from formatshield.observability.audit_log import write_audit_manifest

    manifest = write_audit_manifest(
        audit_path=audit_file,
        manifest_path=output,
        signing_key=signing_key,
        signing_key_id=signing_key_id,
    )

    console.print(
        Panel(
            f"[bold]Manifest written:[/bold] {output}\n"
            f"[bold]Event count:[/bold] {manifest.event_count}\n"
            f"[bold]Chain valid:[/bold] {'yes' if manifest.chain_valid else 'no'}\n"
            f"[bold]Audit checksum:[/bold] {manifest.audit_sha256}\n"
            f"[bold]Signing key id:[/bold] {manifest.signature_key_id or 'n/a'}\n"
            f"[bold]Signed:[/bold] {'yes' if manifest.signature else 'no'}",
            title="[cyan]Audit Manifest[/cyan]",
            border_style="cyan",
        )
    )


@audit_app.command("verify-manifest")
def audit_verify_manifest(
    audit_file: str = typer.Argument(..., help="Path to audit NDJSON file"),
    manifest_file: str = typer.Argument(..., help="Path to audit manifest JSON"),
    signing_key: str | None = typer.Option(
        None,
        "--signing-key",
        help="HMAC signing key when verifying signed manifests",
        envvar="FORMATSHIELD_AUDIT_SIGNING_KEY",
    ),
    expected_signing_key_id: str | None = typer.Option(
        None,
        "--expected-signing-key-id",
        help="Expected signing key id recorded in the manifest",
        envvar="FORMATSHIELD_AUDIT_SIGNING_KEY_ID",
    ),
) -> None:
    """Verify an audit file against a previously exported manifest."""
    from formatshield.observability.audit_log import verify_audit_manifest

    valid, issues, manifest = verify_audit_manifest(
        audit_path=audit_file,
        manifest_path=manifest_file,
        signing_key=signing_key,
        expected_signing_key_id=expected_signing_key_id,
    )

    if valid and manifest is not None:
        console.print(
            Panel(
                f"[bold]Manifest:[/bold] {manifest_file}\n"
                f"[bold]Audit file:[/bold] {audit_file}\n"
                f"[bold]Event count:[/bold] {manifest.event_count}\n"
                f"[bold]Signing key id:[/bold] {manifest.signature_key_id or 'n/a'}\n"
                f"[bold]Result:[/bold] valid",
                title="[green]Manifest Verification[/green]",
                border_style="green",
            )
        )
        return

    issue_rows = "\n".join(f"  - {issue}" for issue in issues) if issues else "  - unknown failure"
    console.print(
        Panel(
            f"[bold]Manifest:[/bold] {manifest_file}\n"
            f"[bold]Audit file:[/bold] {audit_file}\n"
            f"[bold]Issues:[/bold]\n{issue_rows}",
            title="[red]Manifest Verification Failed[/red]",
            border_style="red",
        )
    )
    raise typer.Exit(1)


@app.command()
def version() -> None:
    """Print FormatShield version."""
    import formatshield

    console.print(f"FormatShield {formatshield.__version__}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
