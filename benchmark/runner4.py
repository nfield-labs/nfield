"""Closed-book sweep — every method fills a schema from knowledge, no document.

A thin orchestrator that mirrors ``runner.py``: it loads the closed-book fixtures through
the ``datasets`` module (``get_closed_book``) and reuses ``run_sweep`` + the budget profiles
+ ``report``, so results land in the identical layout:

    results/<model>_closed-book_<stamp>/
        MANIFEST.json
        summary.csv
        native/{raw,scored}/<method>_<fixture>.json
        constrained/{raw,scored}/<method>_<fixture>.json

Fixtures are four reference domains of increasing size (elements, countries, pokemon,
airports), each with a knowledge gradient so abstention separates from confident guessing;
they are materialised by ``datasets/closed_book/build_fixtures.py``. nfield runs closed-book;
the baselines generate from the empty-document + instruction prompt. Manual, budgeted tool —
the sweep costs live API calls. Run:

    uv run python -m benchmark.runner4
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import TYPE_CHECKING

from . import datasets, report
from .adapters.instructor_adapter import InstructorAdapter
from .adapters.langchain_adapter import LangChainAdapter
from .adapters.native_json_adapter import NativeJsonAdapter
from .adapters.nfield_adapter import NfieldAdapter
from .adapters.raw_prompt_adapter import RawPromptAdapter
from .budget import BUDGET_MODES, resolve_budget
from .runner import run_sweep

if TYPE_CHECKING:
    from collections.abc import Callable

    from .adapters import Adapter

__all__ = ["ADAPTERS", "main"]

_MODEL = "groq/llama-3.3-70b-versatile"
_RESULTS = Path(__file__).resolve().parent / "results"

# nfield runs closed-book; the baselines take the same empty-document + instruction prompt.
ADAPTERS: dict[str, Callable[[], Adapter]] = {
    "nfield": lambda: NfieldAdapter(closed_book=True),
    "instructor": InstructorAdapter,
    "langchain": LangChainAdapter,
    "native_json": NativeJsonAdapter,
    "raw_prompt": RawPromptAdapter,
}


def _load_env() -> None:
    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def main(*, seeds: int = 1) -> None:
    """Sweep every method over every closed-book fixture under both budget profiles."""
    _load_env()
    if not os.environ.get("GROQ_API_KEY"):
        print("No GROQ_API_KEY in .env — cannot run live.")
        return
    fixtures = datasets.closed_book_available()
    if not fixtures:
        print("No closed-book fixtures; run datasets/closed_book/build_fixtures.py first.")
        return

    stamp = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    run_root = _RESULTS / f"{_MODEL.replace('/', '-')}_closed-book_{stamp}"

    for budget in BUDGET_MODES:
        limits = resolve_budget(budget, _MODEL)
        for name in fixtures:
            dataset = datasets.get_closed_book(name).load()
            for method, factory in ADAPTERS.items():
                print(f"  [{budget}] {method} x {name} (seeds={seeds}) ...", flush=True)
                run_sweep(
                    factory(),
                    dataset,
                    model=_MODEL,
                    seeds=seeds,
                    out_dir=run_root / budget,
                    context_window=limits.context_window,
                    max_output_tokens=limits.max_output_tokens,
                    budget=budget,
                    manifest_dir=run_root,
                )

    rows = report.collect_rows(run_root)
    report.write_summary_csv(rows, run_root / "summary.csv")
    report.plot_va_vs_n(rows, run_root / "plots" / "va_vs_n.png")
    print("\n" + report.format_table(rows))
    print(f"\nresults -> {run_root}")


if __name__ == "__main__":
    main()
