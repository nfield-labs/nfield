"""Direct-competitor head-to-head - nfield vs the real extraction libraries.

Separate from :mod:`benchmark.runners.runner` (orchestration-layer baselines: raw_prompt,
native_json, instructor, langchain) and :mod:`benchmark.runners.runner2` (nfield-only scale
fixtures). This runs nfield against the *direct* structured-extraction competitors
(LangStruct, ExtractThinker, ContextGem, LangExtract) on the standard fixtures,
each on the SAME hosted model and the SAME shared budget, so the only variable is
how each library decomposes/retrieves. Output mirrors the main runner:
``results/<model>_<stamp>/<budget>/{raw,scored}`` + MANIFEST + summary.csv.

    uv run python -m benchmark.runners.runner3
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from .. import datasets
from ..adapters.contextgem_adapter import ContextGemAdapter
from ..adapters.langextract_adapter import LangExtractAdapter
from ..adapters.langstruct_adapter import LangStructAdapter
from ..adapters.nfield_adapter import NfieldAdapter
from ..budget import BUDGET_MODES, resolve_budget
from ..figures import report
from .runner import _NFIELD_THROTTLE, _load_env, _now_stamp, result_dir, run_sweep

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..adapters import Adapter

_MODEL = "groq/llama-3.3-70b-versatile"
_SEEDS = 1

# nfield plus the direct competitors that run fairly on the shared Groq model.
# Each value is a zero-arg adapter factory (Adapter protocol). Add a competitor here
# once its adapter passes a live smoke test on the shared model.
ADAPTERS: dict[str, Callable[[], Adapter]] = {
    "nfield": lambda: NfieldAdapter(max_concurrent_calls=_NFIELD_THROTTLE),
    "langstruct": LangStructAdapter,
    "langextract": LangExtractAdapter,
    "contextgem": ContextGemAdapter,
}

# The standard competitor fixtures (same as benchmark.runners.runner).
_FIXTURES = datasets.available()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="benchmark.runners.runner3", description=__doc__)
    parser.add_argument(
        "--methods", default=",".join(ADAPTERS), help="comma-separated method names"
    )
    parser.add_argument("--fixtures", default=",".join(_FIXTURES), help="comma-separated fixtures")
    parser.add_argument(
        "--budgets", default=",".join(BUDGET_MODES), help="comma-separated budgets"
    )
    args = parser.parse_args(argv)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    fixtures = [f.strip() for f in args.fixtures.split(",") if f.strip()]
    budgets = [b.strip() for b in args.budgets.split(",") if b.strip()]

    _load_env()
    run_root = result_dir(_MODEL, _now_stamp())
    for budget in budgets:
        limits = resolve_budget(budget, _MODEL)
        for name in methods:
            factory = ADAPTERS[name]
            for fixture in fixtures:
                dataset = datasets.get(fixture).load()
                print(f"  [{budget}] {name} x {dataset.name} (seeds={_SEEDS}) ...", flush=True)
                run_sweep(
                    factory(),
                    dataset,
                    model=_MODEL,
                    seeds=_SEEDS,
                    out_dir=run_root / budget,
                    context_window=limits.context_window,
                    max_output_tokens=limits.max_output_tokens,
                    budget=budget,
                    manifest_dir=run_root,
                )

    rows = report.collect_rows(run_root)
    report.write_summary_csv(rows, run_root / "summary.csv")
    print("\n" + report.format_table(rows))
    print(f"results -> {run_root}")


if __name__ == "__main__":
    main()
