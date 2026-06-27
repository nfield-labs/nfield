"""Scale benchmark - nfield on the large in-house fixtures (2.5k, 4k, 5.6k, and future).

Separate from :mod:`benchmark.runner` (which pits nfield against the competitor
baselines on the standard fixtures). These are big, record-structured documents that
stress nfield's N-field path, so only nfield runs, under both budgets. Output mirrors
the main runner: ``results/<model>_<stamp>/<budget>/{raw,scored}`` + MANIFEST.

    uv run python -m benchmark.runner2
"""

from __future__ import annotations

from . import report
from .adapters.nfield_adapter import NfieldAdapter
from .budget import BUDGET_MODES, resolve_budget
from .datasets import Dataset
from .runner import _load_env, _now_stamp, result_dir, run_sweep

_MODEL = "groq/llama-3.3-70b-versatile"
_SEEDS = 1

_FAITHFULNESS = (
    "Extract each field's value exactly as written for the correct record and section - "
    "keep all amounts, units, dates, codes, and identifiers; never summarize or infer. "
    "Leave a field null if the document does not state it."
)

# In-house large fixtures, registered here (not in the competitor registry). Add new
# scale fixtures to this tuple - each needs benchmark/datasets/real/<name>/.
_FIXTURES: tuple[Dataset, ...] = (
    Dataset(
        "clinical_registry",
        instructions=f"The document is a multi-center clinical patient registry. {_FAITHFULNESS}",
    ),
    Dataset(
        "cre_rent_roll",
        instructions=f"The document is a commercial real estate rent roll and lease abstract. {_FAITHFULNESS}",
    ),
    Dataset(
        "financial_consolidation",
        instructions=f"The document is a consolidated multi-subsidiary financial annual report. {_FAITHFULNESS}",
    ),
)


def _nk_model(model: str) -> str:
    # Tag the provider with "nk" so this scale run lands in its own results folder
    # (e.g. groq-nk-llama-...), separate from the competitor sweep's groq-llama-...
    provider, _, name = model.partition("/")
    return f"{provider}-nk/{name}" if name else f"{model}-nk"


def main() -> None:
    _load_env()
    run_root = result_dir(_nk_model(_MODEL), _now_stamp())
    for budget in BUDGET_MODES:
        limits = resolve_budget(budget, _MODEL)
        for fixture in _FIXTURES:
            dataset = fixture.load()
            print(f"  [{budget}] nfield x {dataset.name} (seeds={_SEEDS}) ...", flush=True)
            run_sweep(
                NfieldAdapter(),
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
