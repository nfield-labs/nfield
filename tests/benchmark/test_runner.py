"""Runner tests — sweep orchestration, persistence, and re-scoring, with a mock.

No API: a deterministic mock adapter stands in for the model so the runner's
file layout, manifest, scoring, and failure handling are tested offline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from benchmark.adapters import AdapterOutput
from benchmark.datasets import LoadedDataset
from benchmark.runner import result_dir, run_sweep, score_existing

_SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}, "n": {"type": "integer"}}}
_GOLD = {"a": "hello", "n": 7}


def _dataset(gold: dict[str, Any] | None = None) -> LoadedDataset:
    return LoadedDataset(
        name="toy",
        schema=_SCHEMA,
        document="doc",
        gold=_GOLD if gold is None else gold,
    )


def _run(adapter, dataset, **kwargs):
    # Default the shared run budget to a fixed toy one for the tests.
    kwargs.setdefault("context_window", 8192)
    kwargs.setdefault("max_output_tokens", 2048)
    kwargs.setdefault("budget", "native")
    return run_sweep(adapter, dataset, model="groq/x", **kwargs)


@dataclass
class _MockAdapter:
    name: str = "mock"
    output: AdapterOutput = field(
        default_factory=lambda: AdapterOutput(
            data={"a": "hello", "n": 7},
            fields_total=2,
            fields_extracted=2,
            k=1,
            k_min=1,
            elapsed_seconds=0.5,
        )
    )

    def run(self, document, schema, *, model, context_window, max_output_tokens, instructions=""):
        return self.output


def test_run_sweep_writes_raw_scored_and_manifest(tmp_path):
    out_dir = tmp_path / "run"
    artifacts = _run(_MockAdapter(), _dataset(), seeds=3, out_dir=out_dir)
    assert artifacts.raw_path.exists()
    assert artifacts.scored_path is not None and artifacts.scored_path.exists()
    assert artifacts.manifest_path.exists()

    records = json.loads(artifacts.raw_path.read_text(encoding="utf-8"))
    assert len(records) == 3  # one record per seed
    first = records[0]
    assert first["value_accuracy"] == 1.0
    assert first["method"] == "mock"
    assert first["budget"] == "native"  # the shared budget is recorded on every row

    scored = json.loads(artifacts.scored_path.read_text(encoding="utf-8"))
    assert scored["runs"] == 3
    assert scored["value_accuracy_mean"] == 1.0
    assert scored["value_accuracy_std"] == 0.0

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["model"] == "groq/x"
    assert manifest["runs"][0]["fixture"] == "toy"
    assert manifest["runs"][0]["budget"] == "native"
    assert "library_version" in manifest


def test_coverage_only_fixture_is_not_scored(tmp_path):
    artifacts = _run(_MockAdapter(), _dataset(gold={}), seeds=1, out_dir=tmp_path / "cov")
    # An empty gold key means coverage-only: a raw sidecar but no scored file.
    assert artifacts.scored_path is None
    assert artifacts.raw_path.exists()


def test_failed_run_scores_as_miss_not_dropped(tmp_path):
    failing = _MockAdapter(
        output=AdapterOutput(
            data={}, fields_total=2, fields_extracted=0, k=0, k_min=0, error="429 rate limit"
        )
    )
    artifacts = _run(failing, _dataset(), seeds=1, out_dir=tmp_path / "fail")
    record = json.loads(artifacts.raw_path.read_text(encoding="utf-8"))[0]
    assert record["error"] == "429 rate limit"
    assert record["value_accuracy"] == 0.0  # in the denominator, scored 0 — never N/A


def test_score_existing_rescores_without_api(tmp_path):
    out_dir = tmp_path / "rescore"
    artifacts = _run(_MockAdapter(), _dataset(), seeds=1, out_dir=out_dir)
    report = score_existing(artifacts.raw_path, _dataset())
    assert report.value_accuracy == 1.0
    assert report.n_fields == 2


def test_two_budget_layout_one_manifest_at_root(tmp_path):
    """A run writes raw/scored under each budget subfolder but ONE manifest at root."""
    from benchmark import report

    run_root = tmp_path / "run"
    for budget in ("native", "constrained"):
        _run(
            _MockAdapter(),
            _dataset(),
            seeds=1,
            out_dir=run_root / budget,
            budget=budget,
            manifest_dir=run_root,
        )

    # Raw/scored live per budget; the manifest is shared at the run root.
    assert (run_root / "native" / "raw" / "mock_toy.json").exists()
    assert (run_root / "constrained" / "scored" / "mock_toy.json").exists()
    assert (run_root / "MANIFEST.json").exists()
    assert not (run_root / "native" / "MANIFEST.json").exists()

    manifest = json.loads((run_root / "MANIFEST.json").read_text(encoding="utf-8"))
    assert {entry["budget"] for entry in manifest["runs"]} == {"native", "constrained"}

    # collect_rows spans both budgets from the single run root.
    rows = report.collect_rows(run_root)
    assert {r.budget for r in rows} == {"native", "constrained"}


def test_result_dir_is_self_describing(tmp_path):
    path = result_dir("groq/llama-3.3-70b-versatile", "2026-06-09_14-30-05", root=tmp_path)
    assert path.name == "groq-llama-3.3-70b-versatile_2026-06-09_14-30-05"


def test_now_stamp_is_sortable_and_dir_safe():
    from benchmark.runner import _now_stamp

    stamp = _now_stamp()
    # YYYY-MM-DD_HH-MM-SS: no path separators or colons, lexically sortable.
    assert len(stamp) == 19 and "/" not in stamp and ":" not in stamp
    assert stamp[10] == "_" and stamp[4] == stamp[7] == "-"


def test_latest_stamp_picks_newest_run(tmp_path):
    from benchmark.runner import _latest_stamp

    model = "groq/llama-3.3-70b-versatile"
    assert _latest_stamp(model, root=tmp_path) is None
    for stamp in ("2026-06-09_09-00-00", "2026-06-10_08-00-00", "2026-06-10_07-00-00"):
        result_dir(model, stamp, root=tmp_path).mkdir(parents=True)
    assert _latest_stamp(model, root=tmp_path) == "2026-06-10_08-00-00"
