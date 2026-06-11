"""Adversarial review tests (agent-reviewer, rev 26-benchmark).

These probe edge cases the original suite did not cover. Tests that PASS lock in
correct behaviour; tests marked ``xfail(strict=True)`` document a real gap the
coder must close (they will turn into failures the moment the gap is fixed).
"""

from __future__ import annotations

from benchmark.adapters import AdapterOutput
from benchmark.datasets import LoadedDataset
from benchmark.runner import run_sweep
from benchmark.score import Outcome, score


def _schema(properties: dict) -> dict:
    return {"type": "object", "properties": properties}


# --- Correct behaviour we want to lock in -----------------------------------


def test_zero_int_is_correct_not_omission():
    """A legitimate 0 must not be mis-bucketed as empty/omission."""
    report = score({"n": 0}, {"n": 0}, _schema({"n": {"type": "integer"}}))
    assert report.outcomes[Outcome.CORRECT] == 1
    assert report.value_accuracy == 1.0


def test_false_bool_is_correct_not_omission():
    """A legitimate False must not be mis-bucketed as empty/omission."""
    report = score({"b": False}, {"b": False}, _schema({"b": {"type": "boolean"}}))
    assert report.outcomes[Outcome.CORRECT] == 1


def test_buckets_total_n_with_mixed_outcomes():
    schema = _schema({f"f{i}": {"type": "string"} for i in range(5)})
    gold = {"f0": "a", "f1": "b", "f2": None, "f3": "d", "f4": "e"}
    extracted = {"f0": "a", "f1": "WRONG", "f2": "halluc", "f3": {"x": 1}}
    report = score(extracted, gold, schema)
    assert sum(report.outcomes.values()) == report.n_fields == 5


# --- HIGH: array reordering penalty is applied but undisclosed ----------------


def test_reordered_array_same_set_is_penalised_positionally():
    """Same set, different order -> 0% under positional matching.

    The design (4.1) mandates the reorder penalty be *disclosed*. The scorer
    silently applies an ordered match (item_N vs item_N). This test documents
    the behaviour; if a set-match or disclosed policy lands, update it.
    """
    schema = _schema({"xs": {"type": "array", "items": {"type": "string"}}})
    report = score({"xs": ["B", "A"]}, {"xs.item_0": "A", "xs.item_1": "B"}, schema)
    assert report.value_accuracy == 0.0
    assert report.outcomes[Outcome.ACCURACY] == 2


# --- HIGH: a whole-run API failure is mislabelled as model omission ----------


def test_total_call_failure_is_credited_to_call_failed(monkeypatch):
    """A pure API failure must surface as call-failed, not model omission.

    Exercises the real adapter: when ``nfield()`` raises (timeout / 429), every
    targeted field was lost to the call, so the adapter must report them all as
    call-failed (design §4.3 / §7), not leave the count at zero.
    """
    from benchmark.adapters.nfield_adapter import NfieldAdapter

    def boom(*_args, **_kwargs):
        raise ConnectionError("timed out")

    monkeypatch.setattr("formatshield.nfield", boom)
    schema = _schema({"a": {"type": "string"}, "b": {"type": "string"}})
    output = NfieldAdapter().run(
        "doc", schema, model="groq/x", context_window=8192, max_output_tokens=2048
    )
    assert output.failed
    assert output.fields_extracted == 0
    assert output.call_failed == 2  # both targeted fields lost to the call, not the model
    # And the score carries it through as its own category, never as omission attribution.
    report = score(output.data, {"a": "x", "b": "y"}, schema, call_failed=output.call_failed)
    assert report.call_failed == 2


def test_runner_keeps_failed_run_in_denominator(tmp_path):
    """Honest-counting invariant: a failed run scores 0, never dropped."""
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    dataset = LoadedDataset(name="toy", schema=schema, document="d", gold={"a": "x"})

    class _Failing:
        name = "x"

        def run(
            self, document, schema, *, model, context_window, max_output_tokens, instructions=""
        ):
            return AdapterOutput(
                data={}, fields_total=1, fields_extracted=0, k=0, k_min=0, error="429"
            )

    artifacts = run_sweep(
        _Failing(),
        dataset,
        model="m",
        seeds=1,
        out_dir=tmp_path / "f",
        context_window=8192,
        max_output_tokens=2048,
        budget="native",
    )
    assert artifacts.scored_path is not None
