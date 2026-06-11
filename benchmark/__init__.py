"""nfield-bench — the field-count scaling benchmark for grounded extraction.

A development/research artifact (not part of the shipped wheel). It turns the
metrics the pipeline already emits — coverage, K, K_min, latency — plus a single
new gold-diff scorer into an honest, single-model, dated, reproducible curve of
field-level Value Accuracy as a function of schema field count N.

The benchmark answers one question and refuses to answer any other:

    On model M, on date D, at N fields, what fraction of fields does each method
    extract correctly?

It never produces an averaged-over-models number and never compares across model
substrates. See ``benchmark/README.md`` for the full honest-claims charter.
"""

from __future__ import annotations

from .score import FieldType, Outcome, ScoreReport, score

__all__ = ["FieldType", "Outcome", "ScoreReport", "score"]
