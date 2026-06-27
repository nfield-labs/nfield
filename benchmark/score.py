"""Gold-diff scorer - the one new measurement nfield-bench adds.

The pipeline already reports *coverage* (did a value come back). This module
adds *Value Accuracy* (is the value correct), the benchmark's headline metric.

A run's extracted data is flattened to dot-notation leaves, each leaf is matched
against the gold answer key under a type-aware rule (exact for scalars, numeric
tolerance for numbers, normalised-exact for short strings, bounded edit distance
for free text), and every gold field lands in exactly one outcome bucket:

Arrays are matched by **position**, not as sets: a list flattens to ``item_0``,
``item_1``, … and each element is scored against the gold element at the same
index. A correct set in a different order therefore scores as per-element
accuracy errors - a deliberate, disclosed reorder penalty (the design follows
SOB, which penalises reordering). Schemas needing order-insensitive lists are
out of scope for this scorer until a per-field policy is added.


    CORRECT      value present and judged equal to gold
    ACCURACY     value present but wrong
    OMISSION     value absent though gold has one
    HALLUCINATION gold marks the field empty but a value was produced
    STRUCTURAL   wrong shape at the path (a container where a scalar is due)

Value Accuracy is ``CORRECT / |gold|``. Failures stay in the denominator - a
method that returns nothing is scored a miss, never dropped (honest-claims
charter, rule 4). The scorer is pure and deterministic: no API, no clock, no
randomness, so it runs in CI and re-scores old raw outputs without re-generating
them.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "FieldScore",
    "FieldType",
    "Outcome",
    "ScoreReport",
    "TypeStat",
    "score",
]

# A free-text leaf is matched by bounded edit distance rather than exact string
# equality. Gold strings at or below this length are treated as identifiers /
# names / codes (normalised-exact); longer ones as prose. Length is the only
# domain-agnostic signal available without per-schema configuration.
SHORT_STRING_MAX_CHARS: int = 80

# Long strings match when their normalised Levenshtein distance is within this
# fraction of the longer string - the design's "semantic equivalence" proxy
# (10% edit budget tolerates reflow / punctuation drift, not paraphrase).
LONG_STRING_MAX_DISTANCE: float = 0.1

# Numbers match within this relative tolerance. Effectively exact; widened only
# when a schema implies units (caller-supplied), per the metrics table.
NUMERIC_REL_TOLERANCE: float = 1e-9

_ITEM_SEGMENT = re.compile(r"item_\d+\Z")
_WHITESPACE = re.compile(r"\s+")
_EMPTY_VALUES: tuple[Any, ...] = (None, "")


class FieldType(Enum):
    """Match-rule class a leaf falls into, resolved from schema + gold value."""

    BOOLEAN = "boolean"
    ENUM = "enum"
    INTEGER = "integer"
    NUMBER = "number"
    SHORT_STRING = "short_string"
    LONG_STRING = "long_string"


class Outcome(Enum):
    """Disjoint outcome of scoring one gold field."""

    CORRECT = "correct"
    ACCURACY = "accuracy"
    OMISSION = "omission"
    HALLUCINATION = "hallucination"
    STRUCTURAL = "structural"


@dataclass(frozen=True, slots=True)
class FieldScore:
    """Outcome of scoring a single gold field.

    Args:
        path: Dot-notation path of the field, e.g. ``"geography.area.total"``.
        field_type: The match-rule class applied.
        gold: The expected value from the answer key.
        predicted: The flattened extracted value, or ``None`` if absent.
        outcome: Which disjoint bucket this field fell into.
    """

    path: str
    field_type: FieldType
    gold: Any
    predicted: Any
    outcome: Outcome

    @property
    def correct(self) -> bool:
        """Return ``True`` when the predicted value matched the gold value."""
        return self.outcome is Outcome.CORRECT


@dataclass(frozen=True, slots=True)
class TypeStat:
    """Per-field-type accuracy tally (the mandatory breakdown)."""

    field_type: FieldType
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        """Fraction correct within this field type; ``0.0`` if the type is unused."""
        return self.correct / self.total if self.total else 0.0


@dataclass(frozen=True, slots=True)
class ScoreReport:
    """Complete gold-diff judgement for one extraction run.

    Args:
        n_fields: Size of the gold answer key - the Value Accuracy denominator.
        value_accuracy: ``correct / n_fields``, the headline metric.
        coverage: Fraction of gold fields for which any value was returned.
        json_pass: ``True`` when no field had a structural (wrong-shape) error.
        call_failed: Count of fields lost to API/call errors, carried through
            from ``Metadata.fields_call_failed``. A separate category from model
            omission - never blamed on the model. It is a run-level count: the
            scorer reports it alongside the per-field outcomes but cannot
            re-bucket individual omissions as call-failed without per-path
            attribution the caller does not supply.
        by_type: Per-field-type accuracy tallies.
        outcomes: Count of gold fields in each outcome bucket.
        fields: Per-field scores, in gold-key order.
    """

    n_fields: int
    value_accuracy: float
    coverage: float
    json_pass: bool
    call_failed: int
    by_type: dict[FieldType, TypeStat]
    outcomes: dict[Outcome, int]
    fields: tuple[FieldScore, ...]

    @property
    def precision(self) -> float:
        """Fraction of *answered* fields that are correct (abstentions excluded)."""
        answered = self.outcomes[Outcome.CORRECT] + self.outcomes[Outcome.ACCURACY]
        return self.outcomes[Outcome.CORRECT] / answered if answered else 0.0

    @property
    def reliability(self) -> float:
        """``(correct - wrong) / n_fields``: rewards abstention, penalises confident error.

        The closed-book headline. Unlike value accuracy, a wrong answer scores worse than an
        abstention, so confidently filling fields the model does not know lowers the score
        (accuracy-only scoring instead rewards that guessing - Nature s41586-026-10549-w).
        """
        if not self.n_fields:
            return 0.0
        return (self.outcomes[Outcome.CORRECT] - self.outcomes[Outcome.ACCURACY]) / self.n_fields


def score(
    extracted: dict[str, Any],
    gold: dict[str, Any],
    schema: dict[str, Any],
    *,
    call_failed: int = 0,
    numeric_tolerance: float = NUMERIC_REL_TOLERANCE,
) -> ScoreReport:
    """Judge one extraction against a gold answer key, type-aware, per field.

    Args:
        extracted: The run's nested result data (``ExtractionResult.data``).
        gold: Flat answer key mapping dot-notation path to expected value.
            Its key set is the scored field set (the denominator).
        schema: The JSON Schema the run targeted, used to resolve each path's
            match-rule class. Paths absent from the schema fall back to the
            gold value's Python type.
        call_failed: ``Metadata.fields_call_failed`` for the run, surfaced as
            its own category and never counted as a model omission.
        numeric_tolerance: Relative tolerance for ``number`` fields. Defaults to
            effectively-exact; widen only when a schema implies units.

    Returns:
        A :class:`ScoreReport` with the headline Value Accuracy, the mandatory
        per-field-type breakdown, coverage, the JSON-pass flag, and the disjoint
        error decomposition.

    Example:
        >>> schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
        >>> report = score({"n": 5}, {"n": 5}, schema)
        >>> report.value_accuracy
        1.0
    """
    flat = _flatten(extracted)
    prefixes = _container_prefixes(flat)
    scores: list[FieldScore] = []
    for path, gold_value in gold.items():
        node = _resolve(schema, path)
        field_type = _classify(node, gold_value)
        predicted = flat.get(path)
        shape_conflict = predicted is None and path in prefixes
        outcome = _judge(gold_value, predicted, field_type, numeric_tolerance, shape_conflict)
        scores.append(FieldScore(path, field_type, gold_value, predicted, outcome))

    return _aggregate(tuple(scores), call_failed)


def _aggregate(fields: tuple[FieldScore, ...], call_failed: int) -> ScoreReport:
    n = len(fields)
    outcomes: dict[Outcome, int] = dict.fromkeys(Outcome, 0)
    by_type_correct: dict[FieldType, int] = dict.fromkeys(FieldType, 0)
    by_type_total: dict[FieldType, int] = dict.fromkeys(FieldType, 0)
    covered = 0

    for fs in fields:
        outcomes[fs.outcome] += 1
        by_type_total[fs.field_type] += 1
        if fs.correct:
            by_type_correct[fs.field_type] += 1
        if not _is_empty(fs.predicted):
            covered += 1

    by_type = {
        ft: TypeStat(ft, by_type_correct[ft], by_type_total[ft])
        for ft in FieldType
        if by_type_total[ft]
    }
    correct = outcomes[Outcome.CORRECT]
    return ScoreReport(
        n_fields=n,
        value_accuracy=correct / n if n else 0.0,
        coverage=covered / n if n else 0.0,
        json_pass=outcomes[Outcome.STRUCTURAL] == 0,
        call_failed=call_failed,
        by_type=by_type,
        outcomes=outcomes,
        fields=fields,
    )


def _judge(
    gold: Any,
    predicted: Any,
    field_type: FieldType,
    numeric_tolerance: float,
    shape_conflict: bool,
) -> Outcome:
    # _flatten never stores an empty value, so an absent leaf arrives here as
    # None. shape_conflict distinguishes a true omission from a wrong-shape path
    # (a container where the gold expects a scalar) that flattening dissolved.
    if _is_empty(gold):
        return Outcome.CORRECT if _is_empty(predicted) else Outcome.HALLUCINATION
    if _is_empty(predicted):
        return Outcome.STRUCTURAL if shape_conflict else Outcome.OMISSION
    return (
        Outcome.CORRECT
        if _matches(gold, predicted, field_type, numeric_tolerance)
        else Outcome.ACCURACY
    )


def _matches(gold: Any, predicted: Any, field_type: FieldType, numeric_tolerance: float) -> bool:
    if field_type is FieldType.BOOLEAN:
        return _as_bool(gold) == _as_bool(predicted)
    if field_type is FieldType.INTEGER:
        return _as_int(gold) == _as_int(predicted)
    if field_type is FieldType.NUMBER:
        return _numeric_close(gold, predicted, numeric_tolerance)
    if field_type is FieldType.ENUM:
        return _norm(gold) == _norm(predicted)
    if field_type is FieldType.LONG_STRING:
        return _edit_ratio(_norm(gold), _norm(predicted)) <= LONG_STRING_MAX_DISTANCE
    return _norm(gold) == _norm(predicted)


def _classify(node: dict[str, Any] | None, gold_value: Any) -> FieldType:
    if node is not None:
        if "enum" in node:
            return FieldType.ENUM
        schema_type = node.get("type")
        if schema_type == "boolean":
            return FieldType.BOOLEAN
        if schema_type == "integer":
            return FieldType.INTEGER
        if schema_type == "number":
            return FieldType.NUMBER
        if schema_type == "string":
            return _string_type(gold_value)
    return _infer_type(gold_value)


def _string_type(gold_value: Any) -> FieldType:
    if isinstance(gold_value, str) and len(gold_value) > SHORT_STRING_MAX_CHARS:
        return FieldType.LONG_STRING
    return FieldType.SHORT_STRING


def _infer_type(gold_value: Any) -> FieldType:
    if isinstance(gold_value, bool):
        return FieldType.BOOLEAN
    if isinstance(gold_value, int):
        return FieldType.INTEGER
    if isinstance(gold_value, float):
        return FieldType.NUMBER
    return _string_type(gold_value)


def _resolve(schema: dict[str, Any], path: str) -> dict[str, Any] | None:
    node: Any = schema
    for segment in path.split("."):
        if not isinstance(node, dict):
            return None
        if _ITEM_SEGMENT.match(segment):
            node = node.get("items")
        else:
            properties = node.get("properties")
            node = properties.get(segment) if isinstance(properties, dict) else None
        if node is None:
            return None
    return node if isinstance(node, dict) else None


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(_flatten(value, f"{prefix}.{key}" if prefix else key))
    elif isinstance(obj, list | tuple):
        for index, value in enumerate(obj):
            out.update(_flatten(value, f"{prefix}.item_{index}"))
    elif not _is_empty(obj):
        out[prefix] = obj
    return out


def _container_prefixes(flat: dict[str, Any]) -> set[str]:
    prefixes: set[str] = set()
    for key in flat:
        parts = key.split(".")
        for cut in range(1, len(parts)):
            prefixes.add(".".join(parts[:cut]))
    return prefixes


def _is_empty(value: Any) -> bool:
    return any(value is empty or value == empty for empty in _EMPTY_VALUES)


def _norm(value: Any) -> str:
    # NFKD splits accented characters into base + combining mark; dropping the
    # marks (category Mn) folds diacritics so ASCII gold matches accented source
    # (e.g. "Kutúzov" -> "kutuzov"), the same fold the retrieval tokenizer uses.
    decomposed = unicodedata.normalize("NFKD", str(value))
    folded = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return _WHITESPACE.sub(" ", folded).strip().casefold()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _norm(value) in {"true", "yes", "1"}


def _as_int(value: Any) -> int | None:
    try:
        return int(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None


def _numeric_close(gold: Any, predicted: Any, tolerance: float) -> bool:
    gold_num, pred_num = _as_float(gold), _as_float(predicted)
    if gold_num is None or pred_num is None:
        return False
    scale = max(abs(gold_num), abs(pred_num), 1.0)
    return abs(gold_num - pred_num) <= tolerance * scale


def _as_float(value: Any) -> float | None:
    try:
        return float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None


def _edit_ratio(a: str, b: str) -> float:
    if a == b:
        return 0.0
    longer = max(len(a), len(b))
    if longer == 0:
        return 0.0
    return _levenshtein(a, b) / longer


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            substitute = previous[j - 1] + (ca != cb)
            current.append(min(insert, delete, substitute))
        previous = current
    return previous[-1]
