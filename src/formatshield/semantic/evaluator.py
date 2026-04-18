"""Deterministic semantic scoring for side-by-side structured-output comparisons."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Any


@dataclass
class MetricScore:
    """Score for a single semantic metric."""

    name: str
    score: float
    max_score: float
    note: str

    def model_dump(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class CandidateSemanticScore:
    """Aggregated semantic score for one candidate output."""

    total: float
    max_total: float
    normalized: float
    metrics: list[MetricScore]

    def model_dump(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "max_total": self.max_total,
            "normalized": self.normalized,
            "metrics": [m.model_dump() for m in self.metrics],
        }


@dataclass
class SemanticComparison:
    """Comparison result across FormatShield and raw outputs."""

    formatshield: CandidateSemanticScore
    raw: CandidateSemanticScore
    delta: float
    winner: str
    summary: str

    def model_dump(self) -> dict[str, Any]:
        return {
            "formatshield": self.formatshield.model_dump(),
            "raw": self.raw.model_dump(),
            "delta": self.delta,
            "winner": self.winner,
            "summary": self.summary,
        }


def _json_from_output(result: dict[str, Any]) -> dict[str, Any] | None:
    output = result.get("output")
    if not isinstance(output, str) or not output.strip():
        return None
    try:
        parsed = json.loads(output)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _is_type_match(value: Any, expected_type: str) -> bool:
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected_type == "null":
        return value is None
    return True


def _type_consistency(parsed: dict[str, Any], schema: dict[str, Any]) -> tuple[float, str]:
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        return 1.0, "No typed properties declared"

    checked = 0
    matched = 0
    for key, child_schema in properties.items():
        if key not in parsed or not isinstance(child_schema, dict):
            continue
        expected = child_schema.get("type")
        if not isinstance(expected, str):
            continue
        checked += 1
        if _is_type_match(parsed[key], expected):
            matched += 1

    if checked == 0:
        return 1.0, "No overlapping typed fields in output"

    ratio = matched / checked
    return ratio, f"{matched}/{checked} overlapping typed fields match"


def _required_field_recall(parsed: dict[str, Any], schema: dict[str, Any]) -> tuple[float, str]:
    required = schema.get("required")
    if not isinstance(required, list) or not required:
        return 1.0, "No required fields declared"

    required_names = [name for name in required if isinstance(name, str)]
    if not required_names:
        return 1.0, "No required fields declared"

    matched = sum(1 for name in required_names if name in parsed)
    ratio = matched / len(required_names)
    return ratio, f"{matched}/{len(required_names)} required fields present"


def _constraint_integrity(
    result: dict[str, Any],
    parsed: dict[str, Any] | None,
    schema: dict[str, Any],
) -> tuple[float, str]:
    if not result.get("ok", False):
        return 0.0, "Call failed"

    if result.get("schema_violation"):
        return 0.0, "Schema constraint violation reported"

    if parsed is None:
        return 0.25, "Output not parseable as object"

    properties = schema.get("properties")
    additional = schema.get("additionalProperties", True)
    if additional is False and isinstance(properties, dict):
        unknown = [key for key in parsed if key not in properties]
        if unknown:
            return 0.5, f"Unknown keys present: {unknown[:3]}"

    return 1.0, "No integrity violations detected"


def evaluate_candidate_semantics(
    result: dict[str, Any],
    schema: dict[str, Any],
    *,
    phi: float | None = None,
) -> CandidateSemanticScore:
    """Compute deterministic semantic proxy score for one candidate output."""
    parsed = _json_from_output(result)

    metrics: list[MetricScore] = []

    schema_validity = 1.0 if result.get("schema_valid", False) else 0.0
    metrics.append(
        MetricScore(
            name="schema_validity",
            score=35.0 * schema_validity,
            max_score=35.0,
            note="Schema-valid output" if schema_validity else "Schema validation failed",
        )
    )

    req_recall, req_note = _required_field_recall(parsed or {}, schema)
    metrics.append(
        MetricScore(
            name="required_field_recall",
            score=25.0 * req_recall,
            max_score=25.0,
            note=req_note,
        )
    )

    type_ratio, type_note = _type_consistency(parsed or {}, schema)
    metrics.append(
        MetricScore(
            name="type_consistency",
            score=20.0 * type_ratio,
            max_score=20.0,
            note=type_note,
        )
    )

    integrity_ratio, integrity_note = _constraint_integrity(result, parsed, schema)
    metrics.append(
        MetricScore(
            name="constraint_integrity",
            score=15.0 * integrity_ratio,
            max_score=15.0,
            note=integrity_note,
        )
    )

    route_note = "No routing signal"
    route_score = 5.0
    route_max = 5.0
    strategy = result.get("routing_strategy")
    if isinstance(strategy, str):
        if phi is None:
            route_score = 3.0
            route_note = "Route present without Φ"
        else:
            expected_ttf = phi > 0.65
            route_is_ttf = strategy == "ttf"
            aligned = expected_ttf == route_is_ttf
            route_score = 5.0 if aligned else 2.0
            route_note = (
                "Route aligned with Φ recommendation"
                if aligned
                else "Route diverged from Φ recommendation"
            )

    metrics.append(
        MetricScore(
            name="route_alignment",
            score=route_score,
            max_score=route_max,
            note=route_note,
        )
    )

    total = sum(metric.score for metric in metrics)
    max_total = sum(metric.max_score for metric in metrics)
    normalized = round((total / max_total) * 100.0, 1) if max_total else 0.0

    return CandidateSemanticScore(
        total=round(total, 2),
        max_total=max_total,
        normalized=normalized,
        metrics=metrics,
    )


def evaluate_semantic_pair(
    formatshield_result: dict[str, Any],
    raw_result: dict[str, Any],
    schema: dict[str, Any],
    *,
    phi: float | None = None,
) -> dict[str, Any]:
    """Evaluate semantic proxy scores for FormatShield and raw outputs."""
    fs_score = evaluate_candidate_semantics(formatshield_result, schema, phi=phi)
    raw_score = evaluate_candidate_semantics(raw_result, schema, phi=None)

    delta = round(fs_score.normalized - raw_score.normalized, 1)
    if delta > 2.0:
        winner = "formatshield"
        summary = (
            f"Semantic proxy favors FormatShield ({fs_score.normalized} vs "
            f"{raw_score.normalized}, Δ {delta:+.1f})."
        )
    elif delta < -2.0:
        winner = "raw"
        summary = (
            f"Semantic proxy favors raw ({raw_score.normalized} vs "
            f"{fs_score.normalized}, Δ {delta:+.1f})."
        )
    else:
        winner = "tie"
        summary = (
            f"Semantic proxy is comparable ({fs_score.normalized} vs "
            f"{raw_score.normalized}, Δ {delta:+.1f})."
        )

    comparison = SemanticComparison(
        formatshield=fs_score,
        raw=raw_score,
        delta=delta,
        winner=winner,
        summary=summary,
    )
    return comparison.model_dump()
