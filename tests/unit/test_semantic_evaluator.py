"""Unit tests for deterministic semantic evaluator."""

from __future__ import annotations

from formatshield.semantic.evaluator import evaluate_semantic_pair


def test_evaluate_semantic_pair_formatshield_wins_on_valid_required_fields() -> None:
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["answer", "confidence"],
        "additionalProperties": False,
    }

    fs_result = {
        "ok": True,
        "output": '{"answer":"ok","confidence":0.91}',
        "schema_valid": True,
        "routing_strategy": "ttf",
    }
    raw_result = {
        "ok": True,
        "output": '{"answer":"ok"}',
        "schema_valid": False,
        "schema_violation": "'confidence' is a required property",
    }

    comparison = evaluate_semantic_pair(fs_result, raw_result, schema, phi=0.72)

    assert comparison["winner"] == "formatshield"
    assert comparison["delta"] > 0
    assert comparison["formatshield"]["normalized"] > comparison["raw"]["normalized"]


def test_evaluate_semantic_pair_reports_tie_for_similar_outputs() -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    fs_result = {
        "ok": True,
        "output": '{"answer":"ok"}',
        "schema_valid": True,
        "routing_strategy": "direct",
    }
    raw_result = {
        "ok": True,
        "output": '{"answer":"ok"}',
        "schema_valid": True,
    }

    comparison = evaluate_semantic_pair(fs_result, raw_result, schema, phi=0.40)

    assert comparison["winner"] in {"tie", "formatshield"}
    assert abs(comparison["delta"]) <= 5.0
