"""Semantic evaluation primitives for deterministic quality checks."""

from formatshield.semantic.evaluator import (
    CandidateSemanticScore,
    MetricScore,
    SemanticComparison,
    evaluate_semantic_pair,
)

__all__ = [
    "CandidateSemanticScore",
    "MetricScore",
    "SemanticComparison",
    "evaluate_semantic_pair",
]
