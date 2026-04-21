"""Semantic evaluation primitives for deterministic quality checks."""

from formatshield.semantic.evaluator import (
    CandidateSemanticScore,
    MetricScore,
    SemanticComparison,
    evaluate_semantic_pair,
)
from formatshield.semantic.schema_alignment import (
    SchemaAlignmentResult,
    SchemaAuthority,
    assess_schema_alignment,
)
from formatshield.semantic.semantic_failure_detector import (
    SemanticFailureMode,
    detect_extraction_only,
    detect_mislabeled_extraction,
    detect_reasoning_intent,
    detect_reasoning_with_flat_schema,
    detect_schema_reasoning_mismatch,
)

__all__ = [
    "CandidateSemanticScore",
    "MetricScore",
    "SemanticComparison",
    "evaluate_semantic_pair",
    "SchemaAlignmentResult",
    "SchemaAuthority",
    "assess_schema_alignment",
    "SemanticFailureMode",
    "detect_extraction_only",
    "detect_mislabeled_extraction",
    "detect_reasoning_intent",
    "detect_reasoning_with_flat_schema",
    "detect_schema_reasoning_mismatch",
]
