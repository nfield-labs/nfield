"""Core value types for FormatShield extraction results.

This module defines the immutable data structures that flow through the
FormatShield pipeline. All classes are frozen dataclasses (slots=True) to
guarantee immutability and provide memory-efficient attribute access.

Classes:
    ExtractionStatus: Enum of pipeline outcome states.
    Metadata:         Statistical summary of an extraction run.
    FieldResult:      Per-field extraction outcome.
    ExtractionResult: Top-level result returned to the caller.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "ExtractionResult",
    "ExtractionStatus",
    "FieldResult",
    "Metadata",
]


class ExtractionStatus(Enum):
    """Overall status of an extraction run.

    Attributes:
        SUCCESS: All required fields were extracted with high confidence.
        PARTIAL: Some fields are missing or have low confidence scores.
        FAILED:  The pipeline could not produce a usable result.

    Example:
        >>> ExtractionStatus.SUCCESS.value
        'success'
        >>> ExtractionStatus("partial") is ExtractionStatus.PARTIAL
        True
    """

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Metadata:
    """Statistical summary produced by the N-field extraction pipeline.

    All numeric fields reflect the state *after* all retry rounds complete.

    Args:
        K: Number of document chunks used in retrieval (actual).
        K_min: Minimum K estimate computed by the routing engine.
        optimality_gap: Fractional gap between actual and optimal chunk count.
            Range [0, 1]; lower is better.
        quality_score: Aggregate quality score for the extraction run.
            Range [0, 1]; higher is better.
        confidence_level: Human-readable confidence tier, e.g. ``"HIGH"``.
        fields_extracted: Number of fields successfully extracted.
        fields_total: Total number of fields in the schema.
        fields_missing: Number of required fields not found in the document.
        fields_conflicted: Number of fields with conflicting evidence sources.
        fields_needs_revalidation: Number of fields flagged for revalidation.
        per_field_confidence: Mapping of field path → confidence score [0, 1].
        retry_rounds: Number of retry rounds performed (0 = first pass only).
        cost: Estimated token cost in USD, or ``None`` if not tracked.
        fields_call_failed: Number of fields left unextracted because an API/call
            error never returned (transient), as distinct from fields genuinely
            absent from the document. ``0`` when every call succeeded.
        calls_by_origin: Breakdown of ``K`` by call site — e.g.
            ``{"extract": 120, "s5_retry": 30, "recovery_extract": 8}``. Lets a
            run attribute its API cost to first-pass extraction vs Stage 5 retry
            vs the recovery pass. Empty when no calls were made.
        fields_grounded: Number of filled values whose grounding score met the
            threshold (supported by the source). ``0`` when grounding is disabled.
        fields_ungrounded: Number of grounding-checked values the source did not
            support (likely hallucinations). ``0`` when grounding is disabled.
        hallucination_rate: ``fields_ungrounded / (fields_grounded + fields_ungrounded)``
            — the fraction of grounding-checked values that were unsupported. ``None``
            when grounding was disabled or no value was groundable.
        unknown_output_lines: Count of extracted lines whose field path was not in the
            schema (the model emitted a field outside the requested set) — a format-drift
            signal. ``0`` when the model stayed within the schema.

    Example:
        >>> meta = Metadata(
        ...     K=5, K_min=3, optimality_gap=0.1, quality_score=0.95,
        ...     confidence_level="HIGH", fields_extracted=10, fields_total=10,
        ...     fields_missing=0, fields_conflicted=0, fields_needs_revalidation=0,
        ...     per_field_confidence={"name": 0.98}, retry_rounds=0,
        ... )
        >>> meta.quality_score
        0.95
    """

    K: int
    K_min: int
    optimality_gap: float
    quality_score: float
    confidence_level: str
    fields_extracted: int
    fields_total: int
    fields_missing: int
    fields_conflicted: int
    fields_needs_revalidation: int
    per_field_confidence: dict[str, float]
    retry_rounds: int
    cost: float | None = None
    fields_call_failed: int = 0
    calls_by_origin: dict[str, int] = field(default_factory=dict)
    fields_grounded: int = 0
    fields_ungrounded: int = 0
    hallucination_rate: float | None = None
    unknown_output_lines: int = 0
    # Closed-book reporting: fraction of fields answered vs left NULL by abstention.
    # Both ``None`` for document extraction.
    answer_rate: float | None = None
    abstain_rate: float | None = None


@dataclass(frozen=True, slots=True)
class FieldResult:
    """Extraction outcome for a single schema field.

    Args:
        path: Dot-separated path of the field within the schema, e.g.
            ``"invoice.line_items[0].amount"``.
        value: The extracted value. May be any JSON-serialisable type.
        confidence: Confidence score for this extraction, range [0, 1].
        is_missing: ``True`` if the field was not found in the document.
        error: Error message if extraction failed for this field, else ``None``.

    Example:
        >>> fr = FieldResult(path="vendor", value="Acme Corp", confidence=0.97)
        >>> fr.is_missing
        False
        >>> fr.error is None
        True
    """

    path: str
    value: Any
    confidence: float
    is_missing: bool = False
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Top-level result returned to callers of the FormatShield pipeline.

    Args:
        data: Extracted fields as a nested dictionary matching the schema
            structure. Missing optional fields are omitted; required missing
            fields are present with a ``None`` value.
        metadata: Statistical summary of the extraction run.
        status: Overall outcome of the extraction (SUCCESS / PARTIAL / FAILED).
        fields: Tuple of per-field results for detailed inspection.
            Defaults to an empty tuple when per-field details are not needed.

    Example:
        >>> from formatshield.types import ExtractionResult, ExtractionStatus, Metadata
        >>> meta = Metadata(
        ...     K=2, K_min=2, optimality_gap=0.0, quality_score=1.0,
        ...     confidence_level="HIGH", fields_extracted=1, fields_total=1,
        ...     fields_missing=0, fields_conflicted=0, fields_needs_revalidation=0,
        ...     per_field_confidence={"vendor": 0.99}, retry_rounds=0,
        ... )
        >>> result = ExtractionResult(
        ...     data={"vendor": "Acme"}, metadata=meta,
        ...     status=ExtractionStatus.SUCCESS,
        ... )
        >>> result.status is ExtractionStatus.SUCCESS
        True
    """

    data: dict[str, Any]
    metadata: Metadata
    status: ExtractionStatus
    fields: tuple[FieldResult, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict form of the result, ready for JSON serialisation.

        The enum status becomes its string value; metadata and per-field results
        become nested dicts. Round-trips with :meth:`from_dict`.

        Returns:
            A JSON-serialisable dict with ``data``, ``metadata``, ``status``,
            ``fields`` keys.

        Example:
            >>> # ExtractionResult.from_dict(r.to_dict()) == r
        """
        return {
            "data": self.data,
            "metadata": asdict(self.metadata),
            "status": self.status.value,
            "fields": [asdict(f) for f in self.fields],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExtractionResult:
        """Reconstruct an ``ExtractionResult`` from its :meth:`to_dict` form.

        Args:
            payload: A dict produced by :meth:`to_dict` (or matching its shape).

        Returns:
            The reconstructed ``ExtractionResult``.

        Example:
            >>> # ExtractionResult.from_dict(r.to_dict()).status is r.status
        """
        return cls(
            data=payload["data"],
            metadata=Metadata(**payload["metadata"]),
            status=ExtractionStatus(payload["status"]),
            fields=tuple(FieldResult(**f) for f in payload.get("fields", [])),
        )
