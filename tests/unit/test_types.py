"""Tests for nfield.types."""

from __future__ import annotations

import pytest

from nfield.types import ExtractionResult, ExtractionStatus, FieldResult, Metadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata(**overrides: object) -> Metadata:
    defaults: dict = {
        "K": 5,
        "K_min": 3,
        "optimality_gap": 0.1,
        "quality_score": 0.95,
        "confidence_level": "HIGH",
        "fields_extracted": 10,
        "fields_total": 10,
        "fields_missing": 0,
        "fields_conflicted": 0,
        "fields_needs_revalidation": 0,
        "per_field_confidence": {"vendor": 0.97},
        "retry_rounds": 0,
    }
    defaults.update(overrides)
    return Metadata(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ExtractionStatus
# ---------------------------------------------------------------------------


class TestExtractionStatus:
    def test_has_success_member(self) -> None:
        assert ExtractionStatus.SUCCESS is not None

    def test_has_partial_member(self) -> None:
        assert ExtractionStatus.PARTIAL is not None

    def test_has_failed_member(self) -> None:
        assert ExtractionStatus.FAILED is not None

    def test_success_value(self) -> None:
        assert ExtractionStatus.SUCCESS.value == "success"

    def test_partial_value(self) -> None:
        assert ExtractionStatus.PARTIAL.value == "partial"

    def test_failed_value(self) -> None:
        assert ExtractionStatus.FAILED.value == "failed"

    def test_round_trip_from_value_success(self) -> None:
        assert ExtractionStatus("success") is ExtractionStatus.SUCCESS

    def test_round_trip_from_value_partial(self) -> None:
        assert ExtractionStatus("partial") is ExtractionStatus.PARTIAL

    def test_round_trip_from_value_failed(self) -> None:
        assert ExtractionStatus("failed") is ExtractionStatus.FAILED

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            ExtractionStatus("unknown")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_construction_with_all_required_fields(self) -> None:
        meta = _make_metadata()
        assert meta.K == 5
        assert meta.K_min == 3

    def test_optimality_gap_stored_correctly(self) -> None:
        meta = _make_metadata(optimality_gap=0.25)
        assert meta.optimality_gap == pytest.approx(0.25)

    def test_quality_score_stored_correctly(self) -> None:
        meta = _make_metadata(quality_score=0.88)
        assert meta.quality_score == pytest.approx(0.88)

    def test_cost_defaults_to_none(self) -> None:
        meta = _make_metadata()
        assert meta.cost is None

    def test_cost_stored_when_provided(self) -> None:
        meta = _make_metadata(cost=0.0042)
        assert meta.cost == pytest.approx(0.0042)

    def test_per_field_confidence_stored(self) -> None:
        meta = _make_metadata(per_field_confidence={"a": 0.9, "b": 0.8})
        assert meta.per_field_confidence["a"] == pytest.approx(0.9)

    def test_retry_rounds_stored(self) -> None:
        meta = _make_metadata(retry_rounds=2)
        assert meta.retry_rounds == 2

    def test_frozen_cannot_reassign(self) -> None:
        meta = _make_metadata()
        with pytest.raises(AttributeError):
            meta.K = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FieldResult
# ---------------------------------------------------------------------------


class TestFieldResult:
    def test_construction(self) -> None:
        fr = FieldResult(path="vendor", value="Acme", confidence=0.97)
        assert fr.path == "vendor"
        assert fr.value == "Acme"
        assert fr.confidence == pytest.approx(0.97)

    def test_is_missing_defaults_to_false(self) -> None:
        fr = FieldResult(path="x", value=None, confidence=0.0)
        assert fr.is_missing is False

    def test_error_defaults_to_none(self) -> None:
        fr = FieldResult(path="x", value=None, confidence=0.0)
        assert fr.error is None

    def test_is_missing_can_be_set(self) -> None:
        fr = FieldResult(path="x", value=None, confidence=0.0, is_missing=True)
        assert fr.is_missing is True

    def test_error_can_be_set(self) -> None:
        fr = FieldResult(path="x", value=None, confidence=0.0, error="parse failed")
        assert fr.error == "parse failed"

    def test_frozen_cannot_reassign(self) -> None:
        fr = FieldResult(path="x", value="v", confidence=0.5)
        with pytest.raises(AttributeError):
            fr.path = "y"  # type: ignore[misc]

    def test_value_can_be_any_type(self) -> None:
        fr = FieldResult(path="items", value=[1, 2, 3], confidence=0.8)
        assert fr.value == [1, 2, 3]


# ---------------------------------------------------------------------------
# ExtractionResult
# ---------------------------------------------------------------------------


class TestExtractionResult:
    def _make_result(self, **overrides: object) -> ExtractionResult:
        meta = _make_metadata()
        defaults: dict = {
            "data": {"vendor": "Acme"},
            "metadata": meta,
            "status": ExtractionStatus.SUCCESS,
        }
        defaults.update(overrides)
        return ExtractionResult(**defaults)  # type: ignore[arg-type]

    def test_construction(self) -> None:
        r = self._make_result()
        assert r.data == {"vendor": "Acme"}
        assert r.status is ExtractionStatus.SUCCESS

    def test_status_is_extraction_status(self) -> None:
        r = self._make_result()
        assert isinstance(r.status, ExtractionStatus)

    def test_data_is_dict(self) -> None:
        r = self._make_result()
        assert isinstance(r.data, dict)

    def test_fields_defaults_to_empty_tuple(self) -> None:
        r = self._make_result()
        assert r.fields == ()
        assert isinstance(r.fields, tuple)

    def test_fields_can_be_populated(self) -> None:
        fr = FieldResult(path="vendor", value="Acme", confidence=0.97)
        r = self._make_result(fields=(fr,))
        assert len(r.fields) == 1
        assert r.fields[0].path == "vendor"

    def test_frozen_cannot_reassign(self) -> None:
        r = self._make_result()
        with pytest.raises(AttributeError):
            r.status = ExtractionStatus.FAILED  # type: ignore[misc]

    def test_partial_status(self) -> None:
        r = self._make_result(status=ExtractionStatus.PARTIAL)
        assert r.status is ExtractionStatus.PARTIAL

    def test_failed_status(self) -> None:
        r = self._make_result(status=ExtractionStatus.FAILED)
        assert r.status is ExtractionStatus.FAILED


# ---------------------------------------------------------------------------
# Import smoke test
# ---------------------------------------------------------------------------


def test_imports_from_nfield_types() -> None:
    from nfield.types import (  # noqa: F401
        ExtractionResult,
        ExtractionStatus,
        FieldResult,
        Metadata,
    )


# ---------------------------------------------------------------------------
# to_dict / from_dict round-trip
# ---------------------------------------------------------------------------


class TestResultSerialization:
    def _result(self) -> ExtractionResult:
        return ExtractionResult(
            data={"vendor": "Acme", "total": 12.5},
            metadata=_make_metadata(),
            status=ExtractionStatus.PARTIAL,
            fields=(
                FieldResult(path="vendor", value="Acme", confidence=0.97),
                FieldResult(path="total", value=12.5, confidence=0.8, is_missing=False),
            ),
        )

    def test_to_dict_is_json_serialisable(self) -> None:
        import json

        payload = self._result().to_dict()
        # status is a plain string, not an enum, so json.dumps does not choke.
        assert payload["status"] == "partial"
        json.dumps(payload)  # must not raise

    def test_round_trip_equals_original(self) -> None:
        original = self._result()
        assert ExtractionResult.from_dict(original.to_dict()) == original

    def test_round_trip_without_field_results(self) -> None:
        bare = ExtractionResult(
            data={"vendor": "Acme"},
            metadata=_make_metadata(),
            status=ExtractionStatus.SUCCESS,
        )
        assert ExtractionResult.from_dict(bare.to_dict()) == bare
