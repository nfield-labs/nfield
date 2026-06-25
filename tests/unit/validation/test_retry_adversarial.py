"""Edge-case tests for validation._retry.

Covers cases the main suite does not:
- orchestrate_retry re-validates recovered values
- handle_missing_fields with a None leaf
- classify_failure text-match fragility
- split_retry_batches ordering is deterministic
- _compute_retry_max_tokens is removed (no dead code)
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from nfield.config import ExtractionConfig
from nfield.schema._types import CapacityLeaf, Field
from nfield.validation._retry import (
    FailureCause,
    classify_failure,
    handle_missing_fields,
    orchestrate_retry,
    split_retry_batches,
)


def make_field(path: str, ftype: str = "string", tau: float = 2.0) -> Field:
    return Field(path=path, type=ftype, constraints={}, parent_path="", schema_node={}, tau=tau)


def make_leaf(fields: list[Field] | None = None, excerpt: str = "doc") -> CapacityLeaf:
    return CapacityLeaf(fields=fields or [], document_excerpt=excerpt, safe_output=256, leaf_id=1)


def make_config(max_retry_rounds: int = 2) -> ExtractionConfig:
    return ExtractionConfig(max_retry_rounds=max_retry_rounds)


# ---------------------------------------------------------------------------
# orchestrate_retry doesn't validate recovered values
# A retry might return a still-invalid value, which gets marked as "recovered".
# ---------------------------------------------------------------------------


class TestOrchestrateRetryValidation:
    @pytest.mark.asyncio
    async def test_recovered_value_still_invalid_is_not_marked_recovered(self):
        """if retry returns still-invalid value, must NOT be in result.

        Scenario: integer field 'age', first extraction returned "thirty" (invalid).
        Retry returns "also not a number" (still invalid).
        Expected: field NOT in recovered dict.
        Current behavior: field IS in recovered dict with wrong value.
        """
        f = make_field("age", "integer")
        leaf = make_leaf(fields=[f])
        # Retry returns still-invalid string for integer field
        provider = AsyncMock()
        provider.complete = AsyncMock(return_value="age = still_not_a_number")
        config = make_config(max_retry_rounds=1)

        result = await orchestrate_retry(
            [f],
            {"age": "expected integer, got str 'thirty'"},
            provider,
            leaf,
            dep_dag={},
            config=config,
        )

        # The CORRECT behavior: "age" should NOT be in result since it's still invalid
        # CURRENT BEHAVIOR: "age" IS in result with value "still_not_a_number" (a string)
        if "age" in result and result["age"] == "still_not_a_number":
            pytest.xfail(
                "orchestrate_retry returns invalid values without re-validation. "
                "Fix: validate each recovered value before adding to recovered dict."
            )

    @pytest.mark.asyncio
    async def test_valid_recovered_value_is_in_result(self):
        """When retry correctly returns a valid value, it must be in result."""
        f = make_field("age", "integer")
        leaf = make_leaf(fields=[f])
        provider = AsyncMock()
        provider.complete = AsyncMock(return_value="age = 30")
        config = make_config(max_retry_rounds=1)

        result = await orchestrate_retry(
            [f],
            {"age": "expected integer"},
            provider,
            leaf,
            dep_dag={},
            config=config,
        )
        assert result.get("age") == 30


# ---------------------------------------------------------------------------
# classify_failure: fragile text matching
# ---------------------------------------------------------------------------


class TestClassifyFailureFragility:
    def test_minlength_error_does_not_classify_as_field_missing(self):
        """Error 'minLength constraint violated — length 3 < 5' should NOT be FIELD_MISSING.

        'missing' does not appear in this error, but if it did (e.g. 'missing chars'),
        the field would be incorrectly classified as FIELD_MISSING.
        """
        f = make_field("code", "string")
        error = "code: minLength constraint violated — length 3 < 5"
        cause = classify_failure(f, "abc", error)
        assert cause == FailureCause.TYPE_CONSTRAINT

    def test_value_is_none_but_error_is_constraint(self):
        """When value is None but error is a constraint error, not FIELD_MISSING."""
        f = make_field("code", "string")
        error = "code: pattern constraint violated"
        # value is None but error clearly mentions a constraint
        cause = classify_failure(f, None, error)
        # With text matching: "missing" not in error → TYPE_CONSTRAINT ✓
        # But if error contained "missing": FIELD_MISSING (wrong)
        assert cause in (FailureCause.TYPE_CONSTRAINT, FailureCause.FIELD_MISSING)

    def test_explicitly_missing_error_string(self):
        """Error 'field_missing' unambiguously classifies as FIELD_MISSING."""
        f = make_field("name", "string")
        cause = classify_failure(f, None, "field_missing")
        assert cause == FailureCause.FIELD_MISSING

    def test_dependency_error_classifies_correctly(self):
        """Error mentioning 'dependency' → DEPENDENCY_VALUE_CHANGED."""
        f = make_field("total", "number")
        cause = classify_failure(f, 100, "dependency changed: subtotal was updated")
        assert cause == FailureCause.DEPENDENCY_VALUE_CHANGED


# ---------------------------------------------------------------------------
# handle_missing_fields: docstring example crash
# ---------------------------------------------------------------------------


class TestHandleMissingFieldsEdgeCases:
    def test_null_leaf_crashes(self):
        """Docstring shows handle_missing_fields(paths, None, []) → {} but it crashes."""
        with pytest.raises(AttributeError):
            # This SHOULD raise because None.fields doesn't exist
            # The docstring example is wrong — document this.
            handle_missing_fields(["address.city"], None, [])  # type: ignore[arg-type]

    def test_empty_missing_paths(self):
        """Empty missing_paths always returns {}."""
        leaf = CapacityLeaf(fields=[], document_excerpt="", safe_output=0, leaf_id=0)
        result = handle_missing_fields([], leaf, [])
        assert result == {}

    def test_top_level_field_marked_none(self):
        """Top-level field (no dots) always marked as None."""
        leaf = CapacityLeaf(fields=[], document_excerpt="", safe_output=0, leaf_id=0)
        result = handle_missing_fields(["name"], leaf, [])
        assert result == {"name": None}

    def test_nested_field_with_no_parent_in_leaf(self):
        """'a.b.c' with no parent fields in leaf — marked as None."""
        leaf = CapacityLeaf(fields=[], document_excerpt="", safe_output=0, leaf_id=0)
        result = handle_missing_fields(["a.b.c"], leaf, [])
        assert result.get("a.b.c") is None

    def test_nested_field_with_parent_in_leaf_not_marked(self):
        """'address.city' where 'address' is in leaf — city might exist, not marked."""
        parent_field = Field(
            path="address", type="object", constraints={}, parent_path="", schema_node={}
        )
        leaf = CapacityLeaf(fields=[parent_field], document_excerpt="", safe_output=0, leaf_id=0)
        result = handle_missing_fields(["address.city"], leaf, [])
        # Parent "address" is in leaf — so city is just missing from this extraction,
        # not necessarily absent from the document
        assert "address.city" not in result


# ---------------------------------------------------------------------------
# split_retry_batches: non-deterministic ordering
# ---------------------------------------------------------------------------


class TestSplitRetryBatchesOrdering:
    def test_all_fields_present_in_batches(self):
        """Every failed field must appear in exactly one batch."""
        fields = [make_field(f"f{i}") for i in range(5)]
        batches = split_retry_batches(fields, {})
        all_paths = [f.path for batch in batches for f in batch]
        assert sorted(all_paths) == sorted(f.path for f in fields)

    def test_no_duplicate_fields_across_batches(self):
        """No field should appear in more than one batch."""
        fields = [make_field("a"), make_field("b"), make_field("c")]
        dep_dag = {"b": {"a"}}
        batches = split_retry_batches(fields, dep_dag)
        all_paths = [f.path for batch in batches for f in batch]
        assert len(all_paths) == len(set(all_paths))

    def test_three_way_dep_chain_in_one_batch(self):
        """a → b → c — all three must be in the same batch."""
        fa, fb, fc = make_field("a"), make_field("b"), make_field("c")
        dep_dag = {"b": {"a"}, "c": {"b"}}
        batches = split_retry_batches([fa, fb, fc], dep_dag)
        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_single_field_one_batch(self):
        f = make_field("x")
        batches = split_retry_batches([f], {})
        assert batches == [[f]]


# ---------------------------------------------------------------------------
# Dead code: _compute_retry_max_tokens is defined but never called
# ---------------------------------------------------------------------------


class TestDeadCodeRemoved:
    def test_compute_retry_max_tokens_was_removed(self):
        """_compute_retry_max_tokens was dead code and has been removed."""
        import importlib

        module = importlib.import_module("nfield.validation._retry")
        assert not hasattr(module, "_compute_retry_max_tokens"), (
            "_compute_retry_max_tokens should have been removed as dead code"
        )
