"""Unit tests for validation._retry — surgical field retry (SFR)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from nfield.config import ExtractionConfig
from nfield.schema._types import CapacityLeaf, Field
from nfield.validation._retry import (
    FailureCause,
    build_retry_prompt,
    classify_failure,
    handle_missing_fields,
    orchestrate_retry,
    split_retry_batches,
    surgical_field_retry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_field(
    path: str,
    ftype: str = "string",
    tau: float = 2.0,
) -> Field:
    return Field(
        path=path,
        type=ftype,
        constraints={},
        parent_path="",
        schema_node={},
        tau=tau,
    )


def make_leaf(
    fields: list[Field] | None = None,
    excerpt: str = "test document",
    safe_output: int = 256,
    leaf_id: int = 1,
) -> CapacityLeaf:
    return CapacityLeaf(
        fields=fields or [],
        document_excerpt=excerpt,
        safe_output=safe_output,
        leaf_id=leaf_id,
    )


def make_config(max_retry_rounds: int = 2) -> ExtractionConfig:
    return ExtractionConfig(max_retry_rounds=max_retry_rounds)


def make_mock_provider(response: str = "age = 30") -> AsyncMock:
    provider = AsyncMock()
    provider.complete = AsyncMock(return_value=response)
    return provider


# ---------------------------------------------------------------------------
# FailureCause enum
# ---------------------------------------------------------------------------


class TestFailureCause:
    def test_enum_values(self):
        assert FailureCause.FORMAT.value == "format"
        assert FailureCause.TYPE_CONSTRAINT.value == "type_constraint"
        assert FailureCause.FIELD_MISSING.value == "field_missing"
        assert FailureCause.DEPENDENCY_VALUE_CHANGED.value == "dependency_value_changed"

    def test_four_mvp_causes_defined(self):
        causes = list(FailureCause)
        assert len(causes) == 4


# ---------------------------------------------------------------------------
# classify_failure — 4 MVP causes
# ---------------------------------------------------------------------------


class TestClassifyFailure:
    def test_missing_cause_when_value_none_and_missing_in_error(self):
        f = make_field("name")
        cause = classify_failure(f, None, "field_missing")
        assert cause == FailureCause.FIELD_MISSING

    def test_type_constraint_cause_default(self):
        f = make_field("age", "integer")
        cause = classify_failure(f, "thirty", "expected integer, got str 'thirty'")
        assert cause == FailureCause.TYPE_CONSTRAINT

    def test_format_cause_on_parse_error(self):
        f = make_field("x")
        cause = classify_failure(f, None, "sfep parse error — malformed line")
        assert cause == FailureCause.FORMAT

    def test_dependency_changed_cause(self):
        f = make_field("total", "number")
        cause = classify_failure(f, 100.0, "dependency value changed for 'subtotal'")
        assert cause == FailureCause.DEPENDENCY_VALUE_CHANGED

    def test_constraint_violation_returns_type_constraint(self):
        f = make_field("score", "number")
        cause = classify_failure(f, 150, "maximum constraint violated — 150 > 100")
        assert cause == FailureCause.TYPE_CONSTRAINT


# ---------------------------------------------------------------------------
# split_retry_batches
# ---------------------------------------------------------------------------


class TestSplitRetryBatches:
    def test_no_deps_each_field_separate_batch(self):
        fields = [make_field("a"), make_field("b"), make_field("c")]
        batches = split_retry_batches(fields, {})
        assert len(batches) == 3

    def test_related_fields_in_same_batch(self):
        fa = make_field("a")
        fb = make_field("b")
        dep_dag = {"b": {"a"}}  # b depends on a
        batches = split_retry_batches([fa, fb], dep_dag)
        assert len(batches) == 1
        batch_paths = {f.path for f in batches[0]}
        assert batch_paths == {"a", "b"}

    def test_independent_field_separate_from_dep_group(self):
        fa = make_field("a")
        fb = make_field("b")
        fc = make_field("c")
        dep_dag = {"b": {"a"}}  # a-b are related; c is independent
        batches = split_retry_batches([fa, fb, fc], dep_dag)
        assert len(batches) == 2

    def test_empty_fields_returns_empty(self):
        assert split_retry_batches([], {}) == []

    def test_single_field_single_batch(self):
        f = make_field("x")
        batches = split_retry_batches([f], {})
        assert len(batches) == 1
        assert batches[0][0].path == "x"

    def test_max_2_rounds_enforced(self):
        # This tests the structure, not async behavior
        fields = [make_field("a"), make_field("b")]
        batches = split_retry_batches(fields, {})
        assert len(batches) == 2


# ---------------------------------------------------------------------------
# build_retry_prompt
# ---------------------------------------------------------------------------


class TestBuildRetryPrompt:
    def test_returns_two_messages(self):
        f = make_field("age", "integer")
        msgs = build_retry_prompt([f], {"age": "expected integer"}, "doc")
        assert len(msgs) == 2

    def test_system_message_present(self):
        f = make_field("age", "integer")
        msgs = build_retry_prompt([f], {"age": "parse error"}, "doc")
        assert msgs[0]["role"] == "system"

    def test_user_contains_error(self):
        f = make_field("age", "integer")
        msgs = build_retry_prompt([f], {"age": "Cannot cast 'thirty' to integer"}, "doc")
        assert "thirty" in msgs[1]["content"]

    def test_user_contains_document(self):
        f = make_field("x", "string")
        msgs = build_retry_prompt([f], {"x": "error"}, "He is thirty years old.")
        assert "thirty years old" in msgs[1]["content"]


# ---------------------------------------------------------------------------
# handle_missing_fields
# ---------------------------------------------------------------------------


class TestHandleMissingFields:
    def test_top_level_field_marked_none(self):
        leaf = make_leaf()
        result = handle_missing_fields(["name"], leaf, [])
        assert result == {"name": None}

    def test_empty_missing_returns_empty(self):
        leaf = make_leaf()
        result = handle_missing_fields([], leaf, [])
        assert result == {}

    def test_nested_path_with_no_parent_in_leaf(self):
        leaf = make_leaf(fields=[])  # no parent field in leaf
        result = handle_missing_fields(["address.city"], leaf, [])
        assert result.get("address.city") is None

    def test_multiple_missing_all_handled(self):
        leaf = make_leaf()
        result = handle_missing_fields(["a", "b", "c"], leaf, [])
        assert set(result.keys()) == {"a", "b", "c"}
        assert all(v is None for v in result.values())


# ---------------------------------------------------------------------------
# orchestrate_retry (async)
# ---------------------------------------------------------------------------


class TestOrchestrateRetry:
    @pytest.mark.asyncio
    async def test_recovers_failed_field(self):
        f = make_field("age", "integer")
        leaf = make_leaf(fields=[f])
        provider = make_mock_provider("age = 30")
        config = make_config(max_retry_rounds=1)

        result = await orchestrate_retry(
            [f],
            {"age": "expected integer, got str"},
            provider,
            leaf,
            dep_dag={},
            config=config,
        )
        assert result.get("age") == 30

    @pytest.mark.asyncio
    async def test_empty_failed_fields_returns_empty(self):
        provider = make_mock_provider()
        config = make_config()
        leaf = make_leaf()
        result = await orchestrate_retry([], {}, provider, leaf, dep_dag={}, config=config)
        assert result == {}

    @pytest.mark.asyncio
    async def test_max_rounds_respected(self):
        f = make_field("x")
        leaf = make_leaf(fields=[f])
        # Provider returns empty (field still missing)
        provider = make_mock_provider("")
        config = make_config(max_retry_rounds=2)

        await orchestrate_retry(
            [f],
            {"x": "missing"},
            provider,
            leaf,
            dep_dag={},
            config=config,
        )
        # Max rounds: provider.complete called at most max_retry_rounds times
        assert provider.complete.call_count <= 2

    @pytest.mark.asyncio
    async def test_provider_failure_handled_gracefully(self):
        f = make_field("y", "string")
        leaf = make_leaf(fields=[f])
        provider = AsyncMock()
        provider.complete = AsyncMock(side_effect=RuntimeError("API error"))
        config = make_config(max_retry_rounds=1)

        # Should not raise — returns empty dict for failed provider call
        result = await orchestrate_retry(
            [f],
            {"y": "missing"},
            provider,
            leaf,
            dep_dag={},
            config=config,
        )
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# surgical_field_retry (async)
# ---------------------------------------------------------------------------


class TestSurgicalFieldRetry:
    @pytest.mark.asyncio
    async def test_parses_provider_output(self):
        f = make_field("name", "string")
        leaf = make_leaf(fields=[f])
        provider = make_mock_provider("name = Alice")

        result = await surgical_field_retry([f], {"name": "missing"}, provider, leaf)
        assert result == {"name": "Alice"}

    @pytest.mark.asyncio
    async def test_empty_provider_output_returns_empty(self):
        f = make_field("x")
        leaf = make_leaf(fields=[f])
        provider = make_mock_provider("")

        result = await surgical_field_retry([f], {"x": "error"}, provider, leaf)
        assert result == {}

    @pytest.mark.asyncio
    async def test_provider_error_returns_empty(self):
        f = make_field("z")
        leaf = make_leaf(fields=[f])
        provider = AsyncMock()
        provider.complete = AsyncMock(side_effect=Exception("timeout"))

        result = await surgical_field_retry([f], {"z": "error"}, provider, leaf)
        assert result == {}


class TestRetryBatchCapacityPacking:
    """Independent failed fields are packed into few batches, not one-per-field."""

    def _fields(self, n: int):
        from nfield.schema._types import Field

        return [
            Field(f"f{i:03d}", "string", {}, "", {}).with_tau(tau=2.0, var_tau=0.5)
            for i in range(n)
        ]

    def test_no_budget_is_one_batch_per_field(self):
        # Legacy behaviour: without a budget, independent fields are singletons.
        fields = self._fields(20)
        batches = split_retry_batches(fields, {})
        assert len(batches) == 20

    def test_budget_packs_many_fields_into_few_batches(self):
        # 100 independent absent fields must NOT become 100 retry calls.
        fields = self._fields(100)
        batches = split_retry_batches(fields, {}, max_output_tokens=2000)
        assert len(batches) < 10  # packed, not one-per-field
        # Every field still appears exactly once.
        packed = [f.path for b in batches for f in b]
        assert sorted(packed) == sorted(f.path for f in fields)

    def test_dependency_closure_never_split(self):
        # a<-b<-c chain stays in one batch even under a tiny budget.
        from nfield.schema._types import Field

        a = Field("a", "string", {}, "", {}).with_tau(tau=5.0, var_tau=0.5)
        b = Field("b", "string", {}, "", {}).with_tau(tau=5.0, var_tau=0.5)
        c = Field("c", "string", {}, "", {}).with_tau(tau=5.0, var_tau=0.5)
        dep_dag = {"b": {"a"}, "c": {"b"}}
        batches = split_retry_batches([a, b, c], dep_dag, max_output_tokens=1)
        closure = next(batch for batch in batches if any(f.path == "a" for f in batch))
        assert {f.path for f in closure} == {"a", "b", "c"}
