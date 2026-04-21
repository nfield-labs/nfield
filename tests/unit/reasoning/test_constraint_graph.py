"""
Unit tests for constraint_graph — bidirectional semantic constraint propagation.

Coverage:
- Graph construction: field extraction, edge detection (DERIVES_FROM, REQUIRED_IF,
  EXCLUSIVE, ORDERED, BOUNDS_VALUE)
- Propagation: domain reduction, determination, narrowing
- Inconsistency detection: committed values vs derived domains
- Batch propagation: complete output analysis
- Reset: domain restoration for reuse
- Edge cases: empty schema, flat schema, no dependencies
"""

from __future__ import annotations

from formatshield.reasoning.constraint_graph import (
    ConstraintPropagationGraph,
    EdgeType,
    GraphEdge,
    PropagationResult,
    build_constraint_graph,
)

# ---------------------------------------------------------------------------
# Fixture schemas
# ---------------------------------------------------------------------------

BOOLEAN_ALL_SCHEMA: dict = {
    "properties": {
        "premises": {
            "type": "array",
            "items": {
                "properties": {
                    "statement": {"type": "string"},
                    "valid": {"type": "boolean"},
                }
            },
        },
        "argument_valid": {"type": "boolean"},
    }
}

ENUM_FAIL_ANY_SCHEMA: dict = {
    "properties": {
        "checks": {
            "type": "array",
            "items": {
                "properties": {
                    "result": {"type": "string", "enum": ["PASS", "FAIL"]},
                }
            },
        },
        "overall": {"type": "string", "enum": ["PASS", "FAIL"]},
    }
}

FLAT_SCHEMA: dict = {
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "score": {"type": "number"},
    }
}

EMPTY_SCHEMA: dict = {}

IF_THEN_SCHEMA: dict = {
    "properties": {
        "type": {"type": "string", "enum": ["basic", "advanced"]},
        "advanced_options": {"type": "object"},
    },
    "if": {"properties": {"type": {"const": "advanced"}}},
    "then": {"required": ["advanced_options"]},
}

ONE_OF_SCHEMA: dict = {
    "properties": {
        "field_a": {"type": "string"},
        "field_b": {"type": "string"},
        "shared": {"type": "string"},
    },
    "oneOf": [
        {"required": ["field_a"]},
        {"required": ["field_b"]},
    ],
}

ARRAY_BOUNDS_SCHEMA: dict = {
    "properties": {
        "items": {
            "type": "array",
            "minItems": 2,
            "maxItems": 5,
            "items": {"properties": {"value": {"type": "string"}}},
        }
    }
}

MULTI_ARRAY_SCHEMA: dict = {
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "properties": {
                    "passed": {"type": "boolean"},
                }
            },
        },
        "overall_passed": {"type": "boolean"},
        "checks": {
            "type": "array",
            "items": {
                "properties": {
                    "result": {"type": "string", "enum": ["PASS", "FAIL"]},
                }
            },
        },
        "summary_result": {"type": "string", "enum": ["PASS", "FAIL"]},
    }
}


# ---------------------------------------------------------------------------
# Graph construction tests
# ---------------------------------------------------------------------------


class TestGraphConstruction:
    def test_nodes_extracted_for_top_level_fields(self) -> None:
        cpg = ConstraintPropagationGraph(BOOLEAN_ALL_SCHEMA)
        assert "argument_valid" in cpg._nodes

    def test_boolean_node_has_correct_initial_domain(self) -> None:
        cpg = ConstraintPropagationGraph(BOOLEAN_ALL_SCHEMA)
        node = cpg._nodes["argument_valid"]
        assert node.initial_domain == {True, False}

    def test_enum_node_has_correct_initial_domain(self) -> None:
        cpg = ConstraintPropagationGraph(ENUM_FAIL_ANY_SCHEMA)
        node = cpg._nodes["overall"]
        assert node.initial_domain == {"PASS", "FAIL"}
        assert node.field_type == "enum"

    def test_array_item_fields_extracted_with_wildcard_path(self) -> None:
        cpg = ConstraintPropagationGraph(BOOLEAN_ALL_SCHEMA)
        # Array item fields should be registered as premises[*].valid
        assert "premises[*].valid" in cpg._nodes

    def test_flat_schema_has_no_edges(self) -> None:
        cpg = ConstraintPropagationGraph(FLAT_SCHEMA)
        assert len(cpg._edges) == 0

    def test_empty_schema_builds_empty_graph(self) -> None:
        cpg = ConstraintPropagationGraph(EMPTY_SCHEMA)
        assert cpg.get_node_count() == 0
        assert len(cpg._edges) == 0

    def test_node_count_matches_schema_fields(self) -> None:
        cpg = ConstraintPropagationGraph(FLAT_SCHEMA)
        assert cpg.get_node_count() == 3  # name, age, score

    def test_get_edges_returns_copy(self) -> None:
        cpg = ConstraintPropagationGraph(BOOLEAN_ALL_SCHEMA)
        edges = cpg.get_edges()
        edges.clear()
        assert len(cpg.get_edges()) > 0  # original not mutated


class TestDerivesFromEdges:
    def test_derives_from_edge_created_for_boolean_all(self) -> None:
        cpg = ConstraintPropagationGraph(BOOLEAN_ALL_SCHEMA)
        derives_edges = [e for e in cpg._edges if e.edge_type == EdgeType.DERIVES_FROM]
        assert len(derives_edges) >= 1

    def test_derives_from_source_is_array_item_path(self) -> None:
        cpg = ConstraintPropagationGraph(BOOLEAN_ALL_SCHEMA)
        derives_edges = [e for e in cpg._edges if e.edge_type == EdgeType.DERIVES_FROM]
        sources = [e.source for e in derives_edges]
        assert any("premises[*]" in s for s in sources)

    def test_derives_from_target_is_parent_field(self) -> None:
        cpg = ConstraintPropagationGraph(BOOLEAN_ALL_SCHEMA)
        derives_edges = [e for e in cpg._edges if e.edge_type == EdgeType.DERIVES_FROM]
        targets = [e.target for e in derives_edges]
        assert "argument_valid" in targets

    def test_ordered_edge_created_alongside_derives_from(self) -> None:
        cpg = ConstraintPropagationGraph(BOOLEAN_ALL_SCHEMA)
        ordered_edges = [e for e in cpg._edges if e.edge_type == EdgeType.ORDERED]
        assert len(ordered_edges) >= 1

    def test_ordered_edge_target_is_parent_field(self) -> None:
        cpg = ConstraintPropagationGraph(BOOLEAN_ALL_SCHEMA)
        ordered_edges = [e for e in cpg._edges if e.edge_type == EdgeType.ORDERED]
        targets = [e.target for e in ordered_edges]
        assert "argument_valid" in targets

    def test_derives_from_metadata_contains_pattern(self) -> None:
        cpg = ConstraintPropagationGraph(BOOLEAN_ALL_SCHEMA)
        derives_edges = [e for e in cpg._edges if e.edge_type == EdgeType.DERIVES_FROM]
        assert any("pattern" in e.metadata for e in derives_edges)

    def test_enum_fail_any_derives_from_edge(self) -> None:
        cpg = ConstraintPropagationGraph(ENUM_FAIL_ANY_SCHEMA)
        derives_edges = [e for e in cpg._edges if e.edge_type == EdgeType.DERIVES_FROM]
        assert len(derives_edges) >= 1

    def test_no_derives_from_edges_for_flat_schema(self) -> None:
        cpg = ConstraintPropagationGraph(FLAT_SCHEMA)
        derives_edges = [e for e in cpg._edges if e.edge_type == EdgeType.DERIVES_FROM]
        assert len(derives_edges) == 0


class TestRequiredIfEdges:
    def test_required_if_edge_from_if_then(self) -> None:
        cpg = ConstraintPropagationGraph(IF_THEN_SCHEMA)
        req_if_edges = [e for e in cpg._edges if e.edge_type == EdgeType.REQUIRED_IF]
        assert len(req_if_edges) >= 1

    def test_required_if_source_and_target(self) -> None:
        cpg = ConstraintPropagationGraph(IF_THEN_SCHEMA)
        req_if_edges = [e for e in cpg._edges if e.edge_type == EdgeType.REQUIRED_IF]
        sources = {e.source for e in req_if_edges}
        targets = {e.target for e in req_if_edges}
        assert "type" in sources
        assert "advanced_options" in targets

    def test_required_if_metadata_has_condition_value(self) -> None:
        cpg = ConstraintPropagationGraph(IF_THEN_SCHEMA)
        req_if_edges = [e for e in cpg._edges if e.edge_type == EdgeType.REQUIRED_IF]
        assert any("condition_value" in e.metadata for e in req_if_edges)

    def test_no_required_if_edges_without_if_then(self) -> None:
        cpg = ConstraintPropagationGraph(FLAT_SCHEMA)
        req_if_edges = [e for e in cpg._edges if e.edge_type == EdgeType.REQUIRED_IF]
        assert len(req_if_edges) == 0


class TestExclusiveEdges:
    def test_exclusive_edges_from_one_of(self) -> None:
        cpg = ConstraintPropagationGraph(ONE_OF_SCHEMA)
        excl_edges = [e for e in cpg._edges if e.edge_type == EdgeType.EXCLUSIVE]
        assert len(excl_edges) >= 1

    def test_exclusive_edges_connect_non_overlapping_fields(self) -> None:
        cpg = ConstraintPropagationGraph(ONE_OF_SCHEMA)
        excl_edges = [e for e in cpg._edges if e.edge_type == EdgeType.EXCLUSIVE]
        sources = {e.source for e in excl_edges}
        targets = {e.target for e in excl_edges}
        # field_a and field_b should be in source/target sets
        assert "field_a" in sources or "field_a" in targets
        assert "field_b" in sources or "field_b" in targets

    def test_no_exclusive_edges_without_one_of(self) -> None:
        cpg = ConstraintPropagationGraph(FLAT_SCHEMA)
        excl_edges = [e for e in cpg._edges if e.edge_type == EdgeType.EXCLUSIVE]
        assert len(excl_edges) == 0


class TestBoundsValueEdges:
    def test_bounds_value_edge_for_array_with_min_items(self) -> None:
        cpg = ConstraintPropagationGraph(ARRAY_BOUNDS_SCHEMA)
        bounds_edges = [e for e in cpg._edges if e.edge_type == EdgeType.BOUNDS_VALUE]
        assert len(bounds_edges) >= 1

    def test_bounds_value_metadata_has_min_max(self) -> None:
        cpg = ConstraintPropagationGraph(ARRAY_BOUNDS_SCHEMA)
        bounds_edges = [e for e in cpg._edges if e.edge_type == EdgeType.BOUNDS_VALUE]
        assert any(e.metadata.get("min_items") == 2 for e in bounds_edges)
        assert any(e.metadata.get("max_items") == 5 for e in bounds_edges)


# ---------------------------------------------------------------------------
# Propagation tests
# ---------------------------------------------------------------------------


class TestBooleanAllPropagation:
    def test_false_child_determines_parent_false(self) -> None:
        cpg = ConstraintPropagationGraph(BOOLEAN_ALL_SCHEMA)
        result = cpg.propagate("premises[0].valid", False)
        assert "argument_valid" in result.determined_fields
        assert result.determined_fields["argument_valid"] is False

    def test_true_child_does_not_determine_parent(self) -> None:
        """Single true item cannot determine parent — more items may exist."""
        cpg = ConstraintPropagationGraph(BOOLEAN_ALL_SCHEMA)
        result = cpg.propagate("premises[0].valid", True)
        assert "argument_valid" not in result.determined_fields

    def test_false_child_produces_domain_reduction(self) -> None:
        cpg = ConstraintPropagationGraph(BOOLEAN_ALL_SCHEMA)
        result = cpg.propagate("premises[0].valid", False)
        assert len(result.reductions) >= 1
        reduction = result.reductions[0]
        assert reduction.field_path == "argument_valid"
        assert reduction.original_domain_size == 2
        assert reduction.new_domain_size == 1

    def test_false_string_value_also_triggers_reduction(self) -> None:
        cpg = ConstraintPropagationGraph(BOOLEAN_ALL_SCHEMA)
        result = cpg.propagate("premises[0].valid", "false")
        assert "argument_valid" in result.determined_fields

    def test_indexed_path_matches_wildcard_edge(self) -> None:
        cpg = ConstraintPropagationGraph(BOOLEAN_ALL_SCHEMA)
        # premises[5].valid should still match premises[*].valid edge
        result = cpg.propagate("premises[5].valid", False)
        assert "argument_valid" in result.determined_fields

    def test_propagation_trace_non_empty_after_reduction(self) -> None:
        cpg = ConstraintPropagationGraph(BOOLEAN_ALL_SCHEMA)
        cpg.propagate("premises[0].valid", False)
        trace = cpg.get_propagation_trace()
        assert len(trace) >= 1

    def test_propagation_result_type(self) -> None:
        cpg = ConstraintPropagationGraph(BOOLEAN_ALL_SCHEMA)
        result = cpg.propagate("premises[0].valid", False)
        assert isinstance(result, PropagationResult)

    def test_committed_fields_recorded(self) -> None:
        cpg = ConstraintPropagationGraph(BOOLEAN_ALL_SCHEMA)
        result = cpg.propagate("premises[0].valid", False)
        assert "premises[0].valid" in result.committed_fields


class TestEnumFailAnyPropagation:
    def test_fail_value_narrows_parent_to_fail_class(self) -> None:
        cpg = ConstraintPropagationGraph(ENUM_FAIL_ANY_SCHEMA)
        cpg.propagate("checks[0].result", "FAIL")
        # Parent should be determined or narrowed to {"FAIL"}
        overall_node = cpg._nodes.get("overall")
        assert overall_node is not None
        assert overall_node.current_domain is not None
        assert all(v == "FAIL" for v in overall_node.current_domain)

    def test_pass_value_does_not_narrow_parent(self) -> None:
        cpg = ConstraintPropagationGraph(ENUM_FAIL_ANY_SCHEMA)
        cpg.propagate("checks[0].result", "PASS")
        overall_node = cpg._nodes.get("overall")
        assert overall_node is not None
        # Domain still contains both PASS and FAIL (or more)
        assert len(overall_node.current_domain or set()) >= 2


class TestPropagationWithNoEdges:
    def test_flat_schema_propagation_returns_empty_determined(self) -> None:
        cpg = ConstraintPropagationGraph(FLAT_SCHEMA)
        result = cpg.propagate("name", "Alice")
        assert len(result.determined_fields) == 0

    def test_flat_schema_propagation_returns_empty_reductions(self) -> None:
        cpg = ConstraintPropagationGraph(FLAT_SCHEMA)
        result = cpg.propagate("age", 25)
        assert len(result.reductions) == 0


# ---------------------------------------------------------------------------
# Batch propagation tests
# ---------------------------------------------------------------------------


class TestBatchPropagation:
    def test_inconsistency_detected_true_parent_with_false_child(self) -> None:
        cpg = ConstraintPropagationGraph(BOOLEAN_ALL_SCHEMA)
        output = {
            "premises": [{"statement": "p1", "valid": False}],
            "argument_valid": True,
        }
        result = cpg.propagate_batch(output)
        assert len(result.inconsistencies) >= 1
        paths = [path for path, _ in result.inconsistencies]
        assert "argument_valid" in paths

    def test_consistent_output_no_inconsistencies(self) -> None:
        cpg = ConstraintPropagationGraph(BOOLEAN_ALL_SCHEMA)
        output = {
            "premises": [{"statement": "p1", "valid": True}],
            "argument_valid": True,
        }
        result = cpg.propagate_batch(output)
        # All premises true — no domain violation (parent stays True or is within domain)
        # No inconsistency for argument_valid=True when all items are True
        arg_inconsistencies = [
            path for path, _ in result.inconsistencies if path == "argument_valid"
        ]
        assert len(arg_inconsistencies) == 0

    def test_false_parent_with_false_child_consistent(self) -> None:
        cpg = ConstraintPropagationGraph(BOOLEAN_ALL_SCHEMA)
        output = {
            "premises": [{"statement": "p1", "valid": False}],
            "argument_valid": False,
        }
        result = cpg.propagate_batch(output)
        # argument_valid=False is consistent with premises[*].valid=False
        arg_inconsistencies = [
            path for path, _ in result.inconsistencies if path == "argument_valid"
        ]
        assert len(arg_inconsistencies) == 0

    def test_batch_on_empty_output(self) -> None:
        cpg = ConstraintPropagationGraph(BOOLEAN_ALL_SCHEMA)
        result = cpg.propagate_batch({})
        assert len(result.inconsistencies) == 0

    def test_batch_on_flat_schema_no_inconsistencies(self) -> None:
        cpg = ConstraintPropagationGraph(FLAT_SCHEMA)
        output = {"name": "Alice", "age": 30, "score": 0.9}
        result = cpg.propagate_batch(output)
        assert len(result.inconsistencies) == 0

    def test_enum_fail_any_inconsistency_detected(self) -> None:
        cpg = ConstraintPropagationGraph(ENUM_FAIL_ANY_SCHEMA)
        output = {
            "checks": [{"result": "FAIL"}],
            "overall": "PASS",  # Inconsistent — FAIL in checks but PASS in overall
        }
        result = cpg.propagate_batch(output)
        assert len(result.inconsistencies) >= 1


# ---------------------------------------------------------------------------
# Get determined/narrowed tests
# ---------------------------------------------------------------------------


class TestGetDeterminedFields:
    def test_no_determined_fields_before_propagation(self) -> None:
        cpg = ConstraintPropagationGraph(BOOLEAN_ALL_SCHEMA)
        assert len(cpg.get_determined_fields()) == 0

    def test_determined_fields_after_false_propagation(self) -> None:
        cpg = ConstraintPropagationGraph(BOOLEAN_ALL_SCHEMA)
        cpg.propagate("premises[0].valid", False)
        determined = cpg.get_determined_fields()
        assert "argument_valid" in determined
        assert determined["argument_valid"] is False


class TestGetNarrowedEnums:
    def test_no_narrowed_enums_for_flat_schema(self) -> None:
        cpg = ConstraintPropagationGraph(FLAT_SCHEMA)
        assert len(cpg.get_narrowed_enums()) == 0


# ---------------------------------------------------------------------------
# Reset tests
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_propagation_trace(self) -> None:
        cpg = ConstraintPropagationGraph(BOOLEAN_ALL_SCHEMA)
        cpg.propagate("premises[0].valid", False)
        assert len(cpg.get_propagation_trace()) > 0
        cpg.reset()
        assert len(cpg.get_propagation_trace()) == 0

    def test_reset_restores_initial_domains(self) -> None:
        cpg = ConstraintPropagationGraph(BOOLEAN_ALL_SCHEMA)
        cpg.propagate("premises[0].valid", False)
        # argument_valid domain is now {False}
        node = cpg._nodes["argument_valid"]
        assert node.current_domain == {False}
        cpg.reset()
        # After reset, domain should be back to {True, False}
        assert node.current_domain == {True, False}

    def test_reset_clears_committed_state(self) -> None:
        cpg = ConstraintPropagationGraph(BOOLEAN_ALL_SCHEMA)
        cpg.propagate("argument_valid", True)
        node = cpg._nodes.get("argument_valid")
        assert node is not None
        assert node.committed
        cpg.reset()
        assert not node.committed

    def test_reuse_after_reset(self) -> None:
        """Graph can be used for a second request after reset."""
        cpg = ConstraintPropagationGraph(BOOLEAN_ALL_SCHEMA)
        # First request — inconsistency
        output_1 = {
            "premises": [{"valid": False}],
            "argument_valid": True,
        }
        result_1 = cpg.propagate_batch(output_1)
        assert len(result_1.inconsistencies) >= 1

        # Second request — consistent
        output_2 = {
            "premises": [{"valid": False}],
            "argument_valid": False,
        }
        result_2 = cpg.propagate_batch(output_2)
        arg_issues = [p for p, _ in result_2.inconsistencies if p == "argument_valid"]
        assert len(arg_issues) == 0


# ---------------------------------------------------------------------------
# Public API tests
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_build_constraint_graph_returns_cpg_instance(self) -> None:
        cpg = build_constraint_graph(BOOLEAN_ALL_SCHEMA)
        assert isinstance(cpg, ConstraintPropagationGraph)

    def test_build_constraint_graph_empty_schema(self) -> None:
        cpg = build_constraint_graph({})
        assert cpg.get_node_count() == 0

    def test_graph_edge_dataclass(self) -> None:
        edge = GraphEdge(
            source="a",
            target="b",
            edge_type=EdgeType.DERIVES_FROM,
            metadata={"pattern": "boolean_all"},
        )
        assert edge.source == "a"
        assert edge.edge_type == EdgeType.DERIVES_FROM

    def test_multi_array_schema_builds_multiple_edges(self) -> None:
        cpg = ConstraintPropagationGraph(MULTI_ARRAY_SCHEMA)
        derives_edges = [e for e in cpg._edges if e.edge_type == EdgeType.DERIVES_FROM]
        # Should have edges for steps→overall_passed AND checks→summary_result
        assert len(derives_edges) >= 2
