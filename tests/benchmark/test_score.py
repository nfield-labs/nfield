"""Gold-diff scorer tests — every match rule and every error bucket.

The scorer is the one new measurement the benchmark adds, so its correctness is
exercised exhaustively and deterministically (no API, no clock).
"""

from __future__ import annotations

import pytest

from benchmark.score import FieldType, Outcome, score


def _schema(properties: dict) -> dict:
    return {"type": "object", "properties": properties}


class TestScalarMatchRules:
    def test_integer_exact_match_is_correct(self):
        schema = _schema({"n": {"type": "integer"}})
        report = score({"n": 42}, {"n": 42}, schema)
        assert report.value_accuracy == 1.0
        assert report.fields[0].field_type is FieldType.INTEGER

    def test_integer_coerces_string_digits(self):
        schema = _schema({"n": {"type": "integer"}})
        assert score({"n": "1,000"}, {"n": 1000}, schema).value_accuracy == 1.0

    def test_integer_wrong_value_is_accuracy_error(self):
        schema = _schema({"n": {"type": "integer"}})
        report = score({"n": 41}, {"n": 42}, schema)
        assert report.value_accuracy == 0.0
        assert report.outcomes[Outcome.ACCURACY] == 1

    def test_boolean_exact_and_textual(self):
        schema = _schema({"flag": {"type": "boolean"}})
        assert score({"flag": True}, {"flag": True}, schema).value_accuracy == 1.0
        assert score({"flag": "true"}, {"flag": True}, schema).value_accuracy == 1.0
        assert score({"flag": False}, {"flag": True}, schema).value_accuracy == 0.0

    def test_enum_is_normalized_exact(self):
        schema = _schema({"s": {"type": "string", "enum": ["COMPLETED", "ACTIVE"]}})
        report = score({"s": "completed"}, {"s": "COMPLETED"}, schema)
        assert report.fields[0].field_type is FieldType.ENUM
        assert report.value_accuracy == 1.0

    def test_number_within_tolerance(self):
        schema = _schema({"x": {"type": "number"}})
        assert score({"x": 3.14}, {"x": 3.14}, schema).value_accuracy == 1.0
        assert score({"x": 3.2}, {"x": 3.14}, schema).value_accuracy == 0.0

    def test_number_tolerance_can_be_widened(self):
        schema = _schema({"x": {"type": "number"}})
        report = score({"x": 100.0}, {"x": 100.05}, schema, numeric_tolerance=0.01)
        assert report.value_accuracy == 1.0


class TestStringMatchRules:
    def test_short_string_folds_case_whitespace_diacritics(self):
        schema = _schema({"name": {"type": "string"}})
        report = score({"name": "  Kutúzov "}, {"name": "kutuzov"}, schema)
        assert report.fields[0].field_type is FieldType.SHORT_STRING
        assert report.value_accuracy == 1.0

    def test_short_string_mismatch_is_accuracy_error(self):
        schema = _schema({"name": {"type": "string"}})
        assert score({"name": "Pierre"}, {"name": "Andrei"}, schema).value_accuracy == 0.0

    def test_long_string_within_edit_budget(self):
        gold = "Napoleon invaded Russia in the summer of 1812 with the Grande Armee, a vast force."
        near = "Napoleon invaded Russia in the summer of 1812 with the Grande Armee, a vast force!"
        schema = _schema({"summary": {"type": "string"}})
        report = score({"summary": near}, {"summary": gold}, schema)
        assert report.fields[0].field_type is FieldType.LONG_STRING
        assert report.value_accuracy == 1.0

    def test_long_string_beyond_edit_budget_fails(self):
        gold = "Napoleon invaded Russia in the summer of 1812 with the Grande Armee, a vast force."
        far = (
            "A completely different sentence about something else entirely, unrelated to the war."
        )
        schema = _schema({"summary": {"type": "string"}})
        assert score({"summary": far}, {"summary": gold}, schema).value_accuracy == 0.0


class TestErrorDecomposition:
    def test_omission_when_value_absent(self):
        schema = _schema({"a": {"type": "string"}, "b": {"type": "string"}})
        report = score({"a": "x"}, {"a": "x", "b": "y"}, schema)
        assert report.outcomes[Outcome.OMISSION] == 1
        assert report.value_accuracy == 0.5
        assert report.coverage == 0.5

    def test_hallucination_when_gold_empty_but_value_produced(self):
        schema = _schema({"a": {"type": "string"}})
        report = score({"a": "invented"}, {"a": None}, schema)
        assert report.outcomes[Outcome.HALLUCINATION] == 1
        assert report.value_accuracy == 0.0

    def test_correct_absence_when_gold_empty_and_no_value(self):
        schema = _schema({"a": {"type": "string"}})
        report = score({}, {"a": None}, schema)
        assert report.outcomes[Outcome.CORRECT] == 1
        assert report.json_pass is True

    def test_structural_when_container_where_scalar_due(self):
        schema = _schema({"a": {"type": "string"}})
        report = score({"a": {"nested": 1}}, {"a": "x"}, schema)
        assert report.outcomes[Outcome.STRUCTURAL] == 1
        assert report.json_pass is False

    def test_outcome_buckets_are_disjoint_and_total(self):
        schema = _schema({f"f{i}": {"type": "string"} for i in range(4)})
        gold = {"f0": "a", "f1": "b", "f2": "c", "f3": None}
        extracted = {"f0": "a", "f1": "WRONG", "f3": "halluc"}
        report = score(extracted, gold, schema)
        assert sum(report.outcomes.values()) == report.n_fields == 4


class TestNestingAndArrays:
    def test_nested_paths_resolve_and_match(self):
        schema = _schema({"book": {"type": "object", "properties": {"title": {"type": "string"}}}})
        report = score(
            {"book": {"title": "War and Peace"}}, {"book.title": "war and peace"}, schema
        )
        assert report.value_accuracy == 1.0

    def test_array_items_flatten_to_item_paths(self):
        schema = _schema(
            {
                "ids": {
                    "type": "object",
                    "properties": {"item_0": {"type": "string"}},
                }
            }
        )
        # Predicted list flattens to ids.item_0; gold key uses the same convention.
        report = score({"ids": ["NCT001"]}, {"ids.item_0": "nct001"}, schema)
        assert report.value_accuracy == 1.0

    def test_array_item_type_resolves_through_items(self):
        schema = _schema({"codes": {"type": "array", "items": {"type": "integer"}}})
        report = score({"codes": [7]}, {"codes.item_0": 7}, schema)
        assert report.fields[0].field_type is FieldType.INTEGER
        assert report.value_accuracy == 1.0


class TestReportShape:
    def test_per_type_breakdown_tracks_each_type(self):
        schema = _schema({"n": {"type": "integer"}, "name": {"type": "string"}})
        report = score({"n": 1, "name": "x"}, {"n": 1, "name": "y"}, schema)
        assert report.by_type[FieldType.INTEGER].accuracy == 1.0
        assert report.by_type[FieldType.SHORT_STRING].accuracy == 0.0

    def test_call_failed_passes_through_as_its_own_category(self):
        schema = _schema({"a": {"type": "string"}})
        report = score({}, {"a": "x"}, schema, call_failed=3)
        assert report.call_failed == 3
        assert report.outcomes[Outcome.OMISSION] == 1

    def test_type_inferred_when_path_absent_from_schema(self):
        report = score({"x": 5}, {"x": 5}, _schema({}))
        assert report.fields[0].field_type is FieldType.INTEGER
        assert report.value_accuracy == 1.0

    def test_empty_gold_does_not_divide_by_zero(self):
        report = score({"a": 1}, {}, _schema({}))
        assert report.n_fields == 0
        assert report.value_accuracy == 0.0
        assert report.coverage == 0.0


@pytest.mark.parametrize(
    ("gold", "predicted", "expected"),
    [
        (5, 5, 1.0),
        (5, 6, 0.0),
        ("Acme Corp", "ACME  corp", 1.0),
        (True, "yes", 1.0),
    ],
)
def test_match_table(gold, predicted, expected):
    report = score({"f": predicted}, {"f": gold}, {"type": "object", "properties": {}})
    assert report.value_accuracy == expected
