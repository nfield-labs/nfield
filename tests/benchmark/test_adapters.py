"""Competitor adapter tests — accounting and failure classification, no API.

The live extraction quality is exercised by the manual sweep; here the concern
is that each adapter normalises into a correct AdapterOutput and that failures
are classified honestly (transport error = call-failed; parse/refusal = method
miss in the denominator).
"""

from __future__ import annotations

from benchmark.adapters import _common
from benchmark.adapters._base import AdapterOutput

_SCHEMA = {
    "type": "object",
    "properties": {
        "a": {"type": "string"},
        "nested": {"type": "object", "properties": {"n": {"type": "integer"}}},
    },
}


class TestCommonAccounting:
    def test_schema_field_count_recurses_objects(self):
        assert _common.schema_field_count(_SCHEMA) == 2

    def test_count_nonempty_leaves_ignores_empties(self):
        assert _common.count_nonempty_leaves({"a": "x", "b": None, "c": "", "d": [1, 2]}) == 3

    def test_model_id_strips_provider_prefix(self):
        assert _common.model_id("groq/llama-3.3-70b-versatile") == "llama-3.3-70b-versatile"
        assert _common.model_id("llama-3.3-70b") == "llama-3.3-70b"

    def test_parse_json_object_tolerates_fences_and_prose(self):
        assert _common.parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
        assert _common.parse_json_object('here it is: {"a": 2} done') == {"a": 2}

    def test_parse_json_object_raises_when_no_object(self):
        try:
            _common.parse_json_object("no json here")
        except ValueError as exc:
            assert "no JSON object" in str(exc)
        else:
            raise AssertionError("expected ValueError")

    def test_success_output_counts_k_one(self):
        out = _common.success_output({"a": "x"}, _SCHEMA, 1.5)
        assert out.k == 1 and out.k_min == 1
        assert out.fields_total == 2
        assert out.fields_extracted == 1
        assert out.elapsed_seconds == 1.5


class TestFailureClassification:
    def test_parse_failure_is_method_miss_not_call_failed(self):
        out = _common.failure_output(_SCHEMA, 0.1, ValueError("no JSON object in model response"))
        assert out.failed
        assert out.call_failed == 0  # the method produced bad output — it owns the miss
        assert out.fields_extracted == 0

    def test_transport_error_is_call_failed(self, monkeypatch):
        # Simulate a groq transport error class so is_call_failure() recognises it.
        import groq

        out = _common.failure_output(_SCHEMA, 0.1, groq.APITimeoutError(request=None))  # type: ignore[arg-type]
        assert out.call_failed == 2  # both fields lost to the call, not the model

    def test_failure_output_keeps_fields_in_denominator(self):
        out = _common.failure_output(_SCHEMA, 0.1, ValueError("bad"))
        assert out.fields_total == 2  # never dropped


def test_all_adapters_registered():
    from benchmark.runner import ADAPTERS

    assert set(ADAPTERS) == {
        "nfield",
        "raw_prompt",
        "native_json",
        "instructor",
        "langchain",
    }


def test_instructions_lead_the_competitor_user_message():
    msgs = _common.messages(
        "doc",
        {"type": "object", "properties": {}},
        context_window=8192,
        max_output_tokens=2048,
        instructions="DOMAIN HINT.",
    )
    system, user = msgs[0]["content"], msgs[1]["content"]
    # Instructions lead the USER message (the channel the model follows), not system.
    assert user.startswith("DOMAIN HINT.")
    assert "DOMAIN HINT." not in system
    assert "OUTPUT" not in system  # the competitor system prompt is generic, not nfield's SFEP
    # No instructions -> user message starts straight at the schema, no leading blank line.
    bare = _common.messages(
        "doc", {"properties": {}}, context_window=8192, max_output_tokens=2048
    )[1]["content"]
    assert not bare.startswith("\n")


def test_registered_gold_datasets_carry_instructions():
    from benchmark import datasets

    for name in ("clinicaltrial", "factbook_us", "factbook_multi"):
        instr = datasets.get(name).instructions
        assert instr and "exactly as written" in instr


def test_adapters_satisfy_protocol():
    from benchmark.adapters._base import Adapter
    from benchmark.runner import ADAPTERS

    for factory in ADAPTERS.values():
        adapter = factory()
        assert isinstance(adapter, Adapter)
        assert isinstance(adapter.name, str) and adapter.name


def test_raw_and_native_report_their_names():
    from benchmark.adapters.native_json_adapter import NativeJsonAdapter
    from benchmark.adapters.raw_prompt_adapter import RawPromptAdapter

    assert RawPromptAdapter().name == "raw_prompt"
    assert NativeJsonAdapter().name == "native_json"


def test_output_failed_property():
    assert AdapterOutput(data={}, fields_total=1, fields_extracted=0, k=0, k_min=0).failed is False
    assert (
        AdapterOutput(data={}, fields_total=1, fields_extracted=0, k=0, k_min=0, error="x").failed
        is True
    )
