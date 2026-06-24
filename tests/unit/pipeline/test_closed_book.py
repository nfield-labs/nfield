"""Tests for closed-book extraction with self-consistency abstention."""

from __future__ import annotations

import asyncio

from formatshield import AsyncFormatShield
from formatshield.config import ExtractionConfig
from formatshield.extraction._papt import TemplateType
from formatshield.extraction._prompt import build_extraction_prompt
from formatshield.extraction._sfep import NEEDS_REVALIDATION
from formatshield.pipeline.s4_extract import _self_consistent
from formatshield.schema._types import Field

_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "city": {"type": "string"},
    },
    "required": ["name"],
}


# ---------------------------------------------------------------------------
# Closed-book prompt
# ---------------------------------------------------------------------------


class TestClosedBookPrompt:
    def test_closed_book_system_prompt_is_positive_knowledge_framing(self) -> None:
        f = Field("name", "string", {}, "", {})
        msgs = build_extraction_prompt([f], "", TemplateType.STANDARD, closed_book=True)
        system = msgs[0]["content"]
        assert "drawing on what you reliably know" in system  # positive knowledge task
        assert "If you are not certain, write NULL" in system  # abstention kept
        assert "document above" not in system  # no document-grounded framing
        assert "No document" not in system  # no absence framing

    def test_default_prompt_unchanged_for_document_path(self) -> None:
        f = Field("name", "string", {}, "", {})
        system = build_extraction_prompt([f], "doc", TemplateType.STANDARD)[0]["content"]
        assert "document" in system.lower()
        assert "No document is provided" not in system


# ---------------------------------------------------------------------------
# Self-consistency merge
# ---------------------------------------------------------------------------


class TestSelfConsistent:
    def test_agreement_is_kept(self) -> None:
        assert _self_consistent({"a": 1, "b": "x"}, {"a": 1, "b": "x"}) == {"a": 1, "b": "x"}

    def test_disagreement_abstains(self) -> None:
        # age 30 vs 31 disagree -> dropped (NULL by abstention); name agrees -> kept.
        merged = _self_consistent({"name": "Alice", "age": 30}, {"name": "Alice", "age": 31})
        assert merged == {"name": "Alice"}

    def test_null_in_either_sample_abstains(self) -> None:
        assert _self_consistent({"a": None}, {"a": None}) == {}
        assert _self_consistent({"a": 5}, {}) == {}  # second sample missing the field

    def test_revalidation_sentinel_abstains(self) -> None:
        merged = _self_consistent({"a": NEEDS_REVALIDATION}, {"a": NEEDS_REVALIDATION})
        assert merged == {}

    def test_falsy_but_concrete_values_are_kept(self) -> None:
        # 0 / False / "" are valid agreed values, not abstentions.
        assert _self_consistent({"a": 0}, {"a": 0}) == {"a": 0}
        assert _self_consistent({"a": False}, {"a": False}) == {"a": False}
        assert _self_consistent({"a": ""}, {"a": ""}) == {"a": ""}


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------


class _SeqProvider:
    """Returns canned completions; records every prompt and the document excerpt seen."""

    model_name = "mock/echo"
    context_window = 8192
    max_output_tokens = 8192

    def __init__(self, completion: str) -> None:
        self._completion = completion
        self.user_messages: list[str] = []

    async def complete(self, messages, *, max_tokens):
        self.user_messages.append("\n".join(m["content"] for m in messages))
        return self._completion

    async def count_tokens(self, text):
        return max(1, len(text) // 4)


def test_closed_book_run_sets_answer_rate_and_ignores_document(monkeypatch) -> None:
    # Both samples return the same values -> all agree -> answered. The provider always
    # gives the same completion, so every field is self-consistent.
    provider = _SeqProvider("name = Alice\nage = 30\ncity = Paris")
    monkeypatch.setattr("formatshield.engine._async.from_model", lambda *a, **k: provider)
    engine = AsyncFormatShield("mock/echo", _SCHEMA, config=ExtractionConfig(closed_book=True))

    # A non-empty document is passed but must be IGNORED in closed-book mode.
    result = asyncio.run(engine.extract("SECRET-DOCUMENT-TEXT", _SCHEMA))

    assert result.data == {"name": "Alice", "age": 30, "city": "Paris"}
    assert result.metadata.answer_rate == 1.0
    assert result.metadata.abstain_rate == 0.0
    assert result.metadata.hallucination_rate is None  # no source to ground against
    # The user's document never reached the model (closed-book ignores it).
    assert all("SECRET-DOCUMENT-TEXT" not in msg for msg in provider.user_messages)


def test_self_consistency_doubles_calls(monkeypatch) -> None:
    # Default closed-book is single-pass (one call per leaf); self_consistency=True samples
    # each leaf twice, so the call count exactly doubles.
    single = _SeqProvider("name = Alice\nage = 30\ncity = Paris")
    monkeypatch.setattr("formatshield.engine._async.from_model", lambda *a, **k: single)
    e1 = AsyncFormatShield("mock/echo", _SCHEMA, config=ExtractionConfig(closed_book=True))
    asyncio.run(e1.extract("", _SCHEMA))

    double = _SeqProvider("name = Alice\nage = 30\ncity = Paris")
    monkeypatch.setattr("formatshield.engine._async.from_model", lambda *a, **k: double)
    e2 = AsyncFormatShield(
        "mock/echo", _SCHEMA, config=ExtractionConfig(closed_book=True, self_consistency=True)
    )
    asyncio.run(e2.extract("", _SCHEMA))

    assert len(single.user_messages) >= 1
    assert len(double.user_messages) == 2 * len(single.user_messages)


def test_closed_book_abstention_is_not_recovered(monkeypatch) -> None:
    # A deliberate abstention (tracked in state.abstained) is excluded from the recovery
    # pool, so it is not re-extracted and the provider is never called.
    from formatshield.assembly._blackboard import Blackboard, FieldState
    from formatshield.pipeline._state import PipelineState
    from formatshield.pipeline.s5b_recover import run_recovery_pass

    state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=1024, C_usable=4096.0)
    state.closed_book = True
    bb = Blackboard(["a"])
    bb.mark_failed("a", "field not found in document (LLM output NULL)")
    state.abstained = {"a"}  # the model deliberately abstained on this field
    state.blackboard = bb

    class _Boom:
        model_name = "mock"
        context_window = 8192
        max_output_tokens = 8192

        async def complete(self, *a, **k):
            raise AssertionError("an abstained field must not be re-extracted")

        async def count_tokens(self, text):
            return 1

    result = asyncio.run(run_recovery_pass(state, _Boom(), ExtractionConfig(closed_book=True)))
    assert result is state
    assert bb.get_state("a") == FieldState.FAILED  # abstention left as-is, not recovered


def test_document_run_leaves_answer_rate_none(monkeypatch) -> None:
    provider = _SeqProvider("name = Alice")
    monkeypatch.setattr("formatshield.engine._async.from_model", lambda *a, **k: provider)
    engine = AsyncFormatShield("mock/echo", _SCHEMA)  # closed_book=False (default)
    result = asyncio.run(engine.extract("Alice lives here.", _SCHEMA))
    assert result.metadata.answer_rate is None
    assert result.metadata.abstain_rate is None
