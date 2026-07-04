"""Tests for output-truncation array continuation in Stage 4."""

from __future__ import annotations

import pytest

from nfield.extraction._sfep import truncated_json_arrays
from nfield.pipeline._state import PipelineState
from nfield.pipeline.s4_extract import (
    _continue_truncated_arrays,
    _ground_norm,
    _last_item_segment,
    _reparse_unclean_arrays,
)
from nfield.schema._types import CapacityLeaf, Field, Segment

REFS = Field("refs", "array", {"items": {"type": "string"}}, "", {})

ENTRY_1 = "Alpha and Beta. A study of segment routing in large systems. Journal A, 2021."
ENTRY_2 = "Gamma and Delta. Windowed extraction over long documents. Journal B, 2022."
ENTRY_3 = "Epsilon and Zeta. Continuation strategies for bounded outputs. Journal C, 2023."
ENTRY_4 = "Eta and Theta. Deduplicating merged item streams. Journal D, 2024."


def _segments() -> list[Segment]:
    body = "This paper surveys extraction systems and their limits in production use."
    part1 = f"References\n[1] {ENTRY_1}\n[2] {ENTRY_2}"
    part2 = f"[3] {ENTRY_3}\n[4] {ENTRY_4}"
    return [
        Segment(text=body, start=0, end=len(body), segment_type="unstructured", segment_id=0),
        Segment(
            text=part1, start=100, end=100 + len(part1), segment_type="unstructured", segment_id=1
        ),
        Segment(
            text=part2, start=400, end=400 + len(part2), segment_type="unstructured", segment_id=2
        ),
    ]


def _state() -> PipelineState:
    state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=1024, C_usable=4096.0)
    state.segments = _segments()
    return state


def _big_segment() -> Segment:
    # A document large enough that the document-sized continuation budget covers
    # many windows, so patience (not the per-document cap) governs the sweep.
    text = "reference body text " * 3_000
    return Segment(text=text, start=0, end=len(text), segment_type="unstructured", segment_id=0)


def _leaf() -> CapacityLeaf:
    return CapacityLeaf(
        fields=[REFS],
        groups=[],
        document_excerpt="",
        overhead=100.0,
        safe_output=1024,
        leaf_id=0,
    )


class ContinuationProvider:
    """Returns the remaining entries for every continuation window call."""

    context_window = 8192
    max_output_tokens = 1024
    model_name = "mock/model"

    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    async def complete(self, messages, *, max_tokens):
        self.calls += 1
        return self.response


class TestTruncatedDetection:
    def test_flags_unclosed_scalar_array(self):
        raw = f'refs = ["{ENTRY_1}", "{ENTRY_2}", "Epsilon and Zeta. Contin'
        assert truncated_json_arrays(raw, [REFS]) == {"refs"}

    def test_clean_array_not_flagged(self):
        raw = f'refs = ["{ENTRY_1}", "{ENTRY_2}"]'
        assert truncated_json_arrays(raw, [REFS]) == set()

    def test_flags_unclosed_object_array(self):
        rows = Field("rows", "array", {"items": {"type": "object", "properties": {}}}, "", {})
        raw = 'rows = [{"a": "x"}, {"a": "y"}, {"a": "cut'
        assert truncated_json_arrays(raw, [rows]) == {"rows"}

    def test_non_array_field_not_flagged(self):
        name = Field("name", "string", {}, "", {})
        assert truncated_json_arrays("name = [not an array value", [name]) == set()


class TestAnchor:
    def test_finds_segment_of_last_item(self):
        segments = _segments()
        norms = {s.segment_id: _ground_norm(s.text) for s in segments}
        seg = _last_item_segment([ENTRY_1, ENTRY_2], segments, norms)
        assert seg is not None and seg.segment_id == 1

    def test_unlocatable_items_return_none(self):
        segments = _segments()
        norms = {s.segment_id: s.text.casefold() for s in segments}
        assert (
            _last_item_segment(
                ["totally invented entry text that matches nothing"], segments, norms
            )
            is None
        )


class TestContinueTruncatedArrays:
    @pytest.mark.asyncio
    async def test_recovers_tail_items(self):
        state = _state()
        leaf = _leaf()
        extracted = {"refs": [ENTRY_1, ENTRY_2]}
        provider = ContinuationProvider(f'refs = ["{ENTRY_3}", "{ENTRY_4}"]')
        await _continue_truncated_arrays(leaf, provider, state, extracted, {"refs"})
        assert provider.calls >= 1
        assert extracted["refs"] == [ENTRY_1, ENTRY_2, ENTRY_3, ENTRY_4]

    @pytest.mark.asyncio
    async def test_duplicates_from_overlap_absorbed(self):
        state = _state()
        leaf = _leaf()
        extracted = {"refs": [ENTRY_1, ENTRY_2]}
        provider = ContinuationProvider(f'refs = ["{ENTRY_2}", "{ENTRY_3}"]')
        await _continue_truncated_arrays(leaf, provider, state, extracted, {"refs"})
        assert extracted["refs"] == [ENTRY_1, ENTRY_2, ENTRY_3]

    @pytest.mark.asyncio
    async def test_no_call_when_nothing_truncated(self):
        state = _state()
        leaf = _leaf()
        extracted = {"refs": [ENTRY_1, ENTRY_2]}
        provider = ContinuationProvider("refs = []")
        await _continue_truncated_arrays(leaf, provider, state, extracted, set())
        assert provider.calls == 0

    @pytest.mark.asyncio
    async def test_unlocatable_items_sweep_whole_document(self):
        # No anchor means the cut position is unknown; the array is re-extracted
        # window by window over the whole document rather than abandoned.
        state = _state()
        leaf = _leaf()
        extracted = {"refs": ["invented text never in the document, long enough to try"]}
        provider = ContinuationProvider(f'refs = ["{ENTRY_1}", "{ENTRY_2}", "{ENTRY_3}"]')
        await _continue_truncated_arrays(leaf, provider, state, extracted, {"refs"})
        assert provider.calls >= 1
        assert ENTRY_1 in extracted["refs"] and ENTRY_3 in extracted["refs"]

    @pytest.mark.asyncio
    async def test_empty_salvage_sweeps_whole_document(self):
        # The cut landed before the first complete item: nothing was salvaged,
        # so the array is extracted afresh over output-sized windows.
        state = _state()
        leaf = _leaf()
        extracted: dict = {"refs": []}
        provider = ContinuationProvider(f'refs = ["{ENTRY_1}", "{ENTRY_2}"]')
        await _continue_truncated_arrays(leaf, provider, state, extracted, {"refs"})
        assert provider.calls >= 1
        assert extracted["refs"] == [ENTRY_1, ENTRY_2]


class TestContinueTruncatedObjectArrays:
    @pytest.mark.asyncio
    async def test_nested_object_items_recovered(self):
        # Object items ground by their string leaves, so a truncated nested
        # array-of-objects continues exactly like a scalar one.
        rows = Field(
            "rows",
            "array",
            {
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                }
            },
            "",
            {},
        )
        seg_text = f"Table\n{ENTRY_1} | fast\n{ENTRY_2} | slow\n{ENTRY_3} | mixed"
        state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=1024, C_usable=4096.0)
        state.segments = [
            Segment(
                text=seg_text, start=0, end=len(seg_text), segment_type="tabular", segment_id=0
            )
        ]
        leaf = CapacityLeaf(
            fields=[rows],
            groups=[],
            document_excerpt="",
            overhead=100.0,
            safe_output=1024,
            leaf_id=0,
        )
        extracted = {"rows": [{"title": ENTRY_1, "tags": ["fast"]}]}
        provider = ContinuationProvider(
            f'rows = [{{"title": "{ENTRY_2}", "tags": ["slow"]}}, '
            f'{{"title": "{ENTRY_3}", "tags": ["mixed"]}}]'
        )
        await _continue_truncated_arrays(leaf, provider, state, extracted, {"rows"})
        assert provider.calls >= 1
        assert [r["title"] for r in extracted["rows"]] == [ENTRY_1, ENTRY_2, ENTRY_3]


class TestNullItemsDropped:
    def test_null_padded_array_collapses(self):
        from nfield.extraction._sfep import typecast

        assert typecast("[null]", REFS) == []
        assert typecast(f'[null, "{ENTRY_1}", null]', REFS) == [ENTRY_1]

    def test_citation_marker_items_dropped(self):
        from nfield.extraction._sfep import typecast

        raw = f'["[3, 4]", "[5]", "{ENTRY_1}", "[6-9]", "{ENTRY_2}"]'
        assert typecast(raw, REFS) == [ENTRY_1, ENTRY_2]

    def test_leading_bracket_ordinals_stripped_when_reordered(self):
        # Any numbered list (references, clauses) opens each item with its number;
        # a window-continued list is reordered so the numbers are not consecutive.
        from nfield.extraction._sfep import typecast

        raw = f'["[142] {ENTRY_2}", "[9] {ENTRY_1}", "[136] {ENTRY_3}"]'
        assert typecast(raw, REFS) == [ENTRY_2, ENTRY_1, ENTRY_3]

    def test_repeated_bracket_token_kept_as_content(self):
        # A repeated bracketed token (a year, a category) is content, not an ordinal.
        from nfield.extraction._sfep import typecast

        raw = '["[2023] Alpha annual report", "[2023] Beta annual report", "[2023] Gamma report"]'
        assert typecast(raw, REFS) == [
            "[2023] Alpha annual report",
            "[2023] Beta annual report",
            "[2023] Gamma report",
        ]

    def test_bare_null_is_absent_not_one_null_item(self):
        from nfield.extraction._sfep import typecast

        assert typecast("null", REFS) is None
        assert typecast("NULL", REFS) is None
        assert typecast("[NULL]", REFS) == []


class TestEmptyRecoveryKeepsOriginal:
    def test_empty_list_write_restores_quality_stash(self):
        from nfield.assembly._blackboard import Blackboard
        from nfield.pipeline.s4_extract import _write_extracted_to_blackboard

        state = _state()
        state.fields = [REFS]
        state.field_by_path = {"refs": REFS}
        state.blackboard = Blackboard(["refs"])
        state.blackboard.mark_pending("refs")
        state.quality_failed_values["refs"] = [ENTRY_1, ENTRY_2]
        _write_extracted_to_blackboard({"refs": []}, state)
        assert state.blackboard.get_filled()["refs"] == [ENTRY_1, ENTRY_2]

    def test_empty_list_write_kept_without_stash(self):
        from nfield.assembly._blackboard import Blackboard
        from nfield.pipeline.s4_extract import _write_extracted_to_blackboard

        state = _state()
        state.fields = [REFS]
        state.field_by_path = {"refs": REFS}
        state.blackboard = Blackboard(["refs"])
        state.blackboard.mark_pending("refs")
        _write_extracted_to_blackboard({"refs": []}, state)
        assert state.blackboard.get_filled()["refs"] == []


class ScriptedProvider:
    """Returns one scripted response per call, then repeats the last."""

    context_window = 8192
    max_output_tokens = 1024
    model_name = "mock/model"

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls = 0

    async def complete(self, messages, *, max_tokens):
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


class TestScarcePatience:
    @pytest.mark.asyncio
    async def test_scarce_array_sweeps_past_tight_stop(self):
        from nfield.pipeline.s4_extract import _sweep_array_windows

        state = _state()
        # The per-document window budget is document-sized; give patience room to run.
        state.segments = [_big_segment()]
        leaf = _leaf()
        extracted = {"refs": [ENTRY_1]}
        windows = ["w1", "w2", "w3", "w4", "w5"]
        provider = ScriptedProvider(
            ["refs = []"] * 4 + [f'refs = ["{ENTRY_2}", "{ENTRY_3}", "{ENTRY_4}"]']
        )
        await _sweep_array_windows(windows, [REFS], leaf, provider, state, extracted)
        assert provider.calls == 5
        assert extracted["refs"] == [ENTRY_1, ENTRY_2, ENTRY_3, ENTRY_4]

    @pytest.mark.asyncio
    async def test_stocked_array_stops_after_patience(self):
        from nfield.pipeline.s4_extract import _CONTINUATION_STOP_AFTER_EMPTY, _sweep_array_windows

        state = _state()
        # The per-document window budget is document-sized; give patience room to run.
        state.segments = [_big_segment()]
        leaf = _leaf()
        extracted = {"refs": [ENTRY_1, ENTRY_2, ENTRY_3]}
        windows = [f"w{i}" for i in range(20)]
        provider = ScriptedProvider(["refs = []"])
        await _sweep_array_windows(windows, [REFS], leaf, provider, state, extracted)
        # A stocked array stops after the patience run of empty windows, not the whole doc.
        assert provider.calls == _CONTINUATION_STOP_AFTER_EMPTY


class TestOrdinalRun:
    def test_consecutive_ints_are_a_run(self):
        from nfield.pipeline.s4_extract import _is_ordinal_run

        assert _is_ordinal_run([1, 2, 3, 4, 5])
        assert _is_ordinal_run(["636", "637", "638", "639"])

    def test_real_values_are_not_a_run(self):
        from nfield.pipeline.s4_extract import _is_ordinal_run

        assert not _is_ordinal_run([ENTRY_1, ENTRY_2, ENTRY_3])
        assert not _is_ordinal_run([90210, 10001, 60601])  # non-consecutive ids
        assert not _is_ordinal_run([1, 2])  # too short to judge

    @pytest.mark.asyncio
    async def test_run_window_retried_and_never_adopted(self):
        # A window answering with entry numbers is retried once; if the retry also
        # fails, the run is discarded rather than merged.
        from nfield.pipeline.s4_extract import _sweep_array_windows

        state = _state()
        state.segments = [_big_segment()]
        leaf = _leaf()
        extracted: dict = {"refs": [ENTRY_1]}
        provider = ScriptedProvider(
            ["refs = [1, 2, 3, 4, 5]", f'refs = ["{ENTRY_2}", "{ENTRY_3}"]']
        )
        await _sweep_array_windows(["w0"], [REFS], leaf, provider, state, extracted)
        assert provider.calls == 2  # the corrective retry ran
        assert extracted["refs"] == [ENTRY_1, ENTRY_2, ENTRY_3]  # retry adopted, no numbers


class TestSandwichResample:
    @pytest.mark.asyncio
    async def test_low_yield_window_between_productive_neighbours_resampled(self):
        from nfield.pipeline.s4_extract import _sweep_array_windows

        state = _state()
        state.segments = [_big_segment()]
        leaf = _leaf()
        extracted: dict = {"refs": []}

        def batch(tag: str, n: int) -> str:
            items = ", ".join(
                f'"{tag} entry number {i} with plenty of text to count"' for i in range(n)
            )
            return f"refs = [{items}]"

        # Windows 0 and 2 are dense (10 items); window 1 under-emits (1 item) and is
        # re-probed as two halves after the pass, recovering its 10.
        provider = ScriptedProvider(
            [batch("alpha", 10), batch("beta", 1), batch("gamma", 10), batch("beta-full", 10)]
        )
        await _sweep_array_windows(
            ["w0", "part-a\n\npart-b", "w2"], [REFS], leaf, provider, state, extracted
        )
        assert provider.calls == 5  # three windows + two half-window resamples
        assert sum(1 for x in extracted["refs"] if x.startswith("beta-full")) == 10

    @pytest.mark.asyncio
    async def test_deduped_neighbour_still_carries_density(self):
        # An overlapped neighbour re-emits items already merged (zero NEW yield) but
        # its RAW count still proves the region is dense, so the sandwiched
        # under-emitter must still be re-probed.
        from nfield.pipeline.s4_extract import _sweep_array_windows

        state = _state()
        state.segments = [_big_segment()]
        leaf = _leaf()
        extracted: dict = {"refs": []}

        def batch(tag: str, n: int) -> str:
            items = ", ".join(
                f'"{tag} entry number {i} with plenty of text to count"' for i in range(n)
            )
            return f"refs = [{items}]"

        # w2 repeats w0's items verbatim: raw 10, new yield 0.
        provider = ScriptedProvider(
            [batch("alpha", 10), batch("beta", 1), batch("alpha", 10), batch("beta-full", 10)]
        )
        await _sweep_array_windows(
            ["w0", "part-a\n\npart-b", "w2"], [REFS], leaf, provider, state, extracted
        )
        assert provider.calls == 5
        assert sum(1 for x in extracted["refs"] if x.startswith("beta-full")) == 10


class TestNeighborPromotion:
    @pytest.mark.asyncio
    async def test_productive_window_promotes_document_neighbours(self):
        # Items cluster: after window 3 yields, its doc neighbours (2 and 4) must be
        # visited next, ahead of the rest of the visit order.
        from nfield.pipeline.s4_extract import _sweep_array_windows

        state = _state()
        state.segments = [_big_segment()]
        leaf = _leaf()
        extracted: dict = {"refs": []}
        windows = [f"w{i}" for i in range(6)]
        seen: list[str] = []

        class _Recorder(ScriptedProvider):
            async def complete(self, messages, *, max_tokens):
                content = messages[-1]["content"]
                for i in range(6):
                    if f"w{i}" in content:
                        seen.append(f"w{i}")
                        break
                return await super().complete(messages, max_tokens=max_tokens)

        # Visit order starts at 3 (highest relevance); it yields, so 2 and 4 follow.
        provider = _Recorder(
            [f'refs = ["{ENTRY_1}", "{ENTRY_2}", "{ENTRY_3}"]'] + ["refs = []"] * 10
        )
        await _sweep_array_windows(
            windows, [REFS], leaf, provider, state, extracted, visit_order=[3, 0, 5, 1, 2, 4]
        )
        assert seen[:3] == ["w3", "w2", "w4"]


DIM_METRIC = Field(
    "revenue",
    "array",
    {
        "items": {
            "type": "object",
            "properties": {
                "segment_type": {
                    "enum": ["company", "business_segment", "geographic_segment"],
                    "type": "string",
                },
                "value": {"type": "number"},
            },
        }
    },
    "",
    {},
)


class TestDimensionArrayWholeDocSweep:
    @pytest.mark.asyncio
    async def test_partially_filled_dimension_array_sweeps_whole_document(self):
        # A dimension array (enum item axis) already holding rows must still sweep the
        # WHOLE document, since its per-segment rows sit in tables already inside the
        # excerpt - the uncovered-remainder-only path would miss them.
        from nfield.pipeline.s4_extract import _extend_arrays_over_windows

        state = _state()
        state.segments = [_big_segment()]
        leaf = _leaf()
        leaf.fields = [DIM_METRIC]
        # Every segment is already "covered" by the excerpt; a remainder-only sweep
        # would therefore do nothing.
        leaf.excerpt_segment_ids = {s.segment_id for s in state.segments}
        extracted = {"revenue": [{"segment_type": "company", "value": 100}]}
        provider = ScriptedProvider(
            ['revenue = [{"segment_type": "geographic_segment", "value": 5}]']
        )
        await _extend_arrays_over_windows(leaf, provider, state, extracted)
        assert provider.calls >= 1  # swept despite full excerpt coverage
        assert len(extracted["revenue"]) >= 2  # picked up a new per-segment row


class TestDimensionSweepRelevanceOrder:
    @pytest.mark.asyncio
    async def test_dimension_sweep_visits_high_relevance_window_first(self):
        # A dimension array's disaggregation table sits in a high-relevance region;
        # the sweep must reach it first, even when it is late in document order.
        from nfield.pipeline.s4_extract import _extend_arrays_over_windows
        from nfield.schema._types import FieldGroup

        early = Segment(
            text="prose " * 400, start=0, end=2400, segment_type="unstructured", segment_id=0
        )
        table = Segment(
            text="Revenue by axis table with the segment figures",
            start=2400,
            end=2450,
            segment_type="tabular",
            segment_id=1,
        )
        state = _state()
        state.segments = [early, table]
        leaf = _leaf()
        leaf.fields = [DIM_METRIC]
        # Retrieval ranks the late table far above the early prose for this leaf.
        leaf.groups = [
            FieldGroup(
                parent_path="",
                fields=[DIM_METRIC],
                matched_segments=[early, table],
                segment_scores=[0.01, 9.0],
            )
        ]
        leaf.excerpt_segment_ids = {0, 1}
        extracted = {"revenue": [{"segment_type": "company", "value": 1}]}
        seen: list[str] = []

        class _Recorder(ScriptedProvider):
            async def complete(self, messages, *, max_tokens):
                seen.append(messages[-1]["content"])
                return await super().complete(messages, max_tokens=max_tokens)

        provider = _Recorder(["revenue = []"])
        await _extend_arrays_over_windows(leaf, provider, state, extracted)
        assert seen, "the dimension array must be swept"
        assert "table with the segment figures" in seen[0]  # high-relevance window first


class TestSplitKeepsWindowMachinery:
    @pytest.mark.asyncio
    async def test_split_leaf_still_extends_arrays(self):
        from nfield.assembly._blackboard import Blackboard
        from nfield.pipeline.s4_extract import _extract_leaf

        class _OverflowThenScripted(ScriptedProvider):
            overflowed = False

            async def complete(self, messages, *, max_tokens):
                if not self.overflowed:
                    self.overflowed = True
                    raise RuntimeError(
                        "Error code: 400 - Please reduce the length of the messages or completion."
                    )
                return await super().complete(messages, max_tokens=max_tokens)

        state = _state()
        filler = "Filler prose about methodology and background material. " * 100
        state.segments.append(
            Segment(
                text=filler,
                start=600,
                end=600 + len(filler),
                segment_type="unstructured",
                segment_id=3,
            )
        )
        state.fields = [REFS]
        state.field_by_path = {"refs": REFS}
        state.blackboard = Blackboard(["refs"])
        leaf = _leaf()
        leaf.document_excerpt = state.segments[1].text
        leaf.excerpt_segment_ids = {1}
        provider = _OverflowThenScripted(
            [
                f'refs = ["{ENTRY_1}", "{ENTRY_2}"]',
                f'refs = ["{ENTRY_3}", "{ENTRY_4}"]',
            ]
        )
        await _extract_leaf(leaf, provider, state)
        filled = state.blackboard.get_filled()["refs"]
        assert ENTRY_1 in filled
        assert ENTRY_3 in filled or ENTRY_4 in filled


class TestReparseSkipsTruncated:
    @pytest.mark.asyncio
    async def test_truncated_paths_not_resampled(self):
        state = _state()
        leaf = _leaf()
        raw = f'refs = ["{ENTRY_1}", "{ENTRY_2}", "Epsilon and Zeta. Contin'
        extracted = {"refs": [ENTRY_1, ENTRY_2]}
        provider = ContinuationProvider("refs = []")
        await _reparse_unclean_arrays(raw, leaf, provider, state, extracted, skip={"refs"})
        assert provider.calls == 0
