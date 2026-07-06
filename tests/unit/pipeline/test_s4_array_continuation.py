"""Tests for output-truncation array continuation in Stage 4."""

from __future__ import annotations

import pytest

from nfield.extraction._sfep import truncated_json_arrays
from nfield.pipeline._state import PipelineState
from nfield.pipeline.s4_extract import (
    _continue_truncated_arrays,
    _ground_norm,
    _last_item_segment,
    _merge_window_items,
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

    def test_smaller_reextraction_restores_quality_stash(self):
        # One recovery call cannot out-collect a windowed sweep: a shorter redo
        # is partial, and the fuller stashed original must win.
        from nfield.assembly._blackboard import Blackboard
        from nfield.pipeline.s4_extract import _write_extracted_to_blackboard

        state = _state()
        state.fields = [REFS]
        state.field_by_path = {"refs": REFS}
        state.blackboard = Blackboard(["refs"])
        state.blackboard.mark_pending("refs")
        state.quality_failed_values["refs"] = [ENTRY_1, ENTRY_2, ENTRY_3, ENTRY_4]
        _write_extracted_to_blackboard({"refs": [ENTRY_1]}, state)
        assert state.blackboard.get_filled()["refs"] == [ENTRY_1, ENTRY_2, ENTRY_3, ENTRY_4]

    def test_fuller_reextraction_replaces_quality_stash(self):
        from nfield.assembly._blackboard import Blackboard
        from nfield.pipeline.s4_extract import _write_extracted_to_blackboard

        state = _state()
        state.fields = [REFS]
        state.field_by_path = {"refs": REFS}
        state.blackboard = Blackboard(["refs"])
        state.blackboard.mark_pending("refs")
        state.quality_failed_values["refs"] = [ENTRY_1]
        _write_extracted_to_blackboard({"refs": [ENTRY_2, ENTRY_3]}, state)
        assert state.blackboard.get_filled()["refs"] == [ENTRY_2, ENTRY_3]


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
        # A stocked array stops after the patience run of empty windows plus the
        # two boundary probes a dry sweep spends - never the whole document.
        assert provider.calls == _CONTINUATION_STOP_AFTER_EMPTY + 2


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


class TestSingleWindowResample:
    @pytest.mark.asyncio
    async def test_empty_array_after_one_window_split_reprobed(self):
        # A one-window sweep has no neighbour to expose an under-emitting draw;
        # an array still empty afterwards is re-asked in two halves.
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

        provider = ScriptedProvider(["refs = []", batch("front", 5), batch("back", 5)])
        await _sweep_array_windows(["part-a\n\npart-b"], [REFS], leaf, provider, state, extracted)
        assert provider.calls == 3  # the window + two half re-probes
        assert len(extracted["refs"]) == 10

    @pytest.mark.asyncio
    async def test_stocked_array_after_one_window_not_reprobed(self):
        from nfield.pipeline.s4_extract import _sweep_array_windows

        state = _state()
        state.segments = [_big_segment()]
        leaf = _leaf()
        extracted: dict = {"refs": [ENTRY_1]}
        provider = ScriptedProvider([f'refs = ["{ENTRY_2}"]'])
        await _sweep_array_windows(["part-a\n\npart-b"], [REFS], leaf, provider, state, extracted)
        assert provider.calls == 1  # items aboard, no re-probe


class TestDocumentOrder:
    def test_items_sorted_by_document_position(self):
        from nfield.pipeline.s4_extract import _document_order

        state = _state()
        text = f"Intro.\n[1] {ENTRY_1}\n[2] {ENTRY_2}\n[3] {ENTRY_3}\n[4] {ENTRY_4}"
        state.segments = [
            Segment(text=text, start=0, end=len(text), segment_type="unstructured", segment_id=0)
        ]
        shuffled = [ENTRY_3, ENTRY_1, ENTRY_4, ENTRY_2]
        assert _document_order(shuffled, state) == [ENTRY_1, ENTRY_2, ENTRY_3, ENTRY_4]

    def test_unlocatable_items_keep_relative_order_at_end(self):
        from nfield.pipeline.s4_extract import _document_order

        state = _state()
        text = f"[1] {ENTRY_1}\n[2] {ENTRY_2}"
        state.segments = [
            Segment(text=text, start=0, end=len(text), segment_type="unstructured", segment_id=0)
        ]
        out = _document_order([ENTRY_2, "ghost entry beta", ENTRY_1, "ghost entry alpha"], state)
        assert out == [ENTRY_1, ENTRY_2, "ghost entry beta", "ghost entry alpha"]

    @pytest.mark.asyncio
    async def test_sweep_output_is_document_ordered(self):
        # High-relevance visit order must not leak into the merged list.
        from nfield.pipeline.s4_extract import _sweep_array_windows

        state = _state()
        big = "reference body text " * 3_000 + f"\n[1] {ENTRY_1}\n[2] {ENTRY_2}"
        state.segments = [
            Segment(text=big, start=0, end=len(big), segment_type="unstructured", segment_id=0)
        ]
        leaf = _leaf()
        extracted: dict = {"refs": []}
        provider = ScriptedProvider(
            [f'refs = ["{ENTRY_2}"]', f'refs = ["{ENTRY_1}"]', "refs = []"]
        )
        await _sweep_array_windows(["w-late", "w-early"], [REFS], leaf, provider, state, extracted)
        assert extracted["refs"] == [ENTRY_1, ENTRY_2]  # document order, not visit order


class TestPlaceholderItems:
    def test_item_restating_field_key_detected(self):
        from nfield.pipeline.s4_extract import _restates_path_key

        assert _restates_path_key("THE LENDERS FROM TIME TO TIME PARTY HERETO", "parties.lenders")
        assert _restates_path_key("TRANCHE A LENDER (as defined herein)", "parties.lenders")
        assert not _restates_path_key("ORION ENERGY CREDIT FUND II, L.P.", "parties.lenders")
        assert not _restates_path_key(ENTRY_1, "refs")

    @pytest.mark.asyncio
    async def test_placeholder_only_array_treated_empty_and_reswept(self):
        # One placeholder item must not suppress the whole-document sweep.
        from nfield.pipeline.s4_extract import _extend_arrays_over_windows

        state = _state()
        state.segments = [_big_segment()]
        leaf = _leaf()
        leaf.fields = [Field("refs", "array", {"items": {"type": "string"}}, "", {})]
        leaf.excerpt_segment_ids = {0}
        extracted: dict = {"refs": ["the refs listed hereto"]}
        provider = ScriptedProvider([f'refs = ["{ENTRY_1}", "{ENTRY_2}", "{ENTRY_3}"]'])
        await _extend_arrays_over_windows(leaf, provider, state, extracted)
        assert "the refs listed hereto" not in extracted["refs"]
        assert ENTRY_1 in extracted["refs"]


class TestScarceRegion:
    @pytest.mark.asyncio
    async def test_scarce_array_sweeps_whole_document(self):
        # Two items of a many-entry list say nothing about the rest; the sweep
        # must cover the whole document, not just the uncovered remainder.
        from nfield.pipeline.s4_extract import _extend_arrays_over_windows

        state = _state()
        state.segments = [_big_segment()]
        leaf = _leaf()
        # Excerpt already covers everything: a remainder-only region would be empty.
        leaf.excerpt_segment_ids = {0}
        extracted: dict = {"refs": [ENTRY_1, ENTRY_2]}
        provider = ScriptedProvider([f'refs = ["{ENTRY_3}", "{ENTRY_4}"]'])
        await _extend_arrays_over_windows(leaf, provider, state, extracted)
        assert provider.calls >= 1  # swept despite full excerpt coverage
        assert ENTRY_3 in extracted["refs"]


class TestBoundaryProbe:
    @pytest.mark.asyncio
    async def test_empty_array_probes_last_window_after_patience(self):
        # Items that live only on end pages (signature blocks) are missed by
        # relevance order; the outermost unvisited windows are probed before
        # the sweep gives up on a fully-empty array.
        from nfield.pipeline.s4_extract import _CONTINUATION_STOP_WHILE_EMPTY, _sweep_array_windows

        state = _state()
        state.segments = [_big_segment()]
        leaf = _leaf()
        extracted: dict = {"refs": []}
        windows = [f"body {i}" for i in range(12)] + ["signature pages"]

        class _EndOnly(ScriptedProvider):
            async def complete(self, messages, *, max_tokens):
                self.calls += 1
                if "signature pages" in messages[-1]["content"]:
                    return f'refs = ["{ENTRY_1}", "{ENTRY_2}", "{ENTRY_3}", "{ENTRY_4}"]'
                return "refs = []"

        # Relevance order keeps the last window out of the patience run.
        order = list(range(12))
        provider = _EndOnly([])
        await _sweep_array_windows(
            windows, [REFS], leaf, provider, state, extracted, visit_order=order
        )
        # patience + last window + its neighbour + the first unvisited window
        assert provider.calls == _CONTINUATION_STOP_WHILE_EMPTY + 3
        assert extracted["refs"] == [ENTRY_1, ENTRY_2, ENTRY_3, ENTRY_4]

    @pytest.mark.asyncio
    async def test_stocked_array_with_dry_sweep_still_probes_ends(self):
        # An array holding items is no proof of completeness: when the sweep
        # itself yields nothing new, the ends must still be verified.
        from nfield.pipeline.s4_extract import _sweep_array_windows

        state = _state()
        state.segments = [_big_segment()]
        leaf = _leaf()
        extracted: dict = {"refs": [ENTRY_1, ENTRY_2, ENTRY_3]}
        windows = [f"body {i}" for i in range(10)] + ["signature pages"]

        class _EndOnly(ScriptedProvider):
            async def complete(self, messages, *, max_tokens):
                self.calls += 1
                if "signature pages" in messages[-1]["content"]:
                    return f'refs = ["{ENTRY_4}"]'
                return "refs = []"

        provider = _EndOnly([])
        await _sweep_array_windows(
            windows, [REFS], leaf, provider, state, extracted, visit_order=list(range(10))
        )
        assert ENTRY_4 in extracted["refs"]  # end window reached despite stocked array

    @pytest.mark.asyncio
    async def test_no_boundary_probe_when_items_found(self):
        from nfield.pipeline.s4_extract import _sweep_array_windows

        state = _state()
        state.segments = [_big_segment()]
        leaf = _leaf()
        extracted: dict = {"refs": []}
        provider = ScriptedProvider([f'refs = ["{ENTRY_1}", "{ENTRY_2}", "{ENTRY_3}"]'])
        await _sweep_array_windows(["w0", "w1"], [REFS], leaf, provider, state, extracted)
        # First window yields; the sweep never needs the boundary pass.
        assert extracted["refs"] == [ENTRY_1, ENTRY_2, ENTRY_3]


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


class TestFocusedReask:
    @pytest.mark.asyncio
    async def test_array_empty_beside_yielding_sibling_reasked_alone(self):
        # A window answers all arrays at once; a field left empty while a sibling
        # yielded is re-asked alone so the selection pressure is gone.
        from nfield.pipeline.s4_extract import _extend_arrays_over_windows

        other = Field("margins", "array", {"items": {"type": "string"}}, "", {})
        state = _state()
        state.segments = [_big_segment()]
        leaf = _leaf()
        leaf.fields = [REFS, other]
        leaf.excerpt_segment_ids = {0}
        extracted: dict = {"refs": [], "margins": []}

        class _Selective(ScriptedProvider):
            async def complete(self, messages, *, max_tokens):
                self.calls += 1
                prompt = messages[-1]["content"]
                if "margins" in prompt and "refs" not in prompt:
                    return f'margins = ["{ENTRY_4}"]'  # focused ask succeeds
                return f'refs = ["{ENTRY_1}", "{ENTRY_2}", "{ENTRY_3}"]\nmargins = []'

        provider = _Selective([])
        await _extend_arrays_over_windows(leaf, provider, state, extracted)
        assert extracted["margins"] == [ENTRY_4]
        assert ENTRY_1 in extracted["refs"]

    @pytest.mark.asyncio
    async def test_all_arrays_empty_not_reasked(self):
        # When every array stayed empty the document lacks them; no focused call.
        from nfield.pipeline.s4_extract import _extend_arrays_over_windows

        other = Field("margins", "array", {"items": {"type": "string"}}, "", {})
        state = _state()
        state.segments = [_big_segment()]
        leaf = _leaf()
        leaf.fields = [REFS, other]
        leaf.excerpt_segment_ids = {0}
        extracted: dict = {"refs": [], "margins": []}
        seen: list[str] = []

        class _Recorder(ScriptedProvider):
            async def complete(self, messages, *, max_tokens):
                seen.append(messages[-1]["content"])
                return await super().complete(messages, max_tokens=max_tokens)

        provider = _Recorder(["refs = []\nmargins = []"])
        await _extend_arrays_over_windows(leaf, provider, state, extracted)
        assert seen, "the sweep must run"
        # Every call asked for both fields together - no single-field re-ask ran.
        assert all("refs" in p and "margins" in p for p in seen)


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


class TestMergeObjectDedup:
    """Near-duplicate object rows collapse; distinct rows are kept."""

    def test_row_missing_a_field_is_merged_and_fuller_kept(self):
        merged = [{"title": "Deep Learning", "year": 2020}]
        added = _merge_window_items(merged, [{"title": "Deep Learning", "year": 2020, "doi": "x"}])
        assert added == 0
        assert merged == [{"title": "Deep Learning", "year": 2020, "doi": "x"}]

    def test_distinct_rows_sharing_a_field_are_kept(self):
        merged = [{"title": "Deep Learning", "venue": "NeurIPS"}]
        added = _merge_window_items(
            merged, [{"title": "Reinforcement Learning", "venue": "NeurIPS"}]
        )
        assert added == 1
        assert len(merged) == 2

    def test_exact_duplicate_not_added(self):
        merged = [{"a": "x"}]
        assert _merge_window_items(merged, [{"a": "x"}]) == 0


class TestDocumentPreamble:
    """Continuation windows carry the document-head context."""

    def test_head_segments_within_cap(self):
        from nfield.pipeline.s4_extract import _document_preamble

        state = _state()
        segs = state.segments
        cap = len(segs[0].text) + len(segs[1].text) + 10
        preamble = _document_preamble(state, cap)
        assert preamble == f"{segs[0].text}\n\n{segs[1].text}"

    def test_oversized_first_segment_cut_at_cap(self):
        from nfield.pipeline.s4_extract import _document_preamble

        state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=1024, C_usable=4096.0)
        state.segments = [_big_segment()]
        preamble = _document_preamble(state, 150)
        assert preamble == _big_segment().text[:150]

    def test_cap_rides_in_input_slack_never_content(self):
        from nfield.pipeline.s4_extract import _preamble_cap, _window_chars

        state = _state()
        leaf = _leaf()
        window_chars = _window_chars(leaf, state)
        cap = _preamble_cap(leaf, state, window_chars)
        # Output-bound window leaves input slack; the cap uses it, bounded by fraction.
        assert 0 < cap <= int(window_chars * 0.15)
        # An input-bound window has no slack: the preamble must vanish, not displace.
        tight = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=1024, C_usable=100.0)
        tight.segments = _segments()
        assert _preamble_cap(_leaf(), tight, _window_chars(_leaf(), tight)) == 0

    def test_no_segments_or_budget_yields_empty(self):
        from nfield.pipeline.s4_extract import _document_preamble

        empty = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=1024, C_usable=4096.0)
        empty.segments = []
        assert _document_preamble(empty, 10_000) == ""
        assert _document_preamble(_state(), 0) == ""

    @pytest.mark.asyncio
    async def test_sweep_prepends_preamble_to_windows(self):
        from nfield.pipeline.s4_extract import _sweep_array_windows

        state = _state()
        leaf = _leaf()

        class CapturingProvider(ContinuationProvider):
            def __init__(self):
                super().__init__("refs = []")
                self.prompts: list[str] = []

            async def complete(self, messages, *, max_tokens):
                self.prompts.append("\n".join(m.get("content", "") for m in messages))
                return await super().complete(messages, max_tokens=max_tokens)

        provider = CapturingProvider()
        extracted: dict[str, object] = {"refs": []}
        head = "ACME CORP QUARTERLY REPORT for the period ended March 31"
        await _sweep_array_windows(
            ["mid-document window text"], [REFS], leaf, provider, state, extracted, preamble=head
        )
        assert provider.prompts and head in provider.prompts[0]

    @pytest.mark.asyncio
    async def test_window_already_starting_with_head_not_doubled(self):
        from nfield.pipeline.s4_extract import _sweep_array_windows

        state = _state()
        leaf = _leaf()

        class CapturingProvider(ContinuationProvider):
            def __init__(self):
                super().__init__("refs = []")
                self.prompts: list[str] = []

            async def complete(self, messages, *, max_tokens):
                self.prompts.append("\n".join(m.get("content", "") for m in messages))
                return await super().complete(messages, max_tokens=max_tokens)

        provider = CapturingProvider()
        extracted: dict[str, object] = {"refs": []}
        head = "ACME CORP QUARTERLY REPORT"
        window = f"{head}\n\nfirst window body"
        await _sweep_array_windows(
            [window], [REFS], leaf, provider, state, extracted, preamble=head
        )
        assert provider.prompts[0].count(head) == 1


def _dim_field(name: str) -> Field:
    items = {
        "type": "object",
        "properties": {
            "level": {"type": "string", "enum": ["total", "regional", "divisional"]},
            "label": {"type": "string"},
            "amount": {"type": "number"},
        },
    }
    return Field(name, "array", {"items": items}, "", {"items": items})


class TestAxisStarvation:
    """A dimension array missing an axis value a sibling proved present is starved."""

    def test_starved_field_detected_with_proof_rows(self):
        from nfield.pipeline.s4_extract import _axis_starved_fields

        revenue, cost = _dim_field("revenue"), _dim_field("cost")
        extracted = {
            "revenue": [{"level": "total", "label": "all", "amount": 100}],
            "cost": [
                {"level": "total", "label": "all", "amount": 40},
                {"level": "regional", "label": "north", "amount": 25},
                {"level": "regional", "label": "south", "amount": 15},
            ],
        }
        starved = _axis_starved_fields([revenue, cost], extracted)
        assert len(starved) == 1
        f, axis, missing, proof = starved[0]
        assert f.path == "revenue"
        assert axis == "level"
        assert missing == {"regional"}
        assert len(proof) == 2

    def test_empty_array_not_reported(self):
        from nfield.pipeline.s4_extract import _axis_starved_fields

        revenue, cost = _dim_field("revenue"), _dim_field("cost")
        extracted = {
            "revenue": [],
            "cost": [{"level": "regional", "label": "north", "amount": 25}],
        }
        assert _axis_starved_fields([revenue, cost], extracted) == []

    def test_equal_coverage_not_reported(self):
        from nfield.pipeline.s4_extract import _axis_starved_fields

        revenue, cost = _dim_field("revenue"), _dim_field("cost")
        row = {"level": "total", "label": "all", "amount": 1}
        extracted = {"revenue": [row], "cost": [dict(row)]}
        assert _axis_starved_fields([revenue, cost], extracted) == []

    def test_different_axis_definitions_not_compared(self):
        from nfield.pipeline.s4_extract import _axis_starved_fields

        revenue = _dim_field("revenue")
        other_items = {
            "type": "object",
            "properties": {
                "level": {"type": "string", "enum": ["high", "low"]},
                "amount": {"type": "number"},
            },
        }
        other = Field("other", "array", {"items": other_items}, "", {"items": other_items})
        extracted = {
            "revenue": [{"level": "total", "label": "all", "amount": 1}],
            "other": [{"level": "high", "amount": 2}],
        }
        assert _axis_starved_fields([revenue, other], extracted) == []


class TestWindowsHoldingRows:
    """A sibling row places in the window whose text carries its leaves."""

    def test_window_with_two_row_leaves_matches(self):
        from nfield.pipeline.s4_extract import _windows_holding_rows

        rows = [{"level": "regional", "label": "north region", "amount": 2666}]
        windows = [
            "unrelated prose about the business overall",
            "breakdown table: north region 2666 south region 1500",
        ]
        assert _windows_holding_rows(windows, rows) == [1]

    def test_no_match_returns_empty(self):
        from nfield.pipeline.s4_extract import _windows_holding_rows

        rows = [{"level": "regional", "label": "north region", "amount": 2666}]
        assert _windows_holding_rows(["nothing relevant here"], rows) == []


class TestAxisDeconflictReask:
    """The starved array is re-asked alone and its missing rows merged."""

    @pytest.mark.asyncio
    async def test_starved_array_filled_from_sibling_window(self):
        from nfield.pipeline.s4_extract import _extend_arrays_over_windows

        revenue, cost = _dim_field("revenue"), _dim_field("cost")
        body = "breakdown table: north region revenue 9090 north region cost 2555"
        state = _state()
        state.segments = [
            Segment(text=body, start=0, end=len(body), segment_type="unstructured", segment_id=0)
        ]
        leaf = CapacityLeaf(
            fields=[revenue, cost],
            groups=[],
            document_excerpt=body,
            overhead=10.0,
            safe_output=1024,
            leaf_id=0,
        )
        extracted = {
            "revenue": [{"level": "total", "label": "all", "amount": 100}],
            "cost": [
                {"level": "total", "label": "all", "amount": 40},
                {"level": "regional", "label": "north region", "amount": 2555},
            ],
        }

        class DeconflictProvider(ContinuationProvider):
            def __init__(self):
                super().__init__("")
                self.solo_revenue_calls = 0

            async def complete(self, messages, *, max_tokens):
                self.calls += 1
                text = "\n".join(m.get("content", "") for m in messages)
                if "revenue (array)" in text and "cost (array)" not in text:
                    self.solo_revenue_calls += 1
                    return (
                        'revenue = [{"level": "regional", "label": "north region", "amount": 90}]'
                    )
                return "revenue = []\ncost = []"

        provider = DeconflictProvider()
        await _extend_arrays_over_windows(leaf, provider, state, extracted)
        assert provider.solo_revenue_calls >= 1
        levels = {r["level"] for r in extracted["revenue"]}
        assert levels == {"total", "regional"}


class TestRescueCapBonus:
    """A rescue pass runs even after the general sweep exhausted the window budget."""

    @pytest.mark.asyncio
    async def test_rescue_runs_at_exhausted_cap(self):
        from nfield.pipeline.s4_extract import (
            _max_continuation_windows_per_doc,
            _sweep_array_windows,
        )

        state = _state()
        leaf = _leaf()
        state.continuation_windows_used = _max_continuation_windows_per_doc(leaf, state)
        provider = ContinuationProvider(f'refs = ["{ENTRY_1}"]')
        extracted: dict[str, object] = {"refs": []}
        await _sweep_array_windows(
            ["window"], [REFS], leaf, provider, state, extracted, cap_bonus=1
        )
        assert provider.calls == 1
        assert extracted["refs"] == [ENTRY_1]

    @pytest.mark.asyncio
    async def test_no_bonus_stays_capped(self):
        from nfield.pipeline.s4_extract import (
            _max_continuation_windows_per_doc,
            _sweep_array_windows,
        )

        state = _state()
        leaf = _leaf()
        state.continuation_windows_used = _max_continuation_windows_per_doc(leaf, state)
        provider = ContinuationProvider(f'refs = ["{ENTRY_1}"]')
        extracted: dict[str, object] = {"refs": []}
        await _sweep_array_windows(["window"], [REFS], leaf, provider, state, extracted)
        assert provider.calls == 0
