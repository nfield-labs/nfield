"""Tests for the truncation-driven window split in the Stage 4 array sweep."""

from __future__ import annotations

import pytest

from nfield.pipeline._state import PipelineState
from nfield.pipeline.s4_extract import _halve_window, _sweep_array_windows
from nfield.schema._types import CapacityLeaf, Field, Segment

REFS = Field("refs", "array", {"items": {"type": "string"}}, "", {})

ENTRY_1 = "Alpha and Beta. A study of segment routing in large systems. Journal A, 2021."
ENTRY_2 = "Gamma and Delta. Windowed extraction over long documents. Journal B, 2022."
ENTRY_3 = "Epsilon and Zeta. Continuation strategies for bounded outputs. Journal C, 2023."
ENTRY_4 = "Eta and Theta. Deduplicating merged item streams. Journal D, 2024."

TRUNCATED = f'refs = ["{ENTRY_1}", "Epsilon and Zeta. Contin'


def _big_segment() -> Segment:
    # Large enough that the document-sized continuation budget covers many windows.
    text = "reference body text " * 3_000
    return Segment(text=text, start=0, end=len(text), segment_type="unstructured", segment_id=0)


def _state() -> PipelineState:
    state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=1024, C_usable=4096.0)
    state.segments = [_big_segment()]
    return state


def _leaf() -> CapacityLeaf:
    return CapacityLeaf(
        fields=[REFS],
        groups=[],
        document_excerpt="",
        overhead=100.0,
        safe_output=1024,
        leaf_id=0,
    )


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


class ContentProvider:
    """Answers by which marker the user message carries; records output budgets."""

    context_window = 8192
    max_output_tokens = 1024
    model_name = "mock/model"

    def __init__(self, by_marker: dict[str, str], default: str):
        self.by_marker = by_marker
        self.default = default
        self.calls = 0
        self.max_tokens_seen: list[int] = []

    async def complete(self, messages, *, max_tokens):
        self.calls += 1
        self.max_tokens_seen.append(max_tokens)
        user = next(m["content"] for m in messages if m["role"] == "user")
        for marker, response in self.by_marker.items():
            if marker in user:
                return response
        return self.default


class TestHalveWindow:
    def test_prefers_segment_boundary(self):
        left, right = _halve_window("aaaa\n\nbbbbbbbbbb")
        assert left == "aaaa"
        assert right == "\n\nbbbbbbbbbb"

    def test_no_boundary_cuts_midpoint(self):
        left, right = _halve_window("abcdefgh")
        assert left + right == "abcdefgh"
        assert len(left) == 4


class TestTruncationSplit:
    @pytest.mark.asyncio
    async def test_truncated_window_reprobed_in_halves_with_full_output(self):
        # A window cut at the output limit re-asks its halves; the halves carry the
        # full output budget and their items merge into the array.
        state = _state()
        leaf = _leaf()
        extracted: dict = {"refs": []}

        full = "LEFT half text\n\nRIGHT half text"
        provider = ContentProvider(
            {
                full: TRUNCATED,
                "RIGHT": f'refs = ["{ENTRY_3}", "{ENTRY_4}"]',
                "LEFT": f'refs = ["{ENTRY_1}", "{ENTRY_2}"]',
            },
            default=TRUNCATED,
        )

        await _sweep_array_windows([full], [REFS], leaf, provider, state, extracted)
        assert provider.calls == 3  # full window + two halves
        assert provider.max_tokens_seen[1:] == [leaf.safe_output] * 2
        assert set(extracted["refs"]) == {ENTRY_1, ENTRY_2, ENTRY_3, ENTRY_4}

    @pytest.mark.asyncio
    async def test_always_truncated_bounded_by_split_depth(self):
        # A pathological doc whose every window truncates stops at the depth bound
        # instead of recursing forever: at most 1 + 2 + 4 + 8 calls per top window.
        state = _state()
        leaf = _leaf()
        extracted: dict = {"refs": []}

        provider = ScriptedProvider([TRUNCATED])
        await _sweep_array_windows(
            ["a\n\nb\n\nc\n\nd\n\ne\n\nf\n\ng\n\nh"], [REFS], leaf, provider, state, extracted
        )
        assert provider.calls <= 15

    @pytest.mark.asyncio
    async def test_halves_replace_parent_salvage(self):
        # The parent's partial list re-covers the same span as the halves; keeping it
        # would stack format variants of the same rows, so only the halves' items stay.
        state = _state()
        leaf = _leaf()
        extracted: dict = {"refs": []}

        full = "LEFT half text\n\nRIGHT half text"
        parent_variant = "Alpha and Beta. A study of segment routing (shortened variant)."
        provider = ContentProvider(
            {
                full: f'refs = ["{parent_variant}", "Epsilon and Zeta. Contin',
                "RIGHT": f'refs = ["{ENTRY_3}", "{ENTRY_4}"]',
                "LEFT": f'refs = ["{ENTRY_1}", "{ENTRY_2}"]',
            },
            default=TRUNCATED,
        )

        await _sweep_array_windows([full], [REFS], leaf, provider, state, extracted)
        assert parent_variant not in extracted["refs"]
        assert set(extracted["refs"]) == {ENTRY_1, ENTRY_2, ENTRY_3, ENTRY_4}

    @pytest.mark.asyncio
    async def test_parent_salvage_kept_when_halves_return_nothing(self):
        # Both half probes yield nothing (here: empty arrays); the parent's salvaged
        # prefix is better than nothing and is merged after all.
        state = _state()
        leaf = _leaf()
        extracted: dict = {"refs": []}

        full = "LEFT half text\n\nRIGHT half text"
        provider = ContentProvider(
            {
                full: f'refs = ["{ENTRY_1}", "Epsilon and Zeta. Contin',
                "RIGHT": "refs = []",
                "LEFT": "refs = []",
            },
            default="refs = []",
        )

        await _sweep_array_windows([full], [REFS], leaf, provider, state, extracted)
        assert ENTRY_1 in extracted["refs"]

    @pytest.mark.asyncio
    async def test_clean_window_never_split(self):
        state = _state()
        leaf = _leaf()
        extracted: dict = {"refs": []}

        provider = ScriptedProvider([f'refs = ["{ENTRY_1}", "{ENTRY_2}"]'])
        await _sweep_array_windows(["only window"], [REFS], leaf, provider, state, extracted)
        assert provider.calls == 1

    @pytest.mark.asyncio
    async def test_empty_truncated_response_not_split(self):
        # Truncation with zero salvaged items carries no evidence of density; the
        # window is left to the low-yield machinery rather than split-probed.
        state = _state()
        leaf = _leaf()
        extracted: dict = {"refs": []}

        provider = ScriptedProvider(['refs = ["Epsilon and Zeta. Contin'])
        await _sweep_array_windows(["x\n\ny"], [REFS], leaf, provider, state, extracted)
        # One top-window probe plus at most the still-empty split re-probe pair;
        # no truncation-driven recursion on top of it.
        assert provider.calls <= 3
