"""Tests for the FinTagging FinNI metric, fact normalization, and slice loader."""

from __future__ import annotations

import pytest

from benchmark.benchmarks.fintagging import (
    FINNI_TYPES,
    build_wide_documents,
    finni_f1,
    load_finni_slice,
    normalize_fact,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("$ 2,560.0", "2560.0"),
        ("( 330.8 )", "330.8"),  # parenthetical negative loses its wrapper and sign
        ("4.27", "4.27"),
        ("15 %", "15"),
        ("-12", "12"),
    ],
)
def test_normalize_fact_strips_formatting(raw: str, expected: str) -> None:
    assert normalize_fact(raw) == expected


def _facts(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"Fact": f, "Type": t} for f, t in pairs]


def test_f1_perfect_match() -> None:
    gold = _facts(("100", "monetaryItemType"), ("4.27", "perShareItemType"))
    p, r, f1 = finni_f1(gold, gold)
    assert (p, r, f1) == (1.0, 1.0, 1.0)


def test_f1_requires_type_to_match() -> None:
    gold = _facts(("100", "monetaryItemType"))
    pred = _facts(("100", "perShareItemType"))  # right value, wrong type -> no hit
    _, _, f1 = finni_f1(pred, gold)
    assert f1 == 0.0


def test_f1_recall_drops_when_facts_are_omitted() -> None:
    # The single-call failure mode: half the facts truncated away.
    gold = _facts(("1", "monetaryItemType"), ("2", "monetaryItemType"))
    pred = _facts(("1", "monetaryItemType"))
    p, r, _ = finni_f1(pred, gold)
    assert p == 1.0
    assert r == 0.5


def test_f1_is_multiset_on_repeated_facts() -> None:
    # A dash repeated three times in gold needs three predicted matches, not one.
    gold = _facts(("0", "monetaryItemType"), ("0", "monetaryItemType"), ("0", "monetaryItemType"))
    pred = _facts(("0", "monetaryItemType"))
    _, r, _ = finni_f1(pred, gold)
    assert r == pytest.approx(1 / 3)


def test_f1_matches_formatted_variants() -> None:
    gold = _facts(("2560.0", "monetaryItemType"))
    pred = _facts(("$ 2,560.0", "monetaryItemType"))  # normalized before matching
    _, _, f1 = finni_f1(pred, gold)
    assert f1 == 1.0


def test_finni_types_are_the_five_xbrl_item_types() -> None:
    assert len(FINNI_TYPES) == 5
    assert all(t.endswith("ItemType") for t in FINNI_TYPES)


def test_slice_loads_and_is_wide() -> None:
    rows = load_finni_slice()
    assert rows
    # The committed slice is the wide-table regime: every context has many facts.
    for row in rows:
        assert row["n_facts"] >= 100
        assert "answer" in row and "query" in row and "context" in row


def test_build_wide_documents_grows_gold_with_size() -> None:
    rows = load_finni_slice()
    docs = build_wide_documents(rows, sizes=(1, 3))
    assert [d.n_tables for d in docs] == [1, 3]
    # Concatenating more tables strictly grows the fact count and the document text.
    assert len(docs[1].gold) > len(docs[0].gold)
    assert len(docs[1].text) > len(docs[0].text)
    # Both methods get the paper's instruction block; the baseline query embeds the doc.
    assert docs[0].instructions and "Input:" in docs[0].query


def test_build_wide_documents_clamps_to_slice_length() -> None:
    rows = load_finni_slice()
    (doc,) = build_wide_documents(rows, sizes=(9999,))
    assert doc.n_tables == len(rows)
