"""Dataset registry tests — names, gold flags, and lookup errors."""

from __future__ import annotations

import pytest

from benchmark import datasets


def test_available_is_sorted_and_nonempty():
    names = datasets.available()
    assert names == tuple(sorted(names))
    assert "clinicaltrial" in names


def test_gold_fixtures_advertise_a_key():
    assert datasets.get("clinicaltrial").has_gold is True
    assert datasets.get("war_and_peace").has_gold is False


def test_medical_crf_was_dropped():
    assert "medical_crf" not in datasets.available()


def test_registered_schema_files_exist_on_disk():
    for name in datasets.available():
        assert (datasets.get(name).directory / "schema.json").exists(), name


def test_gold_fixtures_load_and_score_offline():
    # A scorable fixture must round-trip: schema + document + gold all present.
    loaded = datasets.get("clinicaltrial").load()
    assert loaded.gold and len(loaded.gold) > 0
    assert loaded.document and loaded.schema.get("type") == "object"


def test_unknown_dataset_raises_with_helpful_message():
    with pytest.raises(KeyError, match="unknown dataset"):
        datasets.get("does_not_exist")
