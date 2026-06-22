"""Dataset registry — fixture name to (schema, document, gold answer key).

Datasets are self-contained under ``benchmark/datasets/real/<name>/``, each a
directory of ``schema.json``, ``document.txt``, and (when scorable) ``gold.json``.
The small gold documents are committed for reproducibility; the one large
coverage-only document (War & Peace) is fetched on demand via ``datasets/gen``.

The model budget is NOT a property of the dataset — it is a run-level choice
(:mod:`benchmark.budget`), applied uniformly to every method, so the same fixture
can be swept under different budgets.

A fixture with a gold key can be scored for Value Accuracy; one without (a raw
document with no answer key) can only be measured for coverage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["Dataset", "LoadedDataset", "available", "get"]

_REAL = Path(__file__).resolve().parent / "datasets" / "real"
_SCHEMA_FILE = "schema.json"
_DOCUMENT_FILE = "document.txt"
_GOLD_FILE = "gold.json"

# Domain instructions are general, professional guidance a real caller would
# write — never gold answers. The SAME string is given to every method (nfield
# threads it to each leaf; the baselines prepend it to their prompt), so it is a
# fair, uniform input, not a per-method tuning lever (fairness rule, design §7).
_FAITHFULNESS = (
    "Extract each field's value exactly as written in the document — keep all "
    "units, dates, identifiers, and parenthetical qualifiers (e.g. '(2023 est.)'); "
    "never summarize, normalize, or drop trailing detail. Preserve the order of "
    "any list items as they appear in the source."
)
_INSTRUCTIONS: dict[str, str] = {
    "clinicaltrial": f"The document is a ClinicalTrials.gov study record. {_FAITHFULNESS}",
    "factbook_us": f"The document is CIA World Factbook country profile data. {_FAITHFULNESS}",
    "factbook_multi": f"The document is CIA World Factbook country profile data. {_FAITHFULNESS}",
    # W&P is narrative prose, not tabular data, so faithfulness here means grounding
    # in the text rather than preserving units/qualifiers.
    "war_and_peace": (
        "The document is the novel War and Peace by Leo Tolstoy. Extract each field "
        "from what the narrative states about its book, characters, locations, and "
        "events; do not infer beyond the text, and leave a field null if the novel "
        "does not state it."
    ),
    "chemical_element": f"The document is a periodic-table element profile. {_FAITHFULNESS}",
    "smartphone_spec": f"The document is a smartphone specification sheet. {_FAITHFULNESS}",
}


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    """A dataset with its files read into memory.

    Args:
        name: Registry key of the source dataset.
        schema: The target JSON Schema.
        document: The source text to extract from.
        gold: Flat gold answer key, or ``None`` when the fixture has no key
            (coverage-only).
        instructions: Domain guidance given identically to every method.
    """

    name: str
    schema: dict[str, Any]
    document: str
    gold: dict[str, Any] | None
    instructions: str = ""


@dataclass(frozen=True, slots=True)
class Dataset:
    """A registered fixture: where its directory lives and how to run it.

    The fixture directory holds ``schema.json``, ``document.txt``, and an
    optional ``gold.json`` under ``benchmark/datasets/real/<name>/``.

    Args:
        name: Registry key and directory name.
        instructions: Domain guidance given identically to every method.
    """

    name: str
    instructions: str = ""

    @property
    def directory(self) -> Path:
        """Path to this dataset's directory."""
        return _REAL / self.name

    @property
    def has_gold(self) -> bool:
        """Return ``True`` when the fixture can be scored for Value Accuracy."""
        return (self.directory / _GOLD_FILE).exists()

    def load(self) -> LoadedDataset:
        """Read this dataset's files into memory.

        Returns:
            A :class:`LoadedDataset` with schema, document, and gold populated.

        Raises:
            FileNotFoundError: If the schema or document file is missing.
        """
        schema = json.loads((self.directory / _SCHEMA_FILE).read_text(encoding="utf-8"))
        document = (self.directory / _DOCUMENT_FILE).read_text(encoding="utf-8")
        gold_path = self.directory / _GOLD_FILE
        gold = json.loads(gold_path.read_text(encoding="utf-8")) if gold_path.exists() else None
        return LoadedDataset(
            name=self.name,
            schema=schema,
            document=document,
            gold=gold,
            instructions=self.instructions,
        )


_REGISTRY: dict[str, Dataset] = {
    name: Dataset(name, instructions=_INSTRUCTIONS.get(name, ""))
    for name in (
        "clinicaltrial",
        "factbook_us",
        "factbook_multi",
        "war_and_peace",
        "chemical_element",
        "smartphone_spec",
    )
}


def available() -> tuple[str, ...]:
    """Return the registered fixture names, sorted."""
    return tuple(sorted(_REGISTRY))


def get(name: str) -> Dataset:
    """Look up a registered dataset by name.

    Args:
        name: Registry key, one of :func:`available`.

    Returns:
        The matching :class:`Dataset`.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; available: {', '.join(available())}") from None
