"""Global test fixtures for NField test suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SCHEMAS_DIR = FIXTURES_DIR / "schemas"
DOCUMENTS_DIR = FIXTURES_DIR / "documents"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Return the path to the test fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def schemas_dir() -> Path:
    """Return the path to the test schema fixtures directory."""
    return SCHEMAS_DIR


@pytest.fixture(scope="session")
def simple_flat_schema() -> dict:  # type: ignore[type-arg]
    """Load the simple flat schema fixture."""
    return json.loads((SCHEMAS_DIR / "simple_flat.json").read_text())  # type: ignore[return-value]


@pytest.fixture(scope="session")
def invoice_50fields_schema() -> dict:  # type: ignore[type-arg]
    """Load the 50-field invoice schema fixture."""
    return json.loads((SCHEMAS_DIR / "invoice_50fields.json").read_text())  # type: ignore[return-value]


@pytest.fixture(autouse=True)
def clean_nfield_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all NFIELD_* env vars before each test."""
    import os

    for key in list(os.environ):
        if key.startswith("NFIELD_"):
            monkeypatch.delenv(key)
