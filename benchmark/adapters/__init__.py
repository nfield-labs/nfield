"""Adapter layer - one uniform interface over every method under test.

Each method (nfield and every baseline) is wrapped in an :class:`Adapter` whose
``run`` returns the same :class:`AdapterOutput`, so the runner and scorer treat
all methods identically. Adapters never raise for a model/API failure: a 400,
timeout, or refusal is captured in :attr:`AdapterOutput.error` and scored as a
miss, keeping the failure in the denominator (honest-claims charter, rule 4).
"""

from __future__ import annotations

from ._base import Adapter, AdapterOutput

__all__ = ["Adapter", "AdapterOutput"]
