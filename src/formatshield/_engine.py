"""Backward-compatible re-export shim for the engine package.

The engine implementation lives in :mod:`formatshield.engine`. This module
keeps the historical ``formatshield._engine`` import path working so external
code that referenced it does not break. New code should import from
:mod:`formatshield` (the public API) or :mod:`formatshield.engine`.
"""

from __future__ import annotations

from formatshield.engine import (
    AsyncFormatShield,
    FormatShield,
    nfield,
    nfield_async,
)

__all__ = [
    "AsyncFormatShield",
    "FormatShield",
    "nfield",
    "nfield_async",
]
