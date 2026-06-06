"""FormatShield CLI package.

The Typer application lives in :mod:`formatshield.cli._app`. Importing it
requires the optional ``[cli]`` extra (``pip install "formatshield[cli]"``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from formatshield.cli._app import app, main

__all__ = ["app", "main"]


def __getattr__(name: str) -> object:
    if name in ("app", "main"):
        from formatshield.cli import _app

        return getattr(_app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
