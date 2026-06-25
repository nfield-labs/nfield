"""NField CLI package.

The Typer application lives in :mod:`nfield.cli._app`. Importing it
requires the optional ``[cli]`` extra (``pip install "nfield[cli]"``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nfield.cli._app import app, main

__all__ = ["app", "main"]


def __getattr__(name: str) -> object:
    if name in ("app", "main"):
        from nfield.cli import _app

        return getattr(_app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
