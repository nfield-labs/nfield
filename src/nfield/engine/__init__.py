"""nfield orchestrator - sync and async engines plus the n* entry points.

``_async.py`` holds the real pipeline runner (:class:`AsyncNField`);
``_sync.py`` wraps it for blocking callers (:class:`NField`). Both
``nfield`` and ``nfield_async`` are one-shot conveniences that delegate here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nfield.engine._async import AsyncNField, nfield_async
    from nfield.engine._sync import NField, nfield

__all__ = [
    "AsyncNField",
    "NField",
    "nfield",
    "nfield_async",
]

_dynamic_imports: dict[str, str] = {
    "AsyncNField": "._async",
    "nfield_async": "._async",
    "NField": "._sync",
    "nfield": "._sync",
}


def __getattr__(name: str) -> object:
    """Lazily resolve engine classes and entry points on first access."""
    module_suffix = _dynamic_imports.get(name)
    if module_suffix is not None:
        import importlib

        module = importlib.import_module(module_suffix, package=__name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
