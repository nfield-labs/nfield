"""FormatShield orchestrator — sync and async engines plus the n* entry points.

``_async.py`` holds the real pipeline runner (:class:`AsyncFormatShield`);
``_sync.py`` wraps it for blocking callers (:class:`FormatShield`). Both
``nfield`` and ``nfield_async`` are one-shot conveniences that delegate here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from formatshield.engine._async import AsyncFormatShield, nfield_async
    from formatshield.engine._sync import FormatShield, nfield

__all__ = [
    "AsyncFormatShield",
    "FormatShield",
    "nfield",
    "nfield_async",
]

_dynamic_imports: dict[str, str] = {
    "AsyncFormatShield": "._async",
    "nfield_async": "._async",
    "FormatShield": "._sync",
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
