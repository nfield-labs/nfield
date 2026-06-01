"""FormatShield — N-field structured extraction from documents with LLMs.

Extract hundreds of structured fields from any document without the format tax.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from ._version import __version__

if TYPE_CHECKING:
    from .config import DomainConfig, ExtractionConfig, register_domain
    from .exceptions import (
        AssemblyError,
        ExtractionError,
        FormatShieldError,
        ProviderError,
        SchemaError,
        ValidationError,
    )
    from .types import ExtractionResult, ExtractionStatus, FieldResult, Metadata

__all__ = [
    "AssemblyError",
    "DomainConfig",
    "ExtractionConfig",
    "ExtractionError",
    "ExtractionResult",
    "ExtractionStatus",
    "FieldResult",
    "FormatShieldError",
    "Metadata",
    "ProviderError",
    "SchemaError",
    "ValidationError",
    "__version__",
    "register_domain",
]

_dynamic_imports: dict[str, str] = {
    "ExtractionConfig": ".config",
    "DomainConfig": ".config",
    "register_domain": ".config",
    "ExtractionResult": ".types",
    "FieldResult": ".types",
    "Metadata": ".types",
    "ExtractionStatus": ".types",
    "FormatShieldError": ".exceptions",
    "SchemaError": ".exceptions",
    "ProviderError": ".exceptions",
    "ExtractionError": ".exceptions",
    "ValidationError": ".exceptions",
    "AssemblyError": ".exceptions",
}


def __getattr__(name: str) -> object:
    if name in _dynamic_imports:
        module = importlib.import_module(_dynamic_imports[name], package=__name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
