"""FormatShield — N-field structured extraction from documents with LLMs.

Extract hundreds of structured fields from any document without the format tax.

Quickstart:
    >>> from formatshield import nfield
    >>> # result = nfield(document, MySchema, "groq/llama-3.1-8b")
    >>> # result.data, result.metadata, result.status

Every public name is imported lazily, so ``import formatshield`` stays fast and
never fails because an optional provider SDK (e.g. groq) is not installed.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from ._version import __version__

if TYPE_CHECKING:
    from .config import DomainConfig, ExtractionConfig, register_domain
    from .engine import AsyncFormatShield, FormatShield, nfield, nfield_async
    from .exceptions import (
        AssemblyError,
        ExtractionError,
        FormatShieldError,
        ProviderError,
        SchemaError,
        ValidationError,
    )
    from .providers import from_model
    from .types import ExtractionResult, ExtractionStatus, FieldResult, Metadata

__all__ = [
    "AssemblyError",
    "AsyncFormatShield",
    "DomainConfig",
    "ExtractionConfig",
    "ExtractionError",
    "ExtractionResult",
    "ExtractionStatus",
    "FieldResult",
    "FormatShield",
    "FormatShieldError",
    "Metadata",
    "ProviderError",
    "SchemaError",
    "ValidationError",
    "__version__",
    "from_model",
    "nfield",
    "nfield_async",
    "register_domain",
]

_dynamic_imports: dict[str, str] = {
    # Entry-point functions and engine classes
    "nfield": ".engine",
    "nfield_async": ".engine",
    "FormatShield": ".engine",
    "AsyncFormatShield": ".engine",
    # Provider factory
    "from_model": ".providers",
    # Config
    "ExtractionConfig": ".config",
    "DomainConfig": ".config",
    "register_domain": ".config",
    # Types
    "ExtractionResult": ".types",
    "FieldResult": ".types",
    "Metadata": ".types",
    "ExtractionStatus": ".types",
    # Exceptions
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
