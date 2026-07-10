"""nfield - N-field structured extraction from documents with LLMs.

Extract hundreds of structured fields from any document without the format tax.

Quickstart:
    >>> from nfield import nfield
    >>> # result = nfield(document, MySchema, "groq/llama-3.1-8b")
    >>> # result.data, result.metadata, result.status

Every public name is imported lazily, so ``import nfield`` stays fast and
never fails because an optional provider SDK (e.g. groq) is not installed.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from ._version import __version__

if TYPE_CHECKING:
    from .config import ExtractionConfig
    from .engine import AsyncNField, NField, nfield, nfield_async
    from .exceptions import (
        AssemblyError,
        ExtractionError,
        NFieldError,
        ProviderError,
        SchemaError,
        ValidationError,
    )
    from .export import result_to_dataframe, results_to_csv, results_to_dataframe
    from .io import load_document, load_results, load_schema, save_results
    from .providers import DiskCache, MemoryCache, ResponseCache, from_model
    from .types import ExtractionResult, ExtractionStatus, FieldResult, Metadata
    from .viz import save_html

__all__ = [
    "AssemblyError",
    "AsyncNField",
    "DiskCache",
    "ExtractionConfig",
    "ExtractionError",
    "ExtractionResult",
    "ExtractionStatus",
    "FieldResult",
    "MemoryCache",
    "Metadata",
    "NField",
    "NFieldError",
    "ProviderError",
    "ResponseCache",
    "SchemaError",
    "ValidationError",
    "__version__",
    "from_model",
    "load_document",
    "load_results",
    "load_schema",
    "nfield",
    "nfield_async",
    "result_to_dataframe",
    "results_to_csv",
    "results_to_dataframe",
    "save_html",
    "save_results",
]

_dynamic_imports: dict[str, str] = {
    # Entry-point functions and engine classes
    "nfield": ".engine",
    "nfield_async": ".engine",
    "NField": ".engine",
    "AsyncNField": ".engine",
    # Filesystem helpers (load inputs, persist results)
    "load_document": ".io",
    "load_schema": ".io",
    "save_results": ".io",
    "load_results": ".io",
    # Tabular export (optional pandas dependency)
    "results_to_dataframe": ".export",
    "result_to_dataframe": ".export",
    "results_to_csv": ".export",
    # Grounding visualization (stdlib only)
    "save_html": ".viz",
    # Provider factory and response caches
    "from_model": ".providers",
    "ResponseCache": ".providers",
    "MemoryCache": ".providers",
    "DiskCache": ".providers",
    # Config
    "ExtractionConfig": ".config",
    # Types
    "ExtractionResult": ".types",
    "FieldResult": ".types",
    "Metadata": ".types",
    "ExtractionStatus": ".types",
    # Exceptions
    "NFieldError": ".exceptions",
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
