"""LLM provider abstraction layer.

Exports the provider protocol and factory for creating provider instances.
Specific provider implementations (Groq, OpenAI, etc.) are lazily imported
to minimize import time when not used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nfield.providers._registry import from_model, register_provider

if TYPE_CHECKING:
    from nfield.providers._protocol import LLMProvider

__all__ = [
    "LLMProvider",
    "from_model",
    "register_provider",
]


def __getattr__(name: str) -> object:
    """Lazy import for protocol and base classes.

    Allows importing LLMProvider from nfield.providers without
    forcing an import of the entire module at the package level.

    Args:
        name: Attribute name requested.

    Returns:
        The requested attribute.

    Raises:
        AttributeError: If attribute is not found.

    """
    if name == "LLMProvider":
        from nfield.providers._protocol import LLMProvider

        return LLMProvider
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
