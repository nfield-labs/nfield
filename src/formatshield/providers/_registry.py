"""Provider registry and factory routing.

Maps model string prefixes (e.g., "groq/llama-3.1-8b") to provider
implementations. Extensible by design: adding a new provider requires
only one line in the registry map.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from formatshield.exceptions import ProviderError

if TYPE_CHECKING:
    from formatshield.providers._protocol import LLMProvider

# ---------------------------------------------------------------------------
# Provider registry mapping
# ---------------------------------------------------------------------------

# Maps provider prefix (string before "/") to provider factory function.
# Format: "provider_name" -> (lazy import, class name)
_PROVIDER_REGISTRY: dict[str, tuple[str, str]] = {
    "groq": ("formatshield.providers.groq", "GroqProvider"),
}


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def from_model(model_string: str) -> LLMProvider:
    """Create an LLM provider from a model string identifier.

    Supports model strings of the form "provider/model-name", where the
    provider prefix is used to route to the correct provider class.
    The model name after "/" is passed directly to the provider — any
    model supported by the provider's API can be used.

    Args:
        model_string: Model identifier in format "provider/model-name",
            e.g., "groq/llama-3.1-8b".

    Returns:
        Instantiated provider object.

    Raises:
        ProviderError: If provider prefix is not registered.
        ValueError: If model_string is malformed.

    Example:
        >>> provider = from_model("groq/llama-3.1-8b")
        >>> provider.model_name
        'llama-3.1-8b'
    """
    # Parse model string
    if "/" not in model_string:
        raise ValueError(
            f"Invalid model string: {model_string!r}. "
            f"Expected format: 'provider/model-name' (e.g., 'groq/llama-3.1-8b')"
        )

    provider_name, model_name = model_string.split("/", 1)
    provider_name = provider_name.lower().strip()
    model_name = model_name.strip()

    if not provider_name or not model_name:
        raise ValueError(
            f"Invalid model string: {model_string!r}. Provider and model name must be non-empty."
        )

    # Look up provider in registry
    if provider_name not in _PROVIDER_REGISTRY:
        registered = ", ".join(sorted(_PROVIDER_REGISTRY.keys()))
        raise ProviderError(
            f"Unknown provider: {provider_name!r}. "
            f"Registered providers: {registered}. "
            f"Model string was: {model_string!r}"
        )

    module_name, class_name = _PROVIDER_REGISTRY[provider_name]

    # Dynamic import to avoid hard dependency
    try:
        import importlib

        module = importlib.import_module(module_name)
        provider_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ProviderError(f"Failed to import {provider_name} provider: {e}") from e

    # Instantiate and return
    return provider_class(model_name)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Registry management (for future extensibility)
# ---------------------------------------------------------------------------


def register_provider(
    provider_prefix: str,
    module_path: str,
    class_name: str,
) -> None:
    """Register a new provider in the factory registry.

    Allows users to add custom provider implementations.

    Args:
        provider_prefix: Prefix to match in model strings (e.g., "custom").
        module_path: Python module path containing the provider class
            (e.g., "my_package.providers.custom").
        class_name: Name of the provider class
            (e.g., "CustomProvider").

    Example:
        >>> register_provider("custom", "my_package.providers", "CustomProvider")
        >>> _PROVIDER_REGISTRY["custom"]
        ('my_package.providers', 'CustomProvider')
    """
    _PROVIDER_REGISTRY[provider_prefix.lower()] = (module_path, class_name)
