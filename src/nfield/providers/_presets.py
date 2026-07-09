"""OpenAI-compatible provider presets.

Many hosted endpoints speak the OpenAI ``/v1/chat/completions`` API. Each one is
``OpenAIProvider`` pointed at a fixed base URL and reading a provider-specific key,
so it needs only a row in the table below, not its own class. The model name is
passed through exactly as the endpoint lists it.

Base URLs and key variables are taken from each provider's own API documentation.
"""

from __future__ import annotations

import os

from nfield.providers.openai._provider import OpenAIProvider

# prefix -> (base_url, env_var). env_var is None for a local server that needs no
# key. Verified against each provider's OpenAI-compatibility docs.
OPENAI_COMPATIBLE_PRESETS: dict[str, tuple[str, str | None]] = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "fireworks": ("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "perplexity": ("https://api.perplexity.ai", "PERPLEXITY_API_KEY"),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
}

# Local OpenAI-compatible servers (Ollama) require a non-empty key the SDK sends
# but the server ignores.
_LOCAL_PLACEHOLDER_KEY: str = "not-needed"


def build_preset_provider(
    prefix: str,
    model_name: str,
    *,
    context_window: int | None = None,
    max_output_tokens: int | None = None,
    max_retries: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    reasoning_model: bool = False,
) -> OpenAIProvider:
    """Build an OpenAIProvider configured for an OpenAI-compatible preset.

    The preset supplies the base URL and the environment variable to read the key
    from. An explicit ``api_key`` or ``base_url`` overrides the preset. A local
    preset with no key variable uses a placeholder key the server ignores.

    Args:
        prefix: Preset name; must be a key of ``OPENAI_COMPATIBLE_PRESETS``.
        model_name: Model name as the endpoint lists it.
        context_window: Real context window in tokens, or None for the default.
        max_output_tokens: Real output ceiling in tokens, or None for the default.
        max_retries: Transient-failure retry budget, or None for the default.
        api_key: Explicit key. None reads the preset's environment variable.
        base_url: Explicit endpoint. None uses the preset's base URL.
        reasoning_model: When True, disable the model's thinking per call.

    Returns:
        An ``OpenAIProvider`` pointed at the preset endpoint.

    Example:
        >>> provider = build_preset_provider("deepseek", "deepseek-chat")
        >>> provider.model_name
        'deepseek-chat'
    """
    preset_base, env_var = OPENAI_COMPATIBLE_PRESETS[prefix]
    resolved_key: str | None
    if api_key is not None:
        resolved_key = api_key
    elif env_var is not None:
        resolved_key = os.environ.get(env_var)
    else:
        resolved_key = _LOCAL_PLACEHOLDER_KEY
    return OpenAIProvider(
        model_name,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        max_retries=max_retries,
        api_key=resolved_key,
        base_url=base_url or preset_base,
        reasoning_model=reasoning_model,
    )
