"""FormatShield inference backends package.

Core backends (always available):
    GroqBackend, OpenAIBackend, OpenRouterBackend, OllamaBackend,
    AnthropicBackend, VLLMBackend, DryRunBackend

Optional backends (require extras):
    CohereBackend   — pip install 'formatshield[cohere]'
    MistralBackend  — pip install 'formatshield[mistral]'
    TogetherBackend — pip install 'formatshield[together]'
    OutlinesBackend — pip install 'formatshield[outlines]'
    GuidanceBackend — pip install 'formatshield[guidance]'
"""

from formatshield.backends.anthropic_backend import AnthropicBackend
from formatshield.backends.groq_backend import GroqBackend
from formatshield.backends.ollama_backend import OllamaBackend
from formatshield.backends.openai_backend import OpenAIBackend
from formatshield.backends.openrouter_backend import OpenRouterBackend
from formatshield.backends.protocol import (
    Backend,
    BackendName,
    ModelFamily,
    get_backend_name_from_model,
    get_model_family,
)
from formatshield.backends.vllm_backend import VLLMBackend

__all__ = [
    "AnthropicBackend",
    "Backend",
    "BackendName",
    "GroqBackend",
    "ModelFamily",
    "OllamaBackend",
    "OpenAIBackend",
    "OpenRouterBackend",
    "VLLMBackend",
    "get_backend_name_from_model",
    "get_model_family",
]
