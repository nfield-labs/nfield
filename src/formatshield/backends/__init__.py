"""FormatShield inference backends package.

Active backends:
    AnthropicBackend, OpenAIBackend, GroqBackend, OpenRouterBackend,
    OllamaBackend, VLLMBackend, GeminiBackend, OutlinesBackend,
    GuidanceBackend, DryRunBackend

Pending backends (backends/_pending/ — to be integrated later):
    Bedrock, Cerebras, Cohere, Fireworks, LlamaCpp, Mistral,
    Replay, SGLang, Together, Transformers, VertexAI
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
