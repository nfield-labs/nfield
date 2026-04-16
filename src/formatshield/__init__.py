"""
FormatShield — Route LLM outputs intelligently. Measure what structured generation costs you.

Prior work shows constrained decoding costs LLMs up to 27% accuracy on reasoning tasks
(arXiv 2408.02442). FormatShield routes around it automatically — one import.

Usage::

    import formatshield as fs
    from pydantic import BaseModel

    class MySchema(BaseModel):
        answer: str
        confidence: float

    # Async (default):
    result = await fs.generate(prompt, MySchema, model="groq/llama-3.3-70b-versatile")

    # Sync:
    shield = fs.FormatShield(model="groq/llama-3.3-70b-versatile")
    result = shield.generate_sync(prompt, MySchema)
"""

# Auto-load .env so users never need to `export API_KEY` in their shell.
# Finds the nearest .env file walking up from cwd (same behaviour as dotenv CLI).
try:
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv(override=False)  # don't override vars that are already set in the environment
except ImportError:
    pass  # python-dotenv not installed — env vars must be set manually

from typing import Any

from formatshield import types as types
from formatshield.core import FormatShield, GenerationResult, generate
from formatshield.generator import AsyncFormatShieldGenerator, FormatShieldGenerator
from formatshield.oracle.routing_decision import RoutingDecision
from formatshield.scorer.features import BenchmarkResult, ComplexityFeatures, StreamEvent
from formatshield.types import (
    cfg as cfg,
)
from formatshield.types import (
    json_schema as json_schema,
)
from formatshield.types import (
    regex as regex,
)

__version__ = "0.1.0"


def from_provider(model: str, **kwargs: Any) -> FormatShield:
    """Unified factory — auto-detects provider from model string prefix.

    Args:
        model: Model identifier in ``"provider/model-name"`` format.
            Examples: ``"groq/llama-3.1-70b-versatile"``,
            ``"openai/gpt-4o-mini"``, ``"dryrun/test"``.
        **kwargs: Additional keyword arguments forwarded to :class:`FormatShield`.

    Returns:
        :class:`FormatShield` instance configured for the detected provider.

    Example::

        shield = fs.from_provider("groq/llama-3.1-70b-versatile")
        shield = fs.from_provider("openai/gpt-4o-mini", debug=True)
    """
    return FormatShield(model=model, **kwargs)


__all__ = [
    "AsyncFormatShieldGenerator",
    "BenchmarkResult",
    "ComplexityFeatures",
    "FormatShield",
    "FormatShieldGenerator",
    "GenerationResult",
    "RoutingDecision",
    "StreamEvent",
    "__version__",
    "cfg",
    "from_provider",
    "generate",
    "json_schema",
    "regex",
    "types",
]
