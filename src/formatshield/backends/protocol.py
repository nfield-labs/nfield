"""Backend Protocol definition — all adapters must implement this interface."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from formatshield.scorer.features import StreamEvent

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

BackendName = Literal["vllm", "ollama", "groq", "openrouter", "outlines", "guidance"]
"""Union of all recognised backend identifiers."""

ModelFamily = Literal["openai", "anthropic", "meta", "mistral", "deepseek", "google", "unknown"]
"""High-level model family / provider label."""


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Backend(Protocol):
    """
    Structural interface that every FormatShield inference backend must satisfy.

    Backends wrap a specific inference runtime (vLLM, Ollama, Groq, OpenRouter, …)
    and expose a uniform async API for both single-shot generation and streaming.
    They are discovered and instantiated by :func:`formatshield.router.build_backend`.
    """

    #: Short, stable identifier for this backend (e.g. ``"vllm"``).
    name: str

    # ------------------------------------------------------------------
    # Core generation methods
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        schema: dict | None = None,
        constraints: str | None = None,
        kv_cache_prefix: str | None = None,
    ) -> str:
        """
        Generate a response for *prompt* and return the full text.

        Parameters
        ----------
        prompt:
            The user / system prompt to send to the model.
        schema:
            Optional JSON schema dict describing the expected output shape.
            Backends may use this to enable JSON-mode or grammar-constrained
            generation.
        constraints:
            Optional constraint hint string.  The special value ``"json"``
            requests JSON-only output.
        kv_cache_prefix:
            Optional prefix string to prepend as a system message.  When
            provided by a backend that supports prefix caching (e.g. vLLM),
            the prefix's KV activations can be reused across requests.

        Returns
        -------
        str
            The model's generated text (stripped of any wrapper markup).
        """
        ...

    async def stream(
        self,
        prompt: str,
        schema: dict | None = None,
        constraints: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream the model's response as an async iterator of :class:`StreamEvent`.

        Backends should yield one ``"output"`` event per incremental token and
        a single ``"complete"`` event carrying the full text at the end.

        Parameters
        ----------
        prompt:
            The user / system prompt to send to the model.
        schema:
            Optional JSON schema dict.  Same semantics as in :meth:`generate`.
        constraints:
            Optional constraint hint string.  Same semantics as in
            :meth:`generate`.

        Yields
        ------
        StreamEvent
            Incremental ``output`` events followed by a final ``complete``
            event.
        """
        ...

    # ------------------------------------------------------------------
    # Capability properties
    # ------------------------------------------------------------------

    @property
    def supports_kv_cache_reuse(self) -> bool:
        """
        ``True`` if the backend supports server-side KV-cache prefix reuse.

        When this is ``True``, :meth:`generate` will make use of the
        *kv_cache_prefix* argument to amortise repeated prompt prefixes
        (e.g. system prompts shared across many requests).
        """
        ...

    @property
    def accuracy_loss_baseline(self) -> float | None:
        """
        Estimated baseline accuracy loss (0.0–1.0) for structured-output
        generation on this backend, derived from benchmark literature.

        A value of ``0.18`` means the backend produces invalid or incomplete
        structured outputs ~18 % of the time without TTF.  ``None`` means
        no estimate is available.

        This value is used by :class:`~formatshield.oracle.ThresholdOracle`
        when computing the expected accuracy gain of activating TTF for a
        given request.
        """
        ...


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

# Mapping of known prefixes to BackendName
_PREFIX_TO_BACKEND: dict[str, BackendName] = {
    "vllm": "vllm",
    "ollama": "ollama",
    "groq": "groq",
    "openrouter": "openrouter",
    "outlines": "outlines",
    "guidance": "guidance",
}

# Mapping of model id sub-strings to ModelFamily
_FAMILY_PATTERNS: list[tuple[str, ModelFamily]] = [
    # OpenAI
    ("gpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("text-davinci", "openai"),
    # Anthropic
    ("claude", "anthropic"),
    # Meta / LLaMA
    ("llama", "meta"),
    ("meta-llama", "meta"),
    # Mistral / Mixtral
    ("mistral", "mistral"),
    ("mixtral", "mistral"),
    # DeepSeek
    ("deepseek", "deepseek"),
    # Google
    ("gemini", "google"),
    ("gemma", "google"),
    ("palm", "google"),
    ("bison", "google"),
]


def get_backend_name_from_model(model_id: str) -> BackendName:
    """
    Derive a :data:`BackendName` from a ``"provider/model"`` model identifier.

    The function inspects the portion before the first ``"/"`` and maps it to
    a known backend name.  If no prefix is present, or if the prefix is not
    recognised, it defaults to ``"openrouter"`` (the most general hosted
    backend).

    Parameters
    ----------
    model_id:
        Model identifier in ``"backend/model-name"`` format, e.g.
        ``"groq/llama-3.1-70b-versatile"`` or ``"vllm/meta-llama/Llama-3-70b"``.

    Returns
    -------
    BackendName
        The matched backend name, or ``"openrouter"`` if unknown.

    Examples
    --------
    >>> get_backend_name_from_model("groq/llama-3.1-70b-versatile")
    'groq'
    >>> get_backend_name_from_model("vllm/meta-llama/Llama-3-70b-Instruct")
    'vllm'
    >>> get_backend_name_from_model("gpt-4o")
    'openrouter'
    """
    if "/" in model_id:
        prefix = model_id.split("/", maxsplit=1)[0].lower()
        if prefix in _PREFIX_TO_BACKEND:
            return _PREFIX_TO_BACKEND[prefix]
    return "openrouter"


def get_model_family(model_id: str) -> ModelFamily:
    """
    Classify a model identifier into a high-level :data:`ModelFamily`.

    The model id is normalised to lower-case and matched against a priority-
    ordered list of sub-string patterns.  The first match wins.

    Parameters
    ----------
    model_id:
        Model identifier, optionally prefixed with a backend name
        (e.g. ``"groq/llama-3.1-70b-versatile"``).  The backend prefix is
        stripped before matching.

    Returns
    -------
    ModelFamily
        The matched model family, or ``"unknown"`` if none matched.

    Examples
    --------
    >>> get_model_family("groq/llama-3.1-70b-versatile")
    'meta'
    >>> get_model_family("openrouter/anthropic/claude-3-5-sonnet")
    'anthropic'
    >>> get_model_family("gpt-4o")
    'openai'
    """
    # Strip any leading "backend/" prefix from known backends
    normalised = model_id.lower()
    if "/" in normalised:
        prefix = normalised.split("/", maxsplit=1)[0]
        if prefix in _PREFIX_TO_BACKEND:
            # Remove the backend prefix so "groq/llama-…" → "llama-…"
            normalised = normalised.split("/", maxsplit=1)[1]

    for pattern, family in _FAMILY_PATTERNS:
        if pattern in normalised:
            return family

    return "unknown"
