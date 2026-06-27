"""LLM provider protocol for structural typing.

Defines the LLMProvider Protocol that all provider implementations must
satisfy. Uses Python's structural subtyping (PEP 544) so providers don't
need explicit inheritance.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# LLMProvider Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers supporting async completion.

    Any class that implements these methods and properties can be used as
    an LLMProvider. No explicit inheritance required (structural typing).

    Note: The @runtime_checkable decorator enables isinstance() checks, but
    Python's Protocol does NOT enforce keyword-only parameter constraints at
    runtime. Implementers MUST follow the signature exactly (complete() requires
    max_tokens as keyword-only). This is verified statically by mypy but not
    checked at runtime.

    Methods:
        complete(messages, max_tokens): Generate text completion.

    Properties:
        context_window: Total context size (input + output) in tokens.
        max_output_tokens: Maximum output tokens for a single call.
        model_name: Name of the model (e.g., "gpt-4", "llama-3-8b").
    """

    async def complete(self, messages: list[dict[str, str]], *, max_tokens: int) -> str:
        """Generate a text completion from the LLM.

        Args:
            messages: List of message dictionaries with "role" and "content" keys.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text (content of the response).

        Raises:
            ProviderError: On API failures or rate limiting.
        """
        ...

    @property
    def context_window(self) -> int:
        """Total context window size (input + output) in tokens.

        Returns:
            Total context size.
        """
        ...

    @property
    def max_output_tokens(self) -> int:
        """Maximum output tokens for a single API call.

        Returns:
            Maximum output tokens.
        """
        ...

    @property
    def model_name(self) -> str:
        """Name of the underlying model.

        Returns:
            Model name (e.g., "gpt-4", "llama-3-8b").
        """
        ...
