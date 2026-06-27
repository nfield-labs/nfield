"""Per-model native limits, used by the ``native`` budget profile.

The benchmark compares methods under two budgets: ``native`` (the model's real
ceilings) and ``constrained`` (a fixed small budget). The native ceilings are a
property of the model/provider, not of any method, so they live here in one
registry. Adding a provider or model is a single row - nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ModelLimits", "native_limits"]


@dataclass(frozen=True, slots=True)
class ModelLimits:
    """A model's input/output ceilings.

    Args:
        context_window: Maximum input tokens the model accepts.
        max_output_tokens: Provider-published maximum output tokens per call.
        reliable_output_tokens: Largest single-call output that *reliably* returns,
            bounded by the provider's wall-clock completion limit rather than the
            token ceiling. The ``native`` budget uses this - see below.
    """

    context_window: int
    max_output_tokens: int
    reliable_output_tokens: int


# Keyed by the bare model id (no ``provider/`` prefix). Published limits, e.g. Groq
# llama-3.3-70b-versatile: 131,072 context / 32,768 output
# (https://console.groq.com/docs/model/llama-3.3-70b-versatile).
#
# reliable_output_tokens (24,000) < max_output_tokens (32,768): Groq aborts any
# single completion that runs past ~120s. Generation is ~260 tok/s, so the
# published 32,768 ceiling (~126s) deterministically 502s, and ~28k (~108s)
# straddles the wall (≈50% 502, measured). 24,000 (~94s) finishes with margin on
# every run, so it is the honest "largest single call this model completes".
_REGISTRY: dict[str, ModelLimits] = {
    "llama-3.3-70b-versatile": ModelLimits(
        context_window=131_072, max_output_tokens=32_768, reliable_output_tokens=24_000
    ),
}


def native_limits(model: str) -> ModelLimits:
    """Return the native limits for ``model`` (provider prefix is stripped).

    Args:
        model: Model id, with or without a ``provider/`` prefix.

    Returns:
        The registered :class:`ModelLimits`.

    Raises:
        KeyError: If the model has no registered limits; register it here first.
    """
    bare = model.split("/", 1)[1] if "/" in model else model
    try:
        return _REGISTRY[bare]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(
            f"no native limits registered for {bare!r}; add a row to benchmark/models.py "
            f"(known: {known})"
        ) from None
