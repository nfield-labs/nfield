"""Per-run token usage accounting.

Every provider response carries the model's own token counts; this module folds
them into a per-run counter so the result can report what the run actually spent.
The counter lives in a :mod:`contextvars` variable: the engine starts a fresh one
per ``extract()`` call, and every leaf task spawned under that call inherits it,
so concurrent documents in a batch each keep an exact, isolated tally. A cache
hit never records usage - the counter reflects real API spend only.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass

__all__ = ["Usage", "record_usage", "start_usage"]


@dataclass
class Usage:
    """Mutable token tally for one extraction run.

    Attributes:
        prompt_tokens: Input tokens across every API call in the run.
        completion_tokens: Output tokens across every API call in the run.
        calls: Number of API calls that reported usage.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    def cost(self, pricing: tuple[float, float]) -> float:
        """Return the run's cost in USD for ``(input, output)`` prices per million tokens.

        Args:
            pricing: ``(input_usd, output_usd)`` per million tokens.

        Returns:
            The dollar cost of the recorded usage.

        Example:
            >>> Usage(prompt_tokens=1_000_000, completion_tokens=0).cost((0.5, 1.5))
            0.5
        """
        input_price, output_price = pricing
        return (self.prompt_tokens * input_price + self.completion_tokens * output_price) / 1e6


# The active run's counter. ``None`` outside a tracked run, so direct provider use
# (no engine) records nothing and costs nothing to check.
_current_usage: ContextVar[Usage | None] = ContextVar("nfield_usage", default=None)


def start_usage() -> Usage:
    """Begin a fresh usage tally for the current task tree and return it.

    Tasks created after this call (the run's leaf calls) inherit the counter via
    the context; sibling runs started elsewhere get their own.

    Returns:
        The new, empty :class:`Usage` the run will accumulate into.
    """
    usage = Usage()
    _current_usage.set(usage)
    return usage


def record_usage(prompt_tokens: int | None, completion_tokens: int | None) -> None:
    """Add one API call's reported token counts to the active run's tally.

    A no-op when no run is being tracked or the provider reported nothing.

    Args:
        prompt_tokens: The call's input-token count, or ``None`` if unreported.
        completion_tokens: The call's output-token count, or ``None`` if unreported.
    """
    usage = _current_usage.get()
    if usage is None or (prompt_tokens is None and completion_tokens is None):
        return
    usage.prompt_tokens += prompt_tokens or 0
    usage.completion_tokens += completion_tokens or 0
    usage.calls += 1
