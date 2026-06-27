"""The two budget profiles the benchmark runs every method under.

A :class:`Budget` is one ``(context_window, max_output_tokens)`` pair applied
**uniformly** to every method, so the comparison is apples-to-apples:

- ``native``      - the model's real context, and the largest single-call output
  that *reliably* completes (``reliable_output_tokens``, bounded by the provider's
  wall-clock completion limit, not its token ceiling). Answers: given the model's
  usable single-call capacity, can one call do the job?
- ``constrained`` - one fixed small window every method shares. Answers: under a
  tight identical budget, decomposition vs a single call.

Native deliberately uses the *reliable* output, not the published 32k ceiling:
the published max deterministically 502s (the completion runs past the provider's
~120s wall), so it is not a usable single-call budget. Both budgets feed nfield's
capacity planner and the baselines identically; the only difference between
methods is what they do with the same budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

from benchmark.models import native_limits

__all__ = ["BUDGET_MODES", "Budget", "BudgetMode", "resolve_budget"]

BudgetMode = Literal["native", "constrained"]
BUDGET_MODES: tuple[BudgetMode, ...] = get_args(BudgetMode)

# The constrained profile: a single model-independent window every method shares.
# Small enough that a single call is forced to truncate on large schemas, which
# is exactly the decomposition-vs-single-call contrast this profile isolates.
_CONSTRAINED_CONTEXT_WINDOW: int = 40_000
_CONSTRAINED_MAX_OUTPUT_TOKENS: int = 8_000


@dataclass(frozen=True, slots=True)
class Budget:
    """The ``(context_window, max_output_tokens)`` every method is run under."""

    context_window: int
    max_output_tokens: int


def resolve_budget(mode: BudgetMode, model: str) -> Budget:
    """Return the :class:`Budget` for ``mode``.

    Args:
        mode: ``"native"`` (model context + reliable single-call output) or
            ``"constrained"`` (fixed small window).
        model: Model id, used to look up native limits.

    Returns:
        The :class:`Budget` every method is run under.

    Raises:
        ValueError: If ``mode`` is not a known budget mode.
    """
    if mode == "native":
        limits = native_limits(model)
        return Budget(
            context_window=limits.context_window,
            max_output_tokens=limits.reliable_output_tokens,
        )
    if mode == "constrained":
        return Budget(
            context_window=_CONSTRAINED_CONTEXT_WINDOW,
            max_output_tokens=_CONSTRAINED_MAX_OUTPUT_TOKENS,
        )
    raise ValueError(f"unknown budget mode {mode!r}; expected one of {', '.join(BUDGET_MODES)}")
