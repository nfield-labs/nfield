"""The Adapter protocol and its uniform output record.

Every method under test — nfield and each baseline — implements :class:`Adapter`
and returns an :class:`AdapterOutput`. Same signature, same return shape, so the
runner sweeps them identically and the scorer compares apples to apples. The
fairness rules (same model, same prompt budget, same retry budget) live in the
adapters; this module only fixes the interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = ["Adapter", "AdapterOutput"]


@dataclass(frozen=True, slots=True)
class AdapterOutput:
    """The result of running one method on one (document, schema) pair.

    The scorer only needs :attr:`data`; the remaining fields carry the
    efficiency metrics and the failure signal so the runner can record a fair,
    reproducible row without re-deriving anything.

    Args:
        data: Extracted fields as a nested dict matching the schema. Empty when
            the call failed — the scorer then judges every gold field a miss.
        fields_total: Schema field count the method targeted (the x-axis, N).
        fields_extracted: Count of fields the method returned a value for.
        k: Number of model calls the method made (1 for single-call baselines).
        k_min: Lower bound on calls, if the method reports one (nfield does).
        call_failed: Fields lost to an API/call error rather than to the model
            failing to extract them. Carried into the scorer as its own category.
        elapsed_seconds: Wall-clock latency of the run.
        error: Failure message if the run errored, else ``None``. A non-``None``
            error means ``data`` is unreliable and the run scores as a miss —
            it is never dropped from the denominator. Carries a clean, classified
            reason (see ``error_category``), not a raw SDK dump.
        error_category: Stable failure category (e.g. ``single_call_output_ceiling``,
            ``json_truncated``), or ``None`` on success. Lets the table report *why*
            a method failed without parsing the message.
        raw: Optional provider-level raw payload, retained for re-scoring.
    """

    data: dict[str, Any]
    fields_total: int
    fields_extracted: int
    k: int
    k_min: int
    call_failed: int = 0
    elapsed_seconds: float = 0.0
    error: str | None = None
    error_category: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        """Return ``True`` when the run errored and its data must score as a miss."""
        return self.error is not None


@runtime_checkable
class Adapter(Protocol):
    """Uniform interface every benchmarked method implements.

    Implementations must catch their own model/API failures and return an
    :class:`AdapterOutput` with :attr:`AdapterOutput.error` set — never raise.
    Refusing a hard schema is exactly the capability the benchmark measures, so
    a failure must stay in the denominator, not abort the sweep.
    """

    @property
    def name(self) -> str:
        """Stable identifier for the method, used in result paths and tables."""
        ...

    def run(
        self,
        document: str,
        schema: dict[str, Any],
        *,
        model: str,
        context_window: int,
        max_output_tokens: int,
        instructions: str = "",
    ) -> AdapterOutput:
        """Extract ``schema`` from ``document`` with ``model``.

        Args:
            document: The source text to extract from.
            schema: The target JSON Schema.
            model: Provider-qualified model id, e.g. ``"groq/llama-3.3-70b"``.
            context_window: Input-window budget, shared by every method. nfield's
                capacity planner decomposes within it; the baselines fit the
                document to it. Same window for all, so the comparison is fair.
            max_output_tokens: Output budget, shared by every method. nfield sizes
                each leaf's output by it; the baselines set ``max_tokens`` to it.
            instructions: Domain guidance, given identically to every method, so
                the comparison stays fair. nfield threads it to each leaf; the
                baselines lead their user message with it.

        Returns:
            An :class:`AdapterOutput`; on failure, one with ``error`` set.
        """
        ...
