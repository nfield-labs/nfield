"""nfield adapter - the method under test.

Wraps the library's :func:`nfield.nfield` entry point in the uniform
:class:`Adapter` interface. The library import is deferred to call time so the
pure scorer (and CI that only exercises it) never needs a provider SDK installed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ._base import AdapterOutput
from ._errors import classify_exc

if TYPE_CHECKING:
    from nfield import ExtractionResult

# Retry budget applied to every method equally (fairness rule). Kept on the
# adapter, not hard-coded in run(), so a sweep can hold it constant across the
# whole roster from one place.
DEFAULT_MAX_RETRY_ROUNDS: int = 1


@dataclass(frozen=True, slots=True)
class NfieldAdapter:
    """Run nfield and normalise its result into an :class:`AdapterOutput`.

    Args:
        max_retry_rounds: Retry budget passed to the pipeline; held identical to
            whatever the baselines receive so the comparison stays fair.
        api_key: Explicit provider credential. ``None`` falls back to the SDK's
            own environment pickup (e.g. ``GROQ_API_KEY``).
        base_url: Optional provider base URL for a proxy / gateway endpoint.
    """

    name: str = field(default="nfield", init=False)
    max_retry_rounds: int = DEFAULT_MAX_RETRY_ROUNDS
    closed_book: bool = False
    api_key: str | None = None
    base_url: str | None = None

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
        """Extract ``schema`` from ``document`` with nfield on ``model``.

        ``instructions`` are passed straight to nfield, which threads them into
        every leaf's user message. Any provider/API error is caught and returned
        as a failed :class:`AdapterOutput`, never raised - a failed run scores as a
        miss and stays in the denominator.
        """
        from nfield import nfield
        from nfield.config import ExtractionConfig

        started = time.perf_counter()
        try:
            result = nfield(
                document,
                schema,
                model,
                context_window=context_window,
                max_output_tokens=max_output_tokens,
                api_key=self.api_key,
                base_url=self.base_url,
                instructions=instructions,
                config=ExtractionConfig(
                    max_retry_rounds=self.max_retry_rounds, closed_book=self.closed_book
                ),
            )
        except Exception as exc:  # a baseline-fair failure: record, never abort the sweep
            # A whole-engine failure means no leaf returned, so every targeted field
            # was lost to the call, not to the model - credit it all to call-failed
            # (design §4.3 / §7). The category is labelled honestly like the baselines.
            kind, message = classify_exc(exc)
            total = _schema_field_count(schema)
            return AdapterOutput(
                data={},
                fields_total=total,
                fields_extracted=0,
                k=0,
                k_min=0,
                call_failed=total,
                elapsed_seconds=round(time.perf_counter() - started, 3),
                error=f"{kind.value}: {message}",
                error_category=kind.value,
            )
        return _to_output(result, round(time.perf_counter() - started, 3))


def _to_output(result: ExtractionResult, elapsed: float) -> AdapterOutput:
    meta = result.metadata
    return AdapterOutput(
        data=result.data,
        fields_total=meta.fields_total,
        fields_extracted=meta.fields_extracted,
        k=meta.K,
        k_min=meta.K_min,
        call_failed=meta.fields_call_failed,
        elapsed_seconds=elapsed,
        raw={"status": result.status.value, "quality_score": meta.quality_score},
    )


def _schema_field_count(schema: dict[str, Any]) -> int:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return 0
    total = 0
    for node in properties.values():
        if isinstance(node, dict) and node.get("type") == "object":
            total += _schema_field_count(node)
        else:
            total += 1
    return total
