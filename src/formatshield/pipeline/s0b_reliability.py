"""Stage 0.5: Reliability calibration — the dynamic per-call field cap.

The static ``max_fields_per_call`` is a guess. This measures it. A tiny 2-point
probe asks the model to extract a known key=value document at two field counts and
records how per-field accuracy decays as more fields share one call, then derives
the cap from a target reliability the caller chooses.

Model (instruction-count collapse; IFScale arXiv:2507.11538): per-field success
under ``n`` simultaneous fields decays roughly exponentially,
``r(n) = exp(-beta * (n - 1))``. Two probe sizes ``n1 < n2`` give

    beta = (ln a1 - ln a2) / (n2 - n1)

and the cap is the largest ``n`` keeping ``r(n) >= A_target``:

    cap = 1 + ln(1 / A_target) / beta            (clamped to [CAP_MIN, CAP_MAX])

``beta`` is measured per model; ``A_target`` is the single, meaningful knob (a
quality SLA, not a magic number). One probe per engine, cached like Stage 0.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from formatshield.extraction._papt import TemplateType
from formatshield.extraction._prompt import build_extraction_prompt
from formatshield.extraction._sfep import parse_sfep
from formatshield.schema._types import Field

if TYPE_CHECKING:
    from formatshield.providers._protocol import LLMProvider

__all__ = ["calibrate_field_cap"]

logger = logging.getLogger(__name__)

# Two probe sizes: a small set the model should nail, and a larger one where
# instruction-count decay shows. Their gap is the lever arm for estimating beta.
_PROBE_SMALL_N: int = 10
_PROBE_LARGE_N: int = 60
# Clamp the derived cap to a sane band: never fewer than CAP_MIN fields per call
# (else K explodes), never more than CAP_MAX (beyond which no model is reliable).
_CAP_MIN: int = 10
_CAP_MAX: int = 100
# Output token budget per probe call: a dozen tokens per field's "path = value".
_PROBE_TOKENS_PER_FIELD: int = 14
# A probe is only trustworthy if the small set itself extracted well; below this
# the model/probe is too noisy to fit a curve, so we keep the static default.
_MIN_TRUSTWORTHY_SMALL_ACCURACY: float = 0.5


async def calibrate_field_cap(
    provider: LLMProvider,
    *,
    target_reliability: float,
    chars_per_token: float,
    static_default: int,
) -> int:
    """Measure the model's reliable fields-per-call and derive the cap.

    Args:
        provider: LLM provider for the two probe calls.
        target_reliability: Desired per-field reliability ``A_target`` in (0, 1).
        chars_per_token: Calibrated ratio (Stage 0), only for output sizing.
        static_default: Fallback cap if the probe is inconclusive.

    Returns:
        The per-call field cap (difficulty-unit budget), in ``[CAP_MIN, CAP_MAX]``,
        or ``static_default`` when the probe is too noisy to trust.
    """
    a_target = min(max(target_reliability, 0.01), 0.999)
    try:
        a1 = await _probe_accuracy(provider, _PROBE_SMALL_N, chars_per_token)
        a2 = await _probe_accuracy(provider, _PROBE_LARGE_N, chars_per_token)
    except Exception as exc:  # probe must never break a real run
        logger.warning("Field-cap probe failed (%s); using static default %d", exc, static_default)
        return static_default

    # Untrustworthy probe (small set itself failed) → keep the static default.
    if a1 < _MIN_TRUSTWORTHY_SMALL_ACCURACY:
        logger.info("Field-cap probe noisy (a1=%.2f); using static default %d", a1, static_default)
        return static_default

    # No measurable decay (large set held up) → the model tolerates a full call.
    if a2 >= a1:
        logger.info("Field-cap probe: no decay (a1=%.2f, a2=%.2f) → cap=%d", a1, a2, _CAP_MAX)
        return _CAP_MAX

    # Total collapse at the large set → fall back to the smallest reliable band.
    if a2 <= 0.0:
        return _CAP_MIN

    beta = (math.log(a1) - math.log(a2)) / (_PROBE_LARGE_N - _PROBE_SMALL_N)
    if beta <= 0.0:
        return _CAP_MAX
    cap = 1.0 + math.log(1.0 / a_target) / beta
    measured = max(_CAP_MIN, min(_CAP_MAX, int(cap)))
    logger.info(
        "Field-cap probe: a1=%.2f a2=%.2f beta=%.4f A_target=%.2f → cap=%d",
        a1,
        a2,
        beta,
        a_target,
        measured,
    )
    return measured


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


async def _probe_accuracy(provider: LLMProvider, n: int, chars_per_token: float) -> float:
    """Run one probe call: extract ``n`` known key=value facts, return accuracy.

    Builds a synthetic document of ``n`` unique ``code_i = VALUE_i`` lines and a
    matching schema, runs one extraction call, and returns the fraction of values
    recovered exactly. Self-contained and domain-agnostic — it measures the
    model's raw ability to track ``n`` simultaneous fields in one call.

    Args:
        provider: LLM provider.
        n: Number of fields to probe with.
        chars_per_token: Calibrated ratio (Stage 0), for output token sizing.

    Returns:
        Per-field accuracy in [0, 1].
    """
    fields: list[Field] = []
    truth: dict[str, str] = {}
    doc_lines: list[str] = []
    for i in range(n):
        path = f"code_{i:03d}"
        # A short, unique, unguessable value forces real reading, not priors.
        value = f"{i:03d}-{chr(65 + i % 26)}{chr(90 - i % 26)}"
        fields.append(
            Field(
                path=path,
                type="string",
                constraints={},
                parent_path="",
                schema_node={"type": "string", "description": f"The code for item {i:03d}"},
                tau=4.0,
            )
        )
        truth[path] = value
        doc_lines.append(f"item {i:03d} code: {value}")

    document = "\n".join(doc_lines)
    messages = build_extraction_prompt(fields, document, TemplateType.STANDARD)
    max_tokens = max(64, n * _PROBE_TOKENS_PER_FIELD)
    raw = await provider.complete(messages, max_tokens=max_tokens)
    extracted = parse_sfep(raw, fields)

    correct = sum(
        1 for path, want in truth.items() if str(extracted.get(path, "")).strip() == want
    )
    return correct / n if n else 0.0
