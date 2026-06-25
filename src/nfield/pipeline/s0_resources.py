"""Stage 0: Resource Calibration.

Measures chars_per_token for the model (an API call only if the provider exposes a
token endpoint; providers without one, e.g. Groq, return a local estimate) and
populates the context-window parameters C_eff, M_O, C_usable on the PipelineState.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nfield.pipeline._state import PipelineState
from nfield.providers._token_count import measure_chars_per_token

if TYPE_CHECKING:
    from nfield.config import ExtractionConfig
    from nfield.providers._protocol import LLMProvider

__all__ = ["run_stage_0"]

# CJK primary subtags (ISO 639): map to the "cjk" calibration bucket.
_CJK_PRIMARY_SUBTAGS: frozenset[str] = frozenset({"ja", "zh", "ko", "yue", "wuu", "cmn"})


def _calibration_bucket(language: str) -> str:
    """Map a BCP-47 tag to a calibration sample bucket (en / cjk / mixed).

    measure_chars_per_token only knows the three buckets; any other tag maps to
    ``mixed`` so a real language tag (e.g. ``"ja"``, ``"fr"``) never crashes it.

    Args:
        language: A BCP-47 tag or a bucket name.

    Returns:
        One of ``"en"``, ``"cjk"``, ``"mixed"``.
    """
    normalized = language.strip().lower()
    if normalized in ("en", "cjk", "mixed"):
        return normalized
    primary = normalized.split("-", 1)[0]
    if primary in _CJK_PRIMARY_SUBTAGS:
        return "cjk"
    if primary == "en":
        return "en"
    return "mixed"


async def run_stage_0(
    provider: LLMProvider,
    config: ExtractionConfig,
    *,
    language: str | None = None,
) -> PipelineState:
    """Calibrate token ratios and compute context window parameters.

    Measures ``chars_per_token`` for the current model (an API call only if the
    provider has a token endpoint; otherwise a local estimate), then computes
    ``C_usable = C_eff * config.context_utilization_ratio``.

    Args:
        provider: LLM provider used for the extraction run.
        config: Extraction configuration; ``config.document_language`` selects the
            calibration sample language unless ``language`` overrides it.
        language: Optional explicit BCP-47 tag; ``None`` uses
            ``config.document_language``. Either is mapped to a calibration bucket.

    Returns:
        Fresh ``PipelineState`` populated with calibration values.

    Example:
        >>> callable(run_stage_0)
        True
    """
    bucket = _calibration_bucket(language or config.document_language)
    chars_per_token = await measure_chars_per_token(provider, language=bucket)

    c_eff = provider.context_window
    m_o = provider.max_output_tokens
    c_usable = c_eff * config.context_utilization_ratio

    state = PipelineState(
        chars_per_token=chars_per_token,
        C_eff=c_eff,
        M_O=m_o,
        C_usable=c_usable,
    )
    return state
