"""Stage 0: Resource Calibration.

Makes one API call (measure_chars_per_token) and populates the context
window parameters C_eff, M_O, and C_usable on the shared PipelineState.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from formatshield.pipeline._state import PipelineState
from formatshield.providers._token_count import measure_chars_per_token

if TYPE_CHECKING:
    from formatshield.config import ExtractionConfig
    from formatshield.providers._protocol import LLMProvider

__all__ = ["run_stage_0"]


async def run_stage_0(
    provider: LLMProvider,
    config: ExtractionConfig,
    *,
    language: str = "en",
) -> PipelineState:
    """Calibrate token ratios and compute context window parameters.

    Makes one API call to measure ``chars_per_token`` for the current model,
    then computes ``C_usable = C_eff * config.context_utilization_ratio``.

    Args:
        provider: LLM provider used for the extraction run.
        config: Extraction configuration with utilization ratio.
        language: BCP-47 language code for calibration sample selection.

    Returns:
        Fresh ``PipelineState`` populated with calibration values.

    Example:
        >>> # result = await run_stage_0(provider, config)
        >>> # result.chars_per_token > 0
        True
    """
    chars_per_token = await measure_chars_per_token(provider, language=language)

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
