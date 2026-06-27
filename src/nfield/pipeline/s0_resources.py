"""Stage 0: Resource Calibration.

Resolves the characters-per-token ratio and populates the context-window
parameters C_eff, M_O, C_usable on the PipelineState. The ratio is the explicit
``config.chars_per_token`` override when set, otherwise a script-aware estimate
keyed by the document language. See ``providers/_token_budget.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nfield.pipeline._state import PipelineState
from nfield.providers._token_budget import chars_per_token_for_language

if TYPE_CHECKING:
    from nfield.config import ExtractionConfig
    from nfield.providers._protocol import LLMProvider

__all__ = ["run_stage_0"]


def run_stage_0(
    provider: LLMProvider,
    config: ExtractionConfig,
    *,
    language: str | None = None,
) -> PipelineState:
    """Resolve the chars-per-token ratio and compute context window parameters.

    Uses ``config.chars_per_token`` when set, otherwise a script-aware estimate
    keyed by language, then computes
    ``C_usable = C_eff * config.context_utilization_ratio``.

    Args:
        provider: LLM provider used for the extraction run.
        config: Extraction configuration; ``config.chars_per_token`` overrides the
            estimate, and ``config.document_language`` selects it otherwise.
        language: Optional explicit BCP-47 tag; ``None`` uses
            ``config.document_language``.

    Returns:
        Fresh ``PipelineState`` populated with calibration values.
    """
    lang = language or config.document_language
    # An explicit override pins the model's exact ratio; otherwise a script-aware
    # estimate keyed by language.
    chars_per_token = (
        config.chars_per_token
        if config.chars_per_token is not None
        else chars_per_token_for_language(lang)
    )

    c_eff = provider.context_window
    m_o = provider.max_output_tokens
    c_usable = c_eff * config.context_utilization_ratio

    return PipelineState(
        chars_per_token=chars_per_token,
        C_eff=c_eff,
        M_O=m_o,
        C_usable=c_usable,
    )
