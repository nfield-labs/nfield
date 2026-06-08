"""Token counting and chars-per-token measurement.

Measures the ratio of characters to tokens for a specific model by
sampling the model's token-counting API. Supports fallback ratios for
different languages when API is unavailable.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import TYPE_CHECKING

from formatshield.exceptions import ProviderError

if TYPE_CHECKING:
    from formatshield.providers._protocol import LLMProvider

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FALLBACK_CHARS_PER_TOKEN_EN: float = 3.5
_FALLBACK_CHARS_PER_TOKEN_CJK: float = 1.5
_FALLBACK_CHARS_PER_TOKEN_MIXED: float = 2.5


# ---------------------------------------------------------------------------
# Load calibration samples
# ---------------------------------------------------------------------------


def _load_calibration_samples() -> dict[str, list[str]]:
    """Load multilingual calibration samples from _data/calibration_samples.json.

    Returns:
        Dictionary mapping language code ("en", "cjk", "mixed") to list of sample strings.
    """
    try:
        sample_file = resources.files("formatshield") / "_data" / "calibration_samples.json"
        content = sample_file.read_text(encoding="utf-8")
        data: dict[str, list[str]] = json.loads(content)
        return data
    except (OSError, json.JSONDecodeError):
        # Missing/unreadable file or bad JSON → run with no samples (callers fall
        # back to the per-language hardcoded ratio). OSError covers
        # FileNotFoundError / IsADirectoryError / permission errors.
        return {"en": [], "cjk": [], "mixed": []}


_CALIBRATION_SAMPLES = _load_calibration_samples()


# ---------------------------------------------------------------------------
# Measurement function
# ---------------------------------------------------------------------------


async def measure_chars_per_token(
    provider: LLMProvider,
    *,
    language: str = "en",
) -> float:
    """Measure chars-per-token ratio for a specific provider + model.

    Calls the provider's token counter on a representative sample in the requested
    language and returns chars / tokens. The counter is an API call only if the
    provider supports one; providers without a token endpoint (e.g. Groq) return a
    local estimate, in which case the ratio is an estimate, not a measurement. Falls
    back to a hardcoded per-language ratio when no samples or the counter fails.

    Args:
        provider: LLMProvider instance (or compatible with count_tokens()).
        language: Language for sample selection: "en" (English),
            "cjk" (CJK languages), or "mixed" (multilingual).
            Defaults to "en".

    Returns:
        Characters per token ratio for this model in the given language.
        Higher = fewer tokens per character (more efficient compression).

    Raises:
        ValueError: If language is not recognized.

    Example:
        >>> import asyncio
        >>> # provider = create_provider(...)  # some real provider
        >>> # ratio = await measure_chars_per_token(provider, language="en")
        >>> # print(f"Ratio: {ratio:.2f}")
        >>> # Ratio: 3.45
    """
    if language not in ("en", "cjk", "mixed"):
        raise ValueError(f"Unknown language: {language!r}. Must be 'en', 'cjk', or 'mixed'.")

    # Get samples for the language
    samples = _CALIBRATION_SAMPLES.get(language, [])
    if not samples:
        # No samples available; use fallback
        return _get_fallback_ratio(language)

    # Concatenate all samples
    sample_text = " ".join(samples)

    try:
        # Call provider's token counting API
        token_count = await provider.count_tokens(sample_text)
        char_count = len(sample_text)

        if token_count <= 0:
            # Degenerate case; use fallback
            return _get_fallback_ratio(language)

        ratio: float = float(char_count / token_count)
        return ratio
    except ProviderError:
        # API call failed or provider doesn't support token counting
        # Fall back to hardcoded ratio
        return _get_fallback_ratio(language)


# ---------------------------------------------------------------------------
# Fallback ratios
# ---------------------------------------------------------------------------


def _get_fallback_ratio(language: str) -> float:
    """Get hardcoded fallback chars-per-token ratio for a language.

    Args:
        language: Language code ("en", "cjk", "mixed").

    Returns:
        Fallback ratio.
    """
    if language == "cjk":
        return _FALLBACK_CHARS_PER_TOKEN_CJK
    elif language == "mixed":
        return _FALLBACK_CHARS_PER_TOKEN_MIXED
    else:  # "en" or unknown
        return _FALLBACK_CHARS_PER_TOKEN_EN
