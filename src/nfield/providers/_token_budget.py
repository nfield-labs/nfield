"""Script-aware characters-per-token estimator.

Capacity planning converts a model's token-denominated context window into a
character budget, which needs a characters-per-token ratio. The exact ratio is
the target model's own tokenizer - unavailable for an arbitrary model behind a
``base_url`` - so a static, script-keyed estimate stands in. It is
provider-agnostic (no tokenizer dependency), needs no network call, and carries
no per-model table that goes stale on each release. Callers that know their
model's exact ratio override it via ``ExtractionConfig.chars_per_token``.

The per-script values come from published tokenizer behaviour, not a per-model
lookup: Latin prose packs ~4 chars/token, CJK ~1.5 (logographic, high
fertility), and other scripts fall between.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Script-aware ratios
# ---------------------------------------------------------------------------

# Latin prose: OpenAI's published "~1 token ≈ 4 characters" English guidance.
_CHARS_PER_TOKEN_EN: float = 4.0
# CJK is logographic; BPE tokenizers (GPT/Llama) assign roughly one token per
# CJK character, so chars/token is far lower than Latin (tokenizer fertility).
_CHARS_PER_TOKEN_CJK: float = 1.5
# Other scripts (Cyrillic/Arabic/Indic/code, non-English Latin): fragment more
# than English yet pack more per character than CJK - a middle estimate.
_CHARS_PER_TOKEN_OTHER: float = 3.0

# CJK primary subtags (ISO 639) that map to the CJK ratio.
_CJK_PRIMARY_SUBTAGS: frozenset[str] = frozenset({"ja", "zh", "ko", "yue", "wuu", "cmn"})


def _language_bucket(language: str) -> str:
    """Map a BCP-47 tag (or a bucket name) to ``en`` / ``cjk`` / ``other``.

    Args:
        language: A BCP-47 tag (e.g. ``"ja"``, ``"en-US"``) or a bucket name.

    Returns:
        One of ``"en"``, ``"cjk"``, ``"other"``.
    """
    normalized = language.strip().lower()
    if normalized in ("en", "cjk"):
        return normalized
    if normalized in ("mixed", "other"):
        return "other"
    primary = normalized.split("-", 1)[0]
    if primary in _CJK_PRIMARY_SUBTAGS:
        return "cjk"
    if primary == "en":
        return "en"
    return "other"


def chars_per_token_for_language(language: str) -> float:
    """Return the script-aware characters-per-token estimate for a language.

    Args:
        language: A BCP-47 tag or a bucket name (``en`` / ``cjk`` / ``mixed``).

    Returns:
        A positive characters-per-token ratio.

    Example:
        >>> chars_per_token_for_language("ja")
        1.5
        >>> chars_per_token_for_language("en")
        4.0
    """
    bucket = _language_bucket(language)
    if bucket == "cjk":
        return _CHARS_PER_TOKEN_CJK
    if bucket == "other":
        return _CHARS_PER_TOKEN_OTHER
    return _CHARS_PER_TOKEN_EN
