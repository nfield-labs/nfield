"""Source grounding - does an extracted value actually appear in the document?

Type/constraint validation (``_type_check``) checks a value's shape, not its truth: a
well-typed, in-range value can still be invented. This adds that axis - a **grounding
score** in ``[0, 1]`` for how well the source text supports a value. Two pure tiers:

* Tier 1 - deterministic ladder: exact substring → all word-tokens present → partial.
* Tier 2 - fuzzy LCS (borderline band): ``difflib`` matching blocks gated by coverage
  AND density (enough of the value, matched tightly, not scattered) - the standard
  span-grounding accept rule, scoring *support* rather than emitting a span.

Type-aware: only string/number/integer/enum are grounded. Booleans/null/structural types
are inferred, not quoted, so grounding them would flag correct inferences.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from typing import TYPE_CHECKING, Any

from nfield.validation._normalize import coerce_number

if TYPE_CHECKING:
    from nfield.schema._types import Field

__all__ = [
    "GROUNDABLE_TYPES",
    "GroundingResult",
    "GroundingStatus",
    "find_span",
    "ground_value",
    "grounding_score",
    "is_groundable",
    "is_grounded",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Types whose extracted value is expected to appear (near-)verbatim in the source, so a
# text search is a meaningful support signal. Booleans/null/object/array are inferred or
# structural - grounding them would penalise correct inference - and are skipped.
GROUNDABLE_TYPES: frozenset[str] = frozenset({"string", "number", "integer", "enum"})

# Score ladder. A value is "grounded" when its score clears the caller's threshold
# (config.grounding_min_score, default 0.5): EXACT and WORDS/FUZZY pass, PARTIAL fails.
_SCORE_EXACT: float = 1.0  # value is a verbatim substring of the text
_SCORE_WORDS: float = 0.85  # every word token of the value is present in the text
_SCORE_FUZZY: float = 0.7  # LCS token coverage clears the fuzzy threshold
_SCORE_PARTIAL: float = 0.4  # some, but not enough, of the value is present
_SCORE_NONE: float = 0.0  # no meaningful overlap - likely hallucinated

# Tier-2 two-gate accept rule (LCS-alignment defaults). Coverage: fraction of the value's
# tokens that must align, in order. Density: matched / source-span length - rejects a
# match whose tokens are scattered across noise (high coverage but spread thin).
_FUZZY_COVERAGE_THRESHOLD: float = 0.75
_FUZZY_MIN_DENSITY: float = 1 / 3
# Words this short carry little identifying signal, so requiring them in the text would
# reject valid values over stopwords ("the", "of"). Only longer tokens gate WORDS/PARTIAL.
_MIN_SIGNIFICANT_TOKEN_LEN: int = 3
# Character n-gram width for the PARTIAL fallback on single-token values (a value with no
# significant *word* still earns PARTIAL if a 4-char slice of it occurs in the text).
_PARTIAL_NGRAM: int = 4

_WORD_RE: re.Pattern[str] = re.compile(r"\w+")

# Currency and unit aliases: a value written as a code or word usually appears in the
# document as its symbol (and the reverse), so either rendering should ground. Universal
# units only - never domain field names.
_UNIT_ALIASES: dict[str, tuple[str, ...]] = {
    "usd": ("$", "us$", "dollars", "dollar"),
    "eur": ("€", "euro", "euros"),
    "gbp": ("£", "pound", "pounds"),
    "jpy": ("¥", "yen"),
    "percent": ("%", "pct"),
    "$": ("usd", "dollars"),
    "€": ("eur", "euro"),
    "£": ("gbp", "pound"),
    "¥": ("jpy", "yen"),
    "%": ("percent", "pct"),
}


class GroundingStatus(str, Enum):
    """Support level of a value against the source, from strongest to none.

    ``SCHEMA_DERIVED`` marks a value chosen from the schema (an enum member), which is
    not quoted from the prose and so is exempt from a literal search.
    """

    EXACT = "exact"
    FUZZY = "fuzzy"
    PARTIAL = "partial"
    NONE = "none"
    SCHEMA_DERIVED = "schema_derived"


@dataclass(frozen=True)
class GroundingResult:
    """Non-destructive grounding label for one value: a status and its raw score.

    Grounding records support; it never drops the value. A caller (or a later
    verification tier) decides what to do with a weak status.
    """

    status: GroundingStatus
    score: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ground_value(value: Any, text: str, field: Field) -> GroundingResult | None:
    """Label how well *value* of *field* is supported by *text* (never drops it).

    Returns ``None`` for values that carry no meaningful text signal (``None`` value or
    a non-groundable type: boolean, null, object, array). An enum value is reported as
    ``SCHEMA_DERIVED`` (chosen from the schema, already validated against the allowed
    set), so a literal search never flags a correct choice. Otherwise the value is
    scored on the ladder and mapped to a :class:`GroundingStatus`.

    Args:
        value: The extracted value.
        text: The source text the model was shown for this value.
        field: The schema field descriptor (its ``type`` drives the decision).

    Returns:
        A :class:`GroundingResult`, or ``None`` when the value is not grounding-checked.

    Example:
        >>> from nfield.schema._types import Field
        >>> s = Field("name", "string", {}, "", {})
        >>> ground_value("Acme", "Acme Corp invoice", s).status
        <GroundingStatus.EXACT: 'exact'>
        >>> e = Field("kind", "enum", {}, "", {})
        >>> ground_value("actual", "reported figures", e).status
        <GroundingStatus.SCHEMA_DERIVED: 'schema_derived'>
    """
    if value is None or field.type not in GROUNDABLE_TYPES:
        return None
    if _is_enum_constrained(field):
        return GroundingResult(GroundingStatus.SCHEMA_DERIVED, _SCORE_EXACT)
    score = grounding_score(value, text, field.type)
    return GroundingResult(_status_from_score(score), score)


def find_span(value: Any, text: str, field: Field) -> tuple[int, int] | None:
    """Return the ``[start, end)`` char interval of *value* in *text*, or ``None``.

    Locates the value's exact rendering (including numeric comma/scale forms and
    currency/unit aliases) so a caller can map an extracted value to its source
    position. ``None`` values, non-groundable types, non-verbatim values, and
    fuzzy-only matches return ``None`` (a span is reported only when an exact rendering
    is found, so a reported offset is always trustworthy). An enum value is located when
    it appears verbatim - provenance is only ever reported when the text truly contains
    the value, so this cannot fabricate a location.

    Args:
        value: The extracted value.
        text: The source text to locate the value in (typically the full document).
        field: The schema field descriptor (its ``type`` drives candidate rendering).

    Returns:
        The half-open char interval of the first exact match, or ``None``.

    Example:
        >>> from nfield.schema._types import Field
        >>> f = Field("vendor", "string", {}, "", {})
        >>> find_span("Acme", "Issued by Acme Corp", f)
        (11, 15)
    """
    if value is None or field.type not in GROUNDABLE_TYPES:
        return None
    if not text:
        return None
    for candidate in _value_candidates(value, field.type):
        if not candidate:
            continue
        # Search the original text (case-insensitive) so the returned offsets index it
        # directly - lowercasing can change length (e.g. U+0130) and shift indices.
        match = re.search(re.escape(candidate), text, re.IGNORECASE)
        if match is not None:
            return match.span()
    return None


def is_groundable(field: Field, value: Any) -> bool:
    """Return whether *value* of *field* should be grounded against the source.

    Grounding only makes sense for values expected to appear (near-)verbatim. A
    ``None`` value (the field was absent) and non-groundable types (boolean, null,
    object, array) are skipped so a correct inference is never flagged.

    Args:
        field: The schema field descriptor (its ``type`` drives the decision).
        value: The extracted value.

    Returns:
        ``True`` if the value should be grounding-checked, ``False`` to skip it.

    Example:
        >>> from nfield.schema._types import Field
        >>> s = Field("name", "string", {}, "", {})
        >>> is_groundable(s, "Alice")
        True
        >>> b = Field("active", "boolean", {}, "", {})
        >>> is_groundable(b, True)
        False
        >>> is_groundable(s, None)
        False
    """
    if value is None:
        return False
    return field.type in GROUNDABLE_TYPES


def grounding_score(value: Any, text: str, field_type: str) -> float:
    """Score how well *value* is supported by *text*, in ``[0, 1]`` (1 = verbatim).

    Walks the ladder exact → all-words → fuzzy (coverage+density) → partial. Numeric
    values also match their formatted forms (``1234568`` against ``"1,234,568"``) so a
    figure copied verbatim from the document still grounds.

    Args:
        value: The extracted value (stringified for matching).
        text: The source text the model was shown for this value.
        field_type: JSON Schema type of the field, used to pick numeric variants.

    Returns:
        Grounding score in ``[0, 1]``: ``1.0`` exact substring, ``0.85`` all words
        present, ``0.7`` fuzzy match, ``0.4`` partial, ``0.0`` no support.

    Example:
        >>> grounding_score("Acme Corp", "Issued by Acme Corp on Friday.", "string")
        1.0
        >>> grounding_score("Acme Limited", "Acme Holdings Limited group", "string")
        0.85
        >>> grounding_score("Zeta", "nothing relevant here", "string")
        0.0
    """
    if not text:
        return _SCORE_NONE
    text_lower = text.lower()

    # Tier 1a - exact substring of any candidate rendering of the value.
    for candidate in _value_candidates(value, field_type):
        if candidate and candidate.lower() in text_lower:
            return _SCORE_EXACT

    value_tokens = _tokens(str(value))
    if not value_tokens:
        return _SCORE_NONE
    text_tokens = _tokens(text)
    text_token_set = set(text_tokens)

    # Tier 1b - every word token of the value is present somewhere in the text.
    if all(tok in text_token_set for tok in value_tokens):
        return _SCORE_WORDS

    # Tier 2 - order-preserving fuzzy match (coverage + density gates) for the
    # borderline band where not every token is present.
    if _fuzzy_accept(value_tokens, text_tokens):
        return _SCORE_FUZZY

    # Tier 1c - partial: any significant word token, or a 4-char slice, occurs in text.
    if _has_partial_support(value_tokens, str(value), text_lower, text_token_set):
        return _SCORE_PARTIAL

    return _SCORE_NONE


def is_grounded(value: Any, text: str, field_type: str, *, min_score: float) -> bool:
    """Return whether *value*'s grounding score clears *min_score*.

    Args:
        value: The extracted value.
        text: The source text the model was shown.
        field_type: JSON Schema type of the field.
        min_score: Minimum grounding score to count as grounded (keyword-only).

    Returns:
        ``True`` if :func:`grounding_score` is ``>= min_score``.

    Example:
        >>> is_grounded("Acme", "Acme Corp invoice", "string", min_score=0.5)
        True
        >>> is_grounded("Zeta", "nothing here", "string", min_score=0.5)
        False
    """
    return grounding_score(value, text, field_type) >= min_score


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _value_candidates(value: Any, field_type: str) -> list[str]:
    """Return string renderings of *value* to test for a verbatim substring match.

    For numeric fields this adds comma-grouped and integer forms so a figure the
    document writes as ``"1,234,568"`` still matches an extracted ``1234568``.

    Args:
        value: The extracted value.
        field_type: JSON Schema type of the field.

    Returns:
        Candidate strings to search for (most specific first), deduplicated.
    """
    raw = str(value)
    candidates = [raw]
    if field_type in ("number", "integer"):
        number = coerce_number(raw) if isinstance(value, str) else _as_number(value)
        # int(inf)/int(nan) raise, so a non-finite figure keeps only its raw string form.
        if number is not None and math.isfinite(number):
            integral = number == int(number)
            base = int(number) if integral else number
            candidates.append(f"{base:,}")  # 1234568 -> "1,234,568"
            candidates.append(str(base))
            if integral:
                candidates.append(str(int(number)))
    candidates.extend(_UNIT_ALIASES.get(raw.strip().casefold(), ()))
    # Order-preserving dedupe, dropping empties.
    return list(dict.fromkeys(c for c in candidates if c))


def _is_enum_constrained(field: Field) -> bool:
    """Return whether *field* is restricted to a fixed set of schema values.

    Covers both a bare enum node (``type == "enum"``) and a typed field carrying an
    ``enum`` constraint (e.g. ``{"type": "string", "enum": [...]}``), which the
    flattener types by its explicit ``type``.
    """
    return field.type == "enum" or "enum" in field.constraints


def _status_from_score(score: float) -> GroundingStatus:
    """Map a grounding score to its coarse support status."""
    if score >= _SCORE_EXACT:
        return GroundingStatus.EXACT
    if score >= _SCORE_FUZZY:  # covers all-words (0.85) and fuzzy (0.7)
        return GroundingStatus.FUZZY
    if score >= _SCORE_PARTIAL:
        return GroundingStatus.PARTIAL
    return GroundingStatus.NONE


def _as_number(value: Any) -> float | None:
    """Return *value* as a float if it is already numeric (not bool), else ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _tokens(text: str) -> list[str]:
    """Lowercase word tokens of *text* (``\\w+`` runs), used for overlap checks."""
    return _WORD_RE.findall(text.lower())


def _fuzzy_accept(value_tokens: list[str], text_tokens: list[str]) -> bool:
    """Return whether *value_tokens* fuzzily match *text*, by coverage AND density.

    Uses an LCS-style accept rule for span grounding: the order-preserving matching
    blocks of :class:`difflib.SequenceMatcher` stand in for the LCS (a standard, faster
    approximation than the full O(n·m²) DP), then **both** gates must hold:

    * **coverage** - ``matched >= ceil(len(value) * threshold)``: enough of the value
      was found;
    * **density** - ``matched / span_len >= min_density``: the matched tokens are tight
      in the source, not scattered across noise (``span_len`` spans the first to the last
      matched source token).

    Args:
        value_tokens: Word tokens of the extracted value.
        text_tokens: Word tokens of the source text.

    Returns:
        ``True`` if both gates pass.
    """
    if not value_tokens:
        return False
    matcher = SequenceMatcher(a=value_tokens, b=text_tokens, autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size > 0]
    if not blocks:
        return False
    matched = sum(block.size for block in blocks)
    needed = math.ceil(len(value_tokens) * _FUZZY_COVERAGE_THRESHOLD)
    span_start = min(block.b for block in blocks)
    span_end = max(block.b + block.size - 1 for block in blocks)
    span_len = span_end - span_start + 1
    if span_len <= 0:
        return False
    density = matched / span_len
    return matched >= needed and density >= _FUZZY_MIN_DENSITY


def _has_partial_support(
    value_tokens: list[str],
    value_str: str,
    text_lower: str,
    text_token_set: set[str],
) -> bool:
    """Return whether the value has *some* presence in the text (partial tier).

    True when a significant (length-gated) value token appears in the text, or - for a
    value with no significant word - when a short character slice of it occurs in the
    text. This separates "a fragment is there" from "nothing is there".

    Args:
        value_tokens: Word tokens of the value.
        value_str: The value rendered as a string.
        text_lower: The lowercased source text.
        text_token_set: Set of the text's word tokens.

    Returns:
        ``True`` if any partial support is found.
    """
    if any(
        len(tok) >= _MIN_SIGNIFICANT_TOKEN_LEN and tok in text_token_set for tok in value_tokens
    ):
        return True
    needle = value_str.lower()
    if len(needle) >= _PARTIAL_NGRAM:
        return any(
            needle[i : i + _PARTIAL_NGRAM] in text_lower
            for i in range(len(needle) - _PARTIAL_NGRAM + 1)
        )
    return False
