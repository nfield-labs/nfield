"""Prompt construction for SFEP extraction calls.

Builds the ``messages`` list passed to ``provider.complete()``. The system
message encodes the SFEP output format contract; the user message contains
the schema field descriptions and the document excerpt.

System message structure
------------------------
1. Role statement: data extraction assistant.
2. SFEP output format rules (the "contract" the LLM must honour).
3. ``--- BEGIN EXTRACTION ---`` trigger line to anchor output parsing.

User message structure
----------------------
1. ``Fields to extract:`` section with one line per field.
2. ``Document:`` section with the trimmed document excerpt.

Post-MVP stubs
--------------
* TEP two-phase format (think block + extract block) — activated when
  the cluster type is COMPLEX and D(f) >= 0.5 for most fields.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from formatshield.extraction._papt import ClusterType, TemplateType, describe_field

if TYPE_CHECKING:
    from formatshield.schema._types import Field

__all__ = [
    "build_extraction_prompt",
    "build_retry_system_message",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SFEP_SYSTEM_PROMPT_TEMPLATE: str = """\
You are a structured data extraction assistant. Extract the specified fields \
from the provided document.

OUTPUT FORMAT — follow exactly:
- Write one field per line using: field.path = value
- Use NULL if a field is not found in the document
- Use NEEDS_REVALIDATION if you find the field but cannot determine its value confidently
- For boolean fields: use true or false (lowercase)
- For integer fields: write the number without quotes (e.g., 42)
- For number fields: write with decimals if needed (e.g., 3.14)
- For array fields: use [item1, item2, item3] notation
- For enum fields: use one of the exact allowed values listed in the schema
- Preserve exact string values — do not paraphrase or translate
- Do not include explanations, only output field = value lines

--- BEGIN EXTRACTION ---"""

_RETRY_SYSTEM_PROMPT_TEMPLATE: str = """\
You are a structured data extraction assistant performing targeted re-extraction.

Some fields from a previous extraction pass failed validation. Re-extract ONLY
the listed fields, using the same output format as before.

OUTPUT FORMAT:
- Write one field per line using: field.path = value
- Use NULL if the field is not present in the document
- Correct the specific error described for each field

--- BEGIN RE-EXTRACTION ---"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_extraction_prompt(
    fields: list[Field],
    document_excerpt: str,
    template_type: TemplateType,
    *,
    cluster_type: ClusterType = ClusterType.STANDARD,
) -> list[dict[str, str]]:
    """Build the messages list for a single SFEP extraction call.

    Returns a two-message list ready for ``provider.complete(messages, ...)``.
    The system message contains the SFEP format contract; the user message
    contains schema field descriptions (at the requested verbosity) and the
    document excerpt.

    Args:
        fields: Fields to extract in this call. Must be non-empty.
        document_excerpt: Document text trimmed to the context budget for
            this leaf. May be empty only if the document is very short.
        template_type: Controls schema description verbosity in the user
            message. See :class:`~formatshield.extraction._papt.TemplateType`.
        cluster_type: Structural classification of the field group. Used for
            cluster-specific phrasing (no-op in MVP; TEP routing in post-MVP).

    Returns:
        List of ``{"role": ..., "content": ...}`` dicts:
        ``[{"role": "system", ...}, {"role": "user", ...}]``.

    Raises:
        ValueError: If *fields* is empty.

    Example:
        >>> from formatshield.schema._types import Field
        >>> from formatshield.extraction._papt import TemplateType
        >>> f = Field("name", "string", {}, "", {"description": "Full name"})
        >>> msgs = build_extraction_prompt([f], "John Smith works here.", TemplateType.STANDARD)
        >>> msgs[0]["role"]
        'system'
        >>> msgs[1]["role"]
        'user'
        >>> "name" in msgs[1]["content"]
        True
    """
    if not fields:
        raise ValueError("fields must be non-empty — cannot build extraction prompt")

    system_content = _build_system_message(cluster_type)
    user_content = _build_user_message(fields, document_excerpt, template_type)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def build_retry_system_message(
    failed_fields: list[Field],
    errors: dict[str, str],
    document_excerpt: str,
) -> list[dict[str, str]]:
    """Build the messages list for a surgical field retry (SFR) call.

    Constructs a targeted retry prompt that includes the specific validation
    error for each failed field, guiding the LLM toward a corrected extraction.

    Args:
        failed_fields: Fields that failed validation in the previous pass.
        errors: Mapping of ``field.path -> error_message`` for each failed field.
        document_excerpt: Same document excerpt used in the original extraction.

    Returns:
        List of ``{"role": ..., "content": ...}`` dicts for the retry call.

    Example:
        >>> from formatshield.schema._types import Field
        >>> f = Field("age", "integer", {}, "", {})
        >>> msgs = build_retry_system_message(
        ...     [f], {"age": "Expected integer, got 'thirty'"}, "He is thirty years old."
        ... )
        >>> len(msgs)
        2
        >>> "thirty" in msgs[1]["content"]
        True
    """
    system_content = _RETRY_SYSTEM_PROMPT_TEMPLATE
    user_content = _build_retry_user_message(failed_fields, errors, document_excerpt)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Private builders
# ---------------------------------------------------------------------------


def _build_system_message(cluster_type: ClusterType) -> str:
    """Build the system prompt for a given cluster type.

    Args:
        cluster_type: Structural classification; reserved for post-MVP TEP routing.

    Returns:
        System prompt string with SFEP format contract.
    """
    # In MVP all cluster types use the same system prompt.
    # Post-MVP: COMPLEX cluster gets TEP two-phase instructions.
    return _SFEP_SYSTEM_PROMPT_TEMPLATE


def _build_user_message(
    fields: list[Field],
    document_excerpt: str,
    template_type: TemplateType,
) -> str:
    """Build the user message containing field descriptions and document.

    Args:
        fields: Fields to extract, ordered by schema depth.
        document_excerpt: Trimmed document text for this extraction call.
        template_type: Controls how much schema detail to include per field.

    Returns:
        Formatted user message string.
    """
    field_lines = _format_field_list(fields, template_type)
    excerpt_block = _format_document_excerpt(document_excerpt)
    return f"Fields to extract:\n{field_lines}\n\n{excerpt_block}"


def _format_field_list(fields: list[Field], template_type: TemplateType) -> str:
    """Format the fields list block for the user message.

    Args:
        fields: Fields to describe.
        template_type: Verbosity level for each field description.

    Returns:
        Multi-line string with one field description per line.
    """
    lines = [describe_field(f, template_type) for f in fields]
    return "\n".join(lines)


def _format_document_excerpt(excerpt: str) -> str:
    """Wrap the document excerpt in a labelled block.

    Args:
        excerpt: Document text for this extraction call.

    Returns:
        Formatted block starting with ``Document:``.
    """
    if not excerpt.strip():
        return "Document:\n(no document provided)"
    return f"Document:\n{excerpt}"


def _build_retry_user_message(
    failed_fields: list[Field],
    errors: dict[str, str],
    document_excerpt: str,
) -> str:
    """Build the user message for a surgical retry call.

    Includes the specific validation error for each failed field to guide
    the LLM toward the correct value.

    Args:
        failed_fields: Fields that need re-extraction.
        errors: Per-field error messages.
        document_excerpt: Document text to re-extract from.

    Returns:
        Formatted retry user message.
    """
    field_lines: list[str] = []
    for f in failed_fields:
        error_msg = errors.get(f.path, "validation failed")
        field_lines.append(f"{f.path} ({f.type}): FAILED — {error_msg}. Please re-extract.")

    fields_block = "\n".join(field_lines)
    excerpt_block = _format_document_excerpt(document_excerpt)
    return f"Fields to re-extract:\n{fields_block}\n\n{excerpt_block}"


def estimate_prompt_tokens(
    fields: list[Field],
    document_excerpt: str,
    template_type: TemplateType,
    *,
    chars_per_token: float = 4.0,
) -> int:
    """Estimate the total token count for a prompt without making an API call.

    Uses the ``chars_per_token`` ratio measured during Stage 0 calibration.
    This is a conservative estimate for capacity planning.

    Args:
        fields: Fields to include in the prompt.
        document_excerpt: Document text for this call.
        template_type: Verbosity tier.
        chars_per_token: Measured characters-per-token ratio from Stage 0.

    Returns:
        Estimated token count for the complete prompt.
    """
    messages = build_extraction_prompt(fields, document_excerpt, template_type)
    total_chars = sum(len(m["content"]) for m in messages)
    return max(1, round(total_chars / chars_per_token))


def build_schema_description_block(
    fields: list[Field],
    template_type: TemplateType,
) -> str:
    """Build only the schema description block (without document excerpt).

    Used by capacity planning to estimate the overhead token cost of the
    schema description before the full prompt is assembled.

    Args:
        fields: Fields to describe.
        template_type: Verbosity tier for descriptions.

    Returns:
        Schema description block string.

    Example:
        >>> from formatshield.schema._types import Field
        >>> from formatshield.extraction._papt import TemplateType
        >>> f = Field("x", "integer", {}, "", {})
        >>> block = build_schema_description_block([f], TemplateType.CONCISE)
        >>> "x (integer)" in block
        True
    """
    return _format_field_list(fields, template_type)
