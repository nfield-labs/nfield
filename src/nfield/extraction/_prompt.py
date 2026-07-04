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
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nfield.extraction._papt import TemplateType, describe_field

if TYPE_CHECKING:
    from nfield.schema._types import Field

# Header that introduces injected upstream dependency values in the user message.
_DEPENDENCY_BLOCK_HEADER: str = (
    "[Resolved dependency values - use these when extracting the fields below]"
)

__all__ = [
    "build_extraction_prompt",
    "build_retry_system_message",
    "builtin_system_message",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The document is presented first and the field list last, so the fields are the most
# recent context when the model answers. The task is framed as the value being present
# (countering under-extraction), and a worked example shows the exact line format
# including how an absent field is written.
_SFEP_SYSTEM_PROMPT: str = """\
You are a structured data extraction assistant. The document is given first, then the \
list of fields to extract from it.

OUTPUT FORMAT - follow exactly:
- Output exactly one line for EVERY field listed, in the order given: field.path = value
- Every listed field's value appears in the document above. Read it out exactly as \
written. {sourcing_rule}
- For boolean fields: use true or false (lowercase)
- For integer fields: write ALL digits with no quotes, commas, or units, keeping every \
trailing zero (e.g. 42; and 2,042,137,000 USD becomes 2042137000, not 2042137)
- For number fields: write with decimals if needed (e.g. 3.14)
- For array fields of scalars: output ONE line of valid JSON - an array with every item as \
a double-quoted string (numbers may be bare), e.g. field.path = ["alpha, beta corp", "gamma"]. \
Emit EVERY item the document lists, in document order, however many there are - never stop \
early, never skip items. Each item must be that entry's COMPLETE text as written in the \
document, from the entry's start to where the next entry begins - keep every name, initial, \
and date; never shorten, summarise, or reword an item, and never output only an entry's \
label, key, or number in place of its text.
- For array fields whose items are objects (shown as "items: object {{...}}"): output the \
value as ONE line of compact, valid JSON - a single array of objects - using the exact keys \
and the nested shape shown. Emit ONE object for EVERY distinct entry the document lists \
(every row, record, period, or line item) and never merge or summarise several entries into \
one. When a key's shape is itself "object {{...}}" or "array of object {{...}}", fill it with \
the matching nested JSON object or array, not a flat value. Close every {{ with }} and every \
[ with ] in the correct order so the line is parseable JSON; use [] only when the document \
reports no such entries.
- For enum fields: use one of the exact allowed values listed in the schema
- Preserve exact string values - do not paraphrase or translate
- Do not include explanations, only output field = value lines

Example (fields a.x, a.y, and an object array people with items object \
{{name: string, roles: array of object {{title: string}}}}):
a.x = 42
a.y = NULL
people = [{{"name": "Ann", "roles": [{{"title": "CEO"}}, {{"title": "Chair"}}]}}, {{"name": "Bo", "roles": [{{"title": "CTO"}}]}}]

--- BEGIN EXTRACTION ---"""

# Anti-null sourcing rules: the value is assumed present, NULL is the checked exception.
_SOURCING_RULE_STRICT: str = (
    "Use NULL only when, after checking the document, the field is genuinely not stated."
)
_SOURCING_RULE_KNOWLEDGE: str = (
    "If the document does not state a field but you know it confidently from "
    "well-established knowledge of the subject, provide that value; use NULL only when "
    "you can neither find nor confidently infer it."
)

# Closed-book prompt: answer from knowledge, NULL when unsure (arXiv:2404.10960).
# Framed as a positive knowledge task; kept separate from the document-grounded prompt.
_CLOSED_BOOK_SYSTEM_PROMPT: str = """\
You are an expert assistant with broad, accurate factual knowledge. Provide the value of \
each field listed below for the subject described, drawing on what you reliably know.

OUTPUT FORMAT - follow exactly:
- Output exactly one line for EVERY field listed, in the order given: field.path = value
- Give a value only when you are confident it is correct. If you are not certain, write NULL \
- do not guess.
- For boolean fields: use true or false (lowercase)
- For integer fields: write ALL digits with no quotes, commas, or units (e.g. 42)
- For number fields: write with decimals if needed (e.g. 3.14)
- For array fields of scalars: use [item1, item2, item3] notation
- For array fields whose items are objects: output a JSON array of objects on that one \
line, one object per entry, e.g. field.path = [{"a": 1}, {"a": 2}]; use [] if none.
- For enum fields: use one of the exact allowed values listed in the schema
- Do not include explanations, only output field = value lines

Example (for two fields named a.x and a.y):
a.x = 42
a.y = NULL

--- BEGIN EXTRACTION ---"""

_RETRY_SYSTEM_PROMPT_TEMPLATE: str = """\
You are a structured data extraction assistant performing targeted re-extraction.

Some fields from a previous extraction pass failed validation. Re-extract ONLY
the listed fields, using the same output format as before.

OUTPUT FORMAT:
- Write one field per line using: field.path = value
{sourcing_rule}
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
    instructions: str = "",
    dependency_values: dict[str, Any] | None = None,
    knowledge_fallback: bool = False,
    closed_book: bool = False,
    field_reasons: dict[str, str] | None = None,
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
            message. See :class:`~nfield.extraction._papt.TemplateType`.
        instructions: Optional caller steering, placed at the top of the user
            message (above the fields and document) so the model follows it
            reliably; the system message stays the pure SFEP format contract.
        dependency_values: Optional ``{path: value}`` of upstream dependency
            fields resolved in earlier rounds, rendered as a labelled block
            before the field list so the model reuses them.
        knowledge_fallback: When ``True``, let the model use its own knowledge for
            fields the document does not state. Default ``False`` (strict grounding).
        field_reasons: Optional ``{path: reason}`` describing why a previous attempt
            failed for a field; each reason is appended to that field's line to guide
            re-extraction. ``None`` produces the plain field list (first-pass form).

    Returns:
        List of ``{"role": ..., "content": ...}`` dicts:
        ``[{"role": "system", ...}, {"role": "user", ...}]``.

    Raises:
        ValueError: If *fields* is empty.

    Example:
        >>> from nfield.schema._types import Field
        >>> from nfield.extraction._papt import TemplateType
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
        raise ValueError("fields must be non-empty - cannot build extraction prompt")

    # Caller instructions go in the USER message, not the system message. Chat
    # models - Llama-70B on Groq especially - follow user-turn instructions far more
    # reliably than system-prompt ones (IHEval arXiv:2502.08745; Llama-70B prompting
    # guidance puts task instructions in the user turn). The system message stays the
    # pure SFEP output contract; the caller's domain instructions frame the task at
    # the top of the user message, right above the fields and document.
    system_content = _build_system_message(
        knowledge_fallback=knowledge_fallback, closed_book=closed_book
    )
    user_content = _build_user_message(
        fields,
        document_excerpt,
        template_type,
        instructions=instructions,
        dependency_values=dependency_values,
        field_reasons=field_reasons,
        closed_book=closed_book,
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def builtin_system_message(*, knowledge_fallback: bool = False) -> str:
    """Return the built-in SFEP system message text (without any caller prefix).

    Exposed so capacity planning can charge the EXACT token cost of the format
    contract (measured from this string) instead of a fixed guess.

    Args:
        knowledge_fallback: Select the knowledge-fallback sourcing rule, which is
            longer than the strict rule - so the overhead estimate matches the
            prompt that will actually be sent.

    Returns:
        The system message string the extraction prompt will use.

    Example:
        >>> "OUTPUT FORMAT" in builtin_system_message()
        True
    """
    return _build_system_message(knowledge_fallback=knowledge_fallback)


def build_retry_system_message(
    failed_fields: list[Field],
    errors: dict[str, str],
    document_excerpt: str,
    *,
    instructions: str = "",
    knowledge_fallback: bool = False,
) -> list[dict[str, str]]:
    """Build the messages list for a surgical field retry (SFR) call.

    Constructs a targeted retry prompt that includes the specific validation
    error for each failed field, guiding the LLM toward a corrected extraction.

    Args:
        failed_fields: Fields that failed validation in the previous pass.
        errors: Mapping of ``field.path -> error_message`` for each failed field.
        document_excerpt: Same document excerpt used in the original extraction.
        instructions: Optional caller steering, prepended before the retry contract.
        knowledge_fallback: When ``True``, the retry may fall back to the model's
            own knowledge for fields the document does not state. Default ``False``.

    Returns:
        List of ``{"role": ..., "content": ...}`` dicts for the retry call.

    Example:
        >>> from nfield.schema._types import Field
        >>> f = Field("age", "integer", {}, "", {})
        >>> msgs = build_retry_system_message(
        ...     [f], {"age": "Expected integer, got 'thirty'"}, "He is thirty years old."
        ... )
        >>> len(msgs)
        2
        >>> "thirty" in msgs[1]["content"]
        True
    """
    sourcing_rule = _SOURCING_RULE_KNOWLEDGE if knowledge_fallback else _SOURCING_RULE_STRICT
    # Caller instructions go in the user turn here too (see build_extraction_prompt).
    system_content = _RETRY_SYSTEM_PROMPT_TEMPLATE.format(sourcing_rule=sourcing_rule)
    user_content = _prepend(
        instructions, _build_retry_user_message(failed_fields, errors, document_excerpt)
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Private builders
# ---------------------------------------------------------------------------


def _format_dependency_value(value: Any) -> str:
    """Render a resolved dependency value in SFEP value style.

    Args:
        value: A typed Python value already extracted for the dependency field.

    Returns:
        SFEP-style string (``NULL`` / ``true`` / ``false`` / ``[a, b]`` / text).
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_dependency_value(v) for v in value) + "]"
    return str(value)


def _format_dependency_block(dependency_values: dict[str, Any] | None) -> str:
    """Render the injected upstream-dependency block, or ``""`` if none.

    Args:
        dependency_values: ``{path: value}`` resolved upstream, or ``None``.

    Returns:
        A labelled block (header + ``  path = value`` lines), or empty string.
    """
    if not dependency_values:
        return ""
    lines = [f"  {path} = {_format_dependency_value(v)}" for path, v in dependency_values.items()]
    return _DEPENDENCY_BLOCK_HEADER + "\n" + "\n".join(lines)


def _prepend(extra: str, base: str) -> str:
    """Prepend caller context to a built-in prompt, separated by a blank line.

    Args:
        extra: Caller-supplied context (may be empty).
        base: The built-in prompt that must always be preserved.

    Returns:
        ``base`` unchanged when *extra* is empty, else ``extra`` + blank line + base.
    """
    extra = extra.strip()
    return f"{extra}\n\n{base}" if extra else base


def _build_system_message(
    *,
    knowledge_fallback: bool = False,
    closed_book: bool = False,
) -> str:
    """Build the SFEP system prompt.

    Args:
        knowledge_fallback: Select the knowledge-fallback sourcing rule instead of
            strict document grounding when ``True``.
        closed_book: Use the closed-book prompt (no document; answer from knowledge,
            abstain with NULL when unsure) when ``True``.

    Returns:
        System prompt string with SFEP format contract.
    """
    if closed_book:
        return _CLOSED_BOOK_SYSTEM_PROMPT
    sourcing_rule = _SOURCING_RULE_KNOWLEDGE if knowledge_fallback else _SOURCING_RULE_STRICT
    return _SFEP_SYSTEM_PROMPT.format(sourcing_rule=sourcing_rule)


def _build_user_message(
    fields: list[Field],
    document_excerpt: str,
    template_type: TemplateType,
    *,
    instructions: str = "",
    dependency_values: dict[str, Any] | None = None,
    field_reasons: dict[str, str] | None = None,
    closed_book: bool = False,
) -> str:
    """Build the user message: document first, field list last.

    Order: caller instructions, then the document, then any resolved dependency
    values, then the field list - so the fields are the most recent context when
    the model produces its answer.

    Args:
        fields: Fields to extract, ordered by schema depth.
        document_excerpt: Trimmed document text for this extraction call.
        template_type: Controls how much schema detail to include per field.
        instructions: Optional caller steering, placed first.
        dependency_values: Optional resolved upstream values, placed before the fields.
        field_reasons: Optional ``{path: reason}`` appended per field to guide
            re-extraction.

    Returns:
        Formatted user message string.
    """
    parts: list[str] = []
    if instructions.strip():
        parts.append(instructions.strip())
    # Closed-book has no document: omit the excerpt block and the "in the document above"
    # framing, which would otherwise tell the model the answer is in a document that does
    # not exist and degrade recall.
    if not closed_book:
        parts.append(_format_document_excerpt(document_excerpt))
    dependency_block = _format_dependency_block(dependency_values)
    if dependency_block:
        parts.append(dependency_block)
    field_lines = _format_field_list(fields, template_type, field_reasons)
    header = (
        "Fields to provide, in order:"
        if closed_book
        else "Fields to extract (every value is in the document above, in order):"
    )
    parts.append(f"{header}\n{field_lines}")
    return "\n\n".join(parts)


def _format_field_list(
    fields: list[Field],
    template_type: TemplateType,
    field_reasons: dict[str, str] | None = None,
) -> str:
    """Format the fields list block for the user message.

    Args:
        fields: Fields to describe.
        template_type: Verbosity level for each field description.
        field_reasons: Optional ``{path: reason}``; a present reason is appended to
            its field's line in square brackets to steer re-extraction.

    Returns:
        Multi-line string with one field description per line.
    """
    reasons = field_reasons or {}
    lines: list[str] = []
    for f in fields:
        line = describe_field(f, template_type)
        reason = reasons.get(f.path)
        lines.append(f"{line}  [{reason}]" if reason else line)
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
        field_lines.append(f"{f.path} ({f.type}): FAILED - {error_msg}. Please re-extract.")

    fields_block = "\n".join(field_lines)
    excerpt_block = _format_document_excerpt(document_excerpt)
    return f"Fields to re-extract:\n{fields_block}\n\n{excerpt_block}"
