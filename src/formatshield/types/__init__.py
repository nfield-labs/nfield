"""FormatShield type system — pre-built types and DSL for structured generation."""

from formatshield.types.dsl import (
    CFG,
    Alternatives,
    Choice,
    JsonSchema,
    KleenePlus,
    KleeneStar,
    Optional,
    QuantifyBetween,
    QuantifyExact,
    QuantifyMaximum,
    QuantifyMinimum,
    Regex,
    Sequence,
    # Concrete Term classes
    String,
    # Base
    Term,
    at_least,
    at_most,
    between,
    cfg,
    either,
    exactly,
    json_schema,
    one_or_more,
    optional,
    # Core utils
    python_types_to_terms,
    # Factory functions
    regex,
    to_regex,
    zero_or_more,
)

# ── Pre-built domain types ────────────────────────────────────────────────────

# Scalars
string = Regex(r'"[^"]*"')
integer = Regex(r"[+-]?(0|[1-9][0-9]*)")
number = Regex(r"[+-]?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-][0-9]+)?")
boolean = Choice(["true", "false"])

# Date/time (ISO 8601)
date = Regex(r"(\d{4})-(0[1-9]|1[0-2])-([0-2][0-9]|3[0-1])")
time = Regex(r"([0-1][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])")
datetime = Regex(
    r"(\d{4})-(0[1-9]|1[0-2])-([0-2][0-9]|3[0-1])"
    r"T([0-1][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])"
)

# Basic patterns
digit = Regex(r"\d")
char = Regex(r"[a-zA-Z]")
newline = Regex(r"(\r\n|\r|\n)")
whitespace = Regex(r"\s")
hex_str = Regex(r"(0x)?[a-fA-F0-9]+")

# Network / identifiers
uuid4 = Regex(
    r"[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-4[a-fA-F0-9]{3}-[89abAB][a-fA-F0-9]{3}-[a-fA-F0-9]{12}"
)
ipv4 = Regex(
    r"((25[0-5]|2[0-4][0-9]|1?[0-9]{1,2})\.){3}"
    r"(25[0-5]|2[0-4][0-9]|1?[0-9]{1,2})"
)

# Document
sentence = Regex(r"[A-Z][^.!?]*[.!?]")
paragraph = Regex(r"[A-Z][^.!?]*[.!?](\s+[A-Z][^.!?]*[.!?])*")

# Email (simplified RFC 5321 compatible)
email = Regex(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

isbn = Regex(
    r"(?:ISBN(?:-1[03])?:? )?(?=[0-9X]{10}$|(?=(?:[0-9]+[- ]){3})[- 0-9X]{13}$"
    r"|97[89][0-9]{10}$|(?=(?:[0-9]+[- ]){4})[- 0-9]{17}$)"
    r"(?:97[89][- ]?)?[0-9]{1,5}[- ]?[0-9]+[- ]?[0-9]+[- ]?[0-9X]"
)

__all__ = [
    "CFG",
    "Alternatives",
    "Choice",
    "JsonSchema",
    "KleenePlus",
    "KleeneStar",
    "Optional",
    "QuantifyBetween",
    "QuantifyExact",
    "QuantifyMaximum",
    "QuantifyMinimum",
    "Regex",
    "Sequence",
    "String",
    # DSL classes
    "Term",
    "at_least",
    "at_most",
    "between",
    "boolean",
    "cfg",
    "char",
    "date",
    "datetime",
    "digit",
    "either",
    "email",
    "exactly",
    "hex_str",
    "integer",
    "ipv4",
    "isbn",
    "json_schema",
    "newline",
    "number",
    "one_or_more",
    "optional",
    "paragraph",
    # Core utils
    "python_types_to_terms",
    # Factory functions
    "regex",
    "sentence",
    # Pre-built types
    "string",
    "time",
    "to_regex",
    "uuid4",
    "whitespace",
    "zero_or_more",
]
