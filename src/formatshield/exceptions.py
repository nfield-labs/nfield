"""FormatShield exception hierarchy.

All exceptions inherit from ``FormatShieldError`` so callers can catch the
entire library with a single ``except FormatShieldError`` clause while still
having the option to handle individual error kinds precisely.

Hierarchy
---------
FormatShieldError
├── SchemaError        — invalid or unsupported JSON Schema
├── ProviderError      — LLM provider call failed
├── ExtractionError    — extraction pipeline failure
├── ValidationError    — a field value failed post-extraction validation
└── AssemblyError      — final assembly / serialization failure
"""

from __future__ import annotations

__all__ = [
    "AssemblyError",
    "ExtractionError",
    "FormatShieldError",
    "ProviderError",
    "SchemaError",
    "ValidationError",
]


class FormatShieldError(Exception):
    """Base class for all FormatShield errors.

    All library exceptions inherit from this class so callers can use a
    single ``except FormatShieldError`` to catch any library error.

    Example:
        >>> try:
        ...     raise FormatShieldError("something went wrong")
        ... except FormatShieldError as exc:
        ...     print(exc)
        something went wrong
    """


class SchemaError(FormatShieldError):
    """Raised when a JSON Schema is invalid, unsupported, or missing required keys.

    Args:
        message: Human-readable description of the error.
        field: The schema field path where the error occurred, if known.
        hint: Suggested fix or explanation, if available.

    Example:
        >>> raise SchemaError(
        ...     "Missing 'type' key", field="properties.name", hint="Add type: string"
        ... )
        Traceback (most recent call last):
            ...
        formatshield.exceptions.SchemaError: Missing 'type' key [field=properties.name] hint: Add type: string
    """

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        hint: str | None = None,
    ) -> None:
        self.field = field
        self.hint = hint
        super().__init__(message)

    def __str__(self) -> str:
        base = super().__str__()
        parts = [base]
        if self.field is not None:
            parts.append(f"[field={self.field}]")
        if self.hint is not None:
            parts.append(f"hint: {self.hint}")
        return " ".join(parts)


class ProviderError(FormatShieldError):
    """Raised when an LLM provider request fails.

    Args:
        message: Human-readable description of the error.
        status_code: HTTP status code returned by the provider, if applicable.

    Example:
        >>> raise ProviderError("Rate limit exceeded", status_code=429)
        Traceback (most recent call last):
            ...
        formatshield.exceptions.ProviderError: Rate limit exceeded
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        self.status_code = status_code
        super().__init__(message)

    @property
    def retryable(self) -> bool:
        """Check if this error is retryable.

        Classifies errors as transient (retryable) or permanent (non-retryable).

        Retryable errors:
        - 429 (rate limit) — temporary resource exhaustion
        - 5xx (server errors) — temporary server-side failures

        Non-retryable errors:
        - 4xx (client errors) — permanent request/auth failures (except 429)
        - None (unknown status) — CONSERVATIVE ASSUMPTION: treats unknown errors
          as non-retryable to avoid retry loops on unexpected failures. Note:
          network timeouts may return status_code=None; consider increasing
          timeout or implementing custom retry logic for timeout-sensitive use cases.

        Returns:
            True if the error is transient and should be retried.
        """
        if self.status_code is None:
            # Unknown status: conservative assumption is non-retryable
            # This prevents retry loops on unexpected failures, but may miss
            # transient errors like timeouts. Override in subclasses if needed.
            return False
        # 429 = rate limit (retryable), 5xx = server error (retryable)
        return self.status_code == 429 or 500 <= self.status_code < 600


class ExtractionError(FormatShieldError):
    """Raised when the extraction pipeline fails for a field or overall.

    Args:
        message: Human-readable description of the error.
        field: The field path being extracted when the error occurred, if known.
        attempt: The retry attempt number (1-based) when the error occurred, if known.

    Example:
        >>> raise ExtractionError("Parse failed", field="invoice.total", attempt=2)
        Traceback (most recent call last):
            ...
        formatshield.exceptions.ExtractionError: Parse failed [field=invoice.total, attempt=2]
    """

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        attempt: int | None = None,
    ) -> None:
        self.field = field
        self.attempt = attempt
        super().__init__(message)

    def __str__(self) -> str:
        base = super().__str__()
        details: list[str] = []
        if self.field is not None:
            details.append(f"field={self.field}")
        if self.attempt is not None:
            details.append(f"attempt={self.attempt}")
        if details:
            return f"{base} [{', '.join(details)}]"
        return base


class ValidationError(FormatShieldError):
    """Raised when a field value fails post-extraction validation.

    Args:
        message: Human-readable description of the validation failure.
        field: The field path that failed validation, if known.
        value: The actual value that failed validation.
        hint: Suggested fix or constraint description, if available.

    Example:
        >>> raise ValidationError(
        ...     "Expected positive number", field="total", value=-5.0, hint="Must be > 0"
        ... )
        Traceback (most recent call last):
            ...
        formatshield.exceptions.ValidationError: Expected positive number [field=total, value=-5.0] hint: Must be > 0
    """

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        value: object = None,
        hint: str | None = None,
    ) -> None:
        self.field = field
        self.value = value
        self.hint = hint
        super().__init__(message)

    def __str__(self) -> str:
        base = super().__str__()
        details: list[str] = []
        if self.field is not None:
            details.append(f"field={self.field}")
        if self.value is not None:
            details.append(f"value={self.value!r}")
        parts = [base]
        if details:
            parts.append(f"[{', '.join(details)}]")
        if self.hint is not None:
            parts.append(f"hint: {self.hint}")
        return " ".join(parts)


class AssemblyError(FormatShieldError):
    """Raised when final output assembly or serialization fails.

    Args:
        message: Human-readable description of the assembly failure.
        path: The output path or key where assembly failed, if known.

    Example:
        >>> raise AssemblyError("Cannot serialize circular reference", path="result.data.nested")
        Traceback (most recent call last):
            ...
        formatshield.exceptions.AssemblyError: Cannot serialize circular reference
    """

    def __init__(
        self,
        message: str,
        *,
        path: str | None = None,
    ) -> None:
        self.path = path
        super().__init__(message)
