"""NField exception hierarchy.

All exceptions inherit from ``NFieldError`` so callers can catch the
entire library with a single ``except NFieldError`` clause while still
having the option to handle individual error kinds precisely.

Hierarchy
---------
NFieldError
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
    "NFieldError",
    "ProviderError",
    "SchemaError",
    "ValidationError",
]


class NFieldError(Exception):
    """Base class for all NField errors.

    All library exceptions inherit from this class so callers can use a
    single ``except NFieldError`` to catch any library error.

    Example:
        >>> try:
        ...     raise NFieldError("something went wrong")
        ... except NFieldError as exc:
        ...     print(exc)
        something went wrong
    """


class SchemaError(NFieldError):
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
        nfield.exceptions.SchemaError: Missing 'type' key [field=properties.name] hint: Add type: string
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


# HTTP status codes worth retrying (REST guidance): request timeout, rate limit,
# and all server errors (5xx handled by range).
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 429})


class ProviderError(NFieldError):
    """Raised when an LLM provider request fails.

    Args:
        message: Human-readable description of the error.
        status_code: HTTP status code returned by the provider, if applicable.
        retryable: Explicit transient/permanent override. Providers set this for
            errors that carry no HTTP status — chiefly timeouts and connection
            resets, which are transient but have ``status_code=None``. Left as
            ``None`` (no override), retryability is inferred from ``status_code``.
        retry_after: Seconds the server asked the caller to wait (the ``Retry-After``
            header), if provided. Honoured by the backoff loop.

    Example:
        >>> raise ProviderError("Rate limit exceeded", status_code=429)
        Traceback (most recent call last):
            ...
        nfield.exceptions.ProviderError: Rate limit exceeded
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool | None = None,
        retry_after: float | None = None,
    ) -> None:
        self.status_code = status_code
        self.retry_after = retry_after
        self._retryable_override = retryable
        super().__init__(message)

    @property
    def retryable(self) -> bool:
        """Whether this error is transient and should be retried.

        An explicit ``retryable`` override wins (providers use it to mark timeouts
        and connection errors, which carry no status code, as transient). Otherwise
        the HTTP status decides: ``408``/``429`` and any ``5xx`` are retryable; a
        permanent ``4xx`` is not; an unknown status (``None``) is treated
        conservatively as non-retryable so an unexpected bug is not retried blindly.

        Returns:
            True if the error is transient and should be retried.
        """
        if self._retryable_override is not None:
            return self._retryable_override
        if self.status_code is None:
            return False
        return self.status_code in _RETRYABLE_STATUS_CODES or 500 <= self.status_code < 600


class ExtractionError(NFieldError):
    """Raised when the extraction pipeline fails for a field or overall.

    Args:
        message: Human-readable description of the error.
        field: The field path being extracted when the error occurred, if known.
        attempt: The retry attempt number (1-based) when the error occurred, if known.

    Example:
        >>> raise ExtractionError("Parse failed", field="invoice.total", attempt=2)
        Traceback (most recent call last):
            ...
        nfield.exceptions.ExtractionError: Parse failed [field=invoice.total, attempt=2]
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


class ValidationError(NFieldError):
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
        nfield.exceptions.ValidationError: Expected positive number [field=total, value=-5.0] hint: Must be > 0
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


class AssemblyError(NFieldError):
    """Raised when final output assembly or serialization fails.

    Args:
        message: Human-readable description of the assembly failure.
        path: The output path or key where assembly failed, if known.

    Example:
        >>> raise AssemblyError("Cannot serialize circular reference", path="result.data.nested")
        Traceback (most recent call last):
            ...
        nfield.exceptions.AssemblyError: Cannot serialize circular reference [path=result.data.nested]
    """

    def __init__(
        self,
        message: str,
        *,
        path: str | None = None,
    ) -> None:
        self.path = path
        super().__init__(message)

    def __str__(self) -> str:
        base = super().__str__()
        if self.path is not None:
            return f"{base} [path={self.path}]"
        return base
