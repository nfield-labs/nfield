"""Sync engine - a thin, Jupyter-safe wrapper over the async engine.

``NField`` mirrors :class:`~nfield.engine._async.AsyncNField`
but drives it synchronously. The only real work here is running an awaitable to
completion: ``asyncio.run`` when no loop is active, and a dedicated worker
thread when one already is (e.g. inside Jupyter or another async host).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

from nfield.engine._async import AsyncNField

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from nfield.config import ExtractionConfig
    from nfield.types import ExtractionResult

__all__ = ["NField", "nfield"]

_T = TypeVar("_T")


def _run_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run a coroutine to completion from synchronous code.

    Uses :func:`asyncio.run` when no event loop is running. If a loop is
    already running in this thread (notably Jupyter, which runs on an active
    loop), the coroutine is executed in a separate worker thread with its own
    loop so we never call ``asyncio.run`` re-entrantly.

    Args:
        coro: The coroutine to execute.

    Returns:
        The coroutine's result.

    Example:
        >>> async def _two() -> int:
        ...     return 2
        >>> _run_sync(_two())
        2
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop in this thread - the common, non-Jupyter case.
        return asyncio.run(coro)

    # A loop is already running here; offload to a thread with a fresh loop.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class NField:
    """Synchronous NField engine: run the full S0-S6 pipeline.

    A blocking facade over :class:`~nfield.engine._async.AsyncNField`.
    Construct once with a model (and optionally a reusable schema), then call the
    instance or :meth:`extract` for each document.

    Args:
        model: Model string ``"provider/model-name"``. If ``None``, resolved
            from ``NFIELD_MODEL`` or ``config.default_model`` at init.
        schema: Optional reusable schema (dict / Pydantic model / dataclass).
        config: Optional :class:`~nfield.config.ExtractionConfig`.
        context_window: The model's real context window in tokens (C_eff).
        max_output_tokens: The model's real output ceiling in tokens (M_O).
        api_key: Provider API key. ``None`` reads it from the environment.
        base_url: Override the provider API base URL. ``None`` uses the default.
        instructions: Optional caller steering, prepended to the SFEP prompt.

    Example:
        >>> # fs = NField("groq/llama-3.1-8b", schema=Invoice)
        >>> # result = fs("invoice text")
        >>> isinstance(NField, type)
        True
    """

    def __init__(
        self,
        model: str | None = None,
        schema: object | None = None,
        *,
        config: ExtractionConfig | None = None,
        context_window: int | None = None,
        max_output_tokens: int | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        instructions: str = "",
    ) -> None:
        self._engine = AsyncNField(
            model,
            schema,
            config=config,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            api_key=api_key,
            base_url=base_url,
            instructions=instructions,
        )

    @property
    def model(self) -> str:
        """Return the resolved model string for this engine."""
        return self._engine.model

    def extract(self, document: str, schema: object | None = None) -> ExtractionResult:
        """Run the full extraction pipeline on a single document (blocking).

        Args:
            document: The source document text.
            schema: Optional per-call schema override (dict / Pydantic / dataclass).

        Returns:
            The :class:`~nfield.types.ExtractionResult`.

        Raises:
            SchemaError: If no schema is available from the call or construction.

        Example:
            >>> # result = fs.extract("invoice text", schema=Invoice)
        """
        return _run_sync(self._engine.extract(document, schema))

    @overload
    def extract_batch(
        self,
        documents: list[str],
        schema: object | None = ...,
        *,
        max_concurrent: int | None = ...,
        return_exceptions: Literal[False] = ...,
    ) -> list[ExtractionResult]: ...

    @overload
    def extract_batch(
        self,
        documents: list[str],
        schema: object | None = ...,
        *,
        max_concurrent: int | None = ...,
        return_exceptions: Literal[True],
    ) -> list[ExtractionResult | BaseException]: ...

    def extract_batch(
        self,
        documents: list[str],
        schema: object | None = None,
        *,
        max_concurrent: int | None = None,
        return_exceptions: bool = False,
    ) -> list[ExtractionResult] | list[ExtractionResult | BaseException]:
        """Extract many documents concurrently (blocking).

        Synchronous facade over
        :meth:`~nfield.engine._async.AsyncNField.extract_batch`.

        Args:
            documents: The source documents to extract.
            schema: Optional schema override applied to every document.
            max_concurrent: Max documents in flight at once.
            return_exceptions: When ``True``, a failed document yields its exception in
                place; when ``False`` (default), the first failure is re-raised.

        Returns:
            One result per input document, in input order.

        Example:
            >>> # results = fs.extract_batch([doc1, doc2])
        """
        # Branch so each call selects a concrete Literal overload of the async method.
        if return_exceptions:
            return _run_sync(
                self._engine.extract_batch(
                    documents, schema, max_concurrent=max_concurrent, return_exceptions=True
                )
            )
        return _run_sync(
            self._engine.extract_batch(
                documents, schema, max_concurrent=max_concurrent, return_exceptions=False
            )
        )

    def __call__(self, document: str, schema: object | None = None) -> ExtractionResult:
        """Alias for :meth:`extract` so ``fs(document)`` works."""
        return self.extract(document, schema)

    def __enter__(self) -> NField:
        """Enter the context manager (returns ``self``)."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Exit the context manager. No long-lived resources to release."""
        return None


def nfield(
    document: str,
    schema: object,
    model: str | None = None,
    *,
    config: ExtractionConfig | None = None,
    context_window: int | None = None,
    max_output_tokens: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    instructions: str = "",
) -> ExtractionResult:
    """Extract N structured fields from a document (synchronous, one-shot).

    The flagship entry point. Creates a temporary :class:`NField`, runs the
    pipeline once, and returns the result. For repeated extraction on the same
    schema, construct a :class:`NField` and reuse it.

    Args:
        document: The source document text.
        schema: The target schema (dict / Pydantic model / dataclass).
        model: Model string ``"provider/model-name"``. If ``None``, resolved
            from ``NFIELD_MODEL`` or ``config.default_model``.
        config: Optional extraction configuration.
        context_window: The model's real context window in tokens (C_eff).
        max_output_tokens: The model's real output ceiling in tokens (M_O).
        api_key: Provider API key. ``None`` reads it from the environment.
        base_url: Override the provider API base URL. ``None`` uses the default.
        instructions: Optional caller steering, prepended to the SFEP prompt.

    Returns:
        The :class:`~nfield.types.ExtractionResult`.

    Raises:
        SchemaError: If no model or schema can be resolved.

    Example:
        >>> # result = nfield(doc, MySchema, "groq/llama-3.1-8b")
        >>> callable(nfield)
        True
    """
    return NField(
        model,
        schema,
        config=config,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        api_key=api_key,
        base_url=base_url,
        instructions=instructions,
    ).extract(document)
