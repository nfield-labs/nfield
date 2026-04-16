"""
FormatShieldGenerator — reusable generator objects with schema state cached at construction.

These classes mirror the ``BlackBoxGenerator`` / ``AsyncBlackBoxGenerator`` pattern from
Outlines but sit on top of FormatShield's routing layer instead of directly wiring a
constrained-decoding backend.

Key advantage: schema features (depth, constraint count) are scored once at construction.
Only prompt features recompute on each call, making repeated calls faster.

Example::

    import formatshield as fs

    shield = fs.FormatShield(model="dryrun/test")
    gen = shield.generator(schema={"type": "object", "properties": {"answer": {"type": "string"}}})

    result = gen("What is 2+2?")
    results = gen.batch(["What is 2+2?", "What is 3+3?"])

    async_gen = shield.async_generator(output_type=int)
    result = await async_gen("What is 2+2?")
    results = await async_gen.batch(["What is 2+2?", "What is 3+3?"], max_concurrency=5)
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from formatshield.core import FormatShield, GenerationResult
    from formatshield.scorer.features import StreamEvent


class FormatShieldGenerator:
    """Reusable synchronous generator — schema features scored once at construction.

    Wraps a :class:`~formatshield.core.FormatShield` instance and caches the
    schema so repeated calls to the same output structure skip re-analysis.

    Args:
        shield: A configured :class:`~formatshield.core.FormatShield` instance.
        schema: JSON Schema dict or Pydantic model class for structured output.
        output_type: Python type to cast the output to (int, Enum, Literal, list[T]).

    Example::

        gen = FormatShieldGenerator(shield, output_type=int)
        result = gen("What is 2+2?")
        assert isinstance(result.parsed, int)
    """

    def __init__(
        self,
        shield: FormatShield,
        schema: Any | None = None,
        output_type: type[Any] | None = None,
    ) -> None:
        self._shield = shield
        self._schema = schema
        self._output_type = output_type

    def __call__(self, prompt: str, **kwargs: Any) -> GenerationResult:
        """Generate a result for *prompt* using the cached schema/output_type.

        Args:
            prompt: The user prompt.
            **kwargs: Additional keyword arguments forwarded to
                :meth:`~formatshield.core.FormatShield.generate_sync`.

        Returns:
            :class:`~formatshield.core.GenerationResult` for the request.
        """
        return self._shield.generate_sync(
            prompt,
            schema=self._schema,
            output_type=self._output_type,
            **kwargs,
        )

    def batch(self, prompts: list[str], **kwargs: Any) -> list[GenerationResult]:
        """Generate results for a list of prompts sequentially.

        Args:
            prompts: List of user prompts.
            **kwargs: Keyword arguments forwarded to each :meth:`__call__`.

        Returns:
            List of :class:`~formatshield.core.GenerationResult` in input order.
        """
        return [self(prompt, **kwargs) for prompt in prompts]

    def stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        """Stream events for *prompt* using the cached schema.

        Args:
            prompt: The user prompt.
            **kwargs: Keyword arguments forwarded to
                :meth:`~formatshield.core.FormatShield.stream`.

        Returns:
            Async iterator of :class:`~formatshield.scorer.features.StreamEvent`.
        """
        return self._shield.stream(prompt, schema=self._schema, **kwargs)

    @property
    def schema(self) -> Any | None:
        """The schema this generator was constructed with."""
        return self._schema

    @property
    def output_type(self) -> type[Any] | None:
        """The output type this generator was constructed with."""
        return self._output_type


class AsyncFormatShieldGenerator:
    """Reusable async generator — supports parallel batch with asyncio.gather.

    Wraps a :class:`~formatshield.core.FormatShield` instance for async usage.
    Batch calls run concurrently up to *max_concurrency* to respect rate limits.

    Args:
        shield: A configured :class:`~formatshield.core.FormatShield` instance.
        schema: JSON Schema dict or Pydantic model class for structured output.
        output_type: Python type to cast the output to (int, Enum, Literal, list[T]).

    Example::

        async_gen = AsyncFormatShieldGenerator(shield, output_type=int)
        result = await async_gen("What is 2+2?")
        results = await async_gen.batch(["What is 2+2?", "What is 3+3?"], max_concurrency=5)
    """

    # Default concurrency cap — conservative for free-tier API limits
    DEFAULT_MAX_CONCURRENCY: int = 10

    def __init__(
        self,
        shield: FormatShield,
        schema: Any | None = None,
        output_type: type[Any] | None = None,
    ) -> None:
        self._shield = shield
        self._schema = schema
        self._output_type = output_type

    async def __call__(self, prompt: str, **kwargs: Any) -> GenerationResult:
        """Generate a result for *prompt* asynchronously.

        Args:
            prompt: The user prompt.
            **kwargs: Keyword arguments forwarded to
                :meth:`~formatshield.core.FormatShield.generate`.

        Returns:
            :class:`~formatshield.core.GenerationResult` for the request.
        """
        return await self._shield.generate(
            prompt,
            schema=self._schema,
            output_type=self._output_type,
            **kwargs,
        )

    async def batch(
        self,
        prompts: list[str],
        max_concurrency: int | None = None,
        **kwargs: Any,
    ) -> list[GenerationResult]:
        """Generate results for a list of prompts concurrently.

        Uses :func:`asyncio.gather` for parallel execution, bounded by
        *max_concurrency* to respect backend rate limits.

        Args:
            prompts: List of user prompts.
            max_concurrency: Maximum simultaneous requests.  Defaults to
                :attr:`DEFAULT_MAX_CONCURRENCY`.  Set to ``None`` for no limit
                (use with caution on rate-limited backends).
            **kwargs: Keyword arguments forwarded to each :meth:`__call__`.

        Returns:
            List of :class:`~formatshield.core.GenerationResult` in input order.
        """
        if max_concurrency is None:
            return list(await asyncio.gather(*[self(p, **kwargs) for p in prompts]))

        sem = asyncio.Semaphore(max_concurrency)

        async def _bounded(p: str) -> GenerationResult:
            async with sem:
                return await self(p, **kwargs)

        return list(await asyncio.gather(*[_bounded(p) for p in prompts]))

    @property
    def schema(self) -> Any | None:
        """The schema this generator was constructed with."""
        return self._schema

    @property
    def output_type(self) -> type[Any] | None:
        """The output type this generator was constructed with."""
        return self._output_type
