"""
RCOCR Backend Protocol.

Any async callable that matches :class:`RCOCRBackend` can be used as the
inference engine for RCOCR two-pass generation.  No base class inheritance
is required — structural subtyping (duck typing) is sufficient.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class RCOCRBackend(Protocol):
    """Minimal interface an LLM backend must implement to work with RCOCR.

    The protocol is intentionally narrow: RCOCR only needs to call the model
    twice (Pass 1 unconstrained, Pass 2 constrained).  Advanced features such
    as streaming, logit biasing, and KV-cache reuse are optional extensions.

    Example — minimal in-process backend::

        class MyBackend:
            name = "my-backend"

            async def generate(
                self,
                prompt: str,
                constraints: str | None = None,
                **kwargs,
            ) -> str:
                # Call your LLM here
                return response_text
    """

    #: Short identifier used in log messages and result records.
    name: str

    async def generate(
        self,
        prompt: str,
        constraints: str | None = None,
        **kwargs: object,
    ) -> str:
        """Generate a response for *prompt* and return the full text.

        Parameters
        ----------
        prompt:
            The prompt to send to the model.
        constraints:
            Optional constraint hint.  The special value ``"json"`` requests
            JSON-only output.  Backends may ignore unknown values.
        **kwargs:
            Optional generation parameters (``temperature``, ``max_tokens``,
            ``seed``, etc.).  Backends silently ignore unknown kwargs.

        Returns
        -------
        str
            The model's generated text.
        """
        ...


@runtime_checkable
class StreamingRCOCRBackend(RCOCRBackend, Protocol):
    """Optional extension of :class:`RCOCRBackend` that supports token streaming.

    Implement this if you want to use :meth:`~rcocr.engine.RCOCREngine.stream`.
    """

    async def stream(
        self,
        prompt: str,
        constraints: str | None = None,
        **kwargs: object,
    ) -> AsyncIterator[str]:
        """Yield incremental token strings, then the final full text.

        The last yielded value should be the complete response text.

        Yields
        ------
        str
            Incremental token chunk.
        """
        ...
