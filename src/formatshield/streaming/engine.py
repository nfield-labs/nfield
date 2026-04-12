"""Streaming engine for FormatShield — assembles, filters, and formats StreamEvents."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from formatshield.scorer.features import StreamEvent


class StreamingEngine:
    """
    High-level controller for FormatShield's streaming pipeline.

    :class:`StreamingEngine` wraps an async generator of
    :class:`~formatshield.scorer.features.StreamEvent` objects produced by a
    backend and provides:

    * **Filtering** — optionally suppress ``"thinking"`` events.
    * **Collection** — drain the stream and return ``(thinking_text, output_text)``.
    * **SSE formatting** — convert individual events to Server-Sent Event strings.
    * **Static construction** — create a minimal stream from a plain string (useful
      for testing or when wrapping a non-streaming backend).

    Parameters
    ----------
    expose_thinking:
        When ``False`` (the default), :meth:`stream` drops ``"thinking"``
        events.  Set to ``True`` to expose reasoning tokens to callers.
    """

    def __init__(self, expose_thinking: bool = False) -> None:
        self.expose_thinking = expose_thinking

    # ------------------------------------------------------------------
    # Core streaming methods
    # ------------------------------------------------------------------

    async def stream(
        self,
        generator: AsyncIterator[StreamEvent],
    ) -> AsyncIterator[StreamEvent]:
        """
        Pass through events from *generator*, filtering thinking events when
        :attr:`expose_thinking` is ``False``.

        Parameters
        ----------
        generator:
            Async iterator of :class:`StreamEvent` objects, typically produced
            by a backend's ``stream()`` method.

        Yields
        ------
        StreamEvent
            Output and complete events (and optionally thinking events).
        """
        return self._stream_filtered(generator, expose_thinking=self.expose_thinking)

    async def _stream_filtered(
        self,
        generator: AsyncIterator[StreamEvent],
        expose_thinking: bool,
    ) -> AsyncIterator[StreamEvent]:
        async for event in generator:
            if event.type == "thinking" and not expose_thinking:
                continue
            yield event

    async def stream_with_thinking(
        self,
        generator: AsyncIterator[StreamEvent],
    ) -> AsyncIterator[StreamEvent]:
        """
        Pass through **all** events from *generator*, including thinking events,
        regardless of the :attr:`expose_thinking` setting.

        Parameters
        ----------
        generator:
            Async iterator of :class:`StreamEvent` objects.

        Yields
        ------
        StreamEvent
            All events including thinking events.
        """
        return self._stream_all(generator)

    async def _stream_all(
        self,
        generator: AsyncIterator[StreamEvent],
    ) -> AsyncIterator[StreamEvent]:
        async for event in generator:
            yield event

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    async def collect(
        self,
        generator: AsyncIterator[StreamEvent],
    ) -> tuple[str, str]:
        """
        Drain *generator* completely and return the aggregated text.

        Thinking text is gathered by concatenating the ``content`` field of
        all ``"thinking"`` events.  Output text is gathered by concatenating
        the ``token`` field of all ``"output"`` events *plus* the ``content``
        field of the final ``"complete"`` event (if the backend emits only a
        single complete event rather than individual token events).

        When both per-token ``"output"`` events and a ``"complete"`` event are
        present, the ``"complete"`` event's ``content`` is used as the
        authoritative output text to avoid double-counting.

        Parameters
        ----------
        generator:
            Async iterator of :class:`StreamEvent` objects.

        Returns
        -------
        tuple[str, str]
            ``(thinking_text, output_text)`` — both strings may be empty if
            the stream produced no events of that type.
        """
        thinking_parts: list[str] = []
        output_parts: list[str] = []
        complete_content: str | None = None

        async for event in generator:
            if event.type == "thinking" and event.content:
                thinking_parts.append(event.content)
            elif event.type == "output" and event.token:
                output_parts.append(event.token)
            elif event.type == "complete":
                if event.content is not None:
                    complete_content = event.content

        thinking_text = "".join(thinking_parts)

        # Prefer the complete event's content when available because it is the
        # authoritative, fully-assembled string from the backend.
        if complete_content is not None:
            output_text = complete_content
        else:
            output_text = "".join(output_parts)

        return thinking_text, output_text

    # ------------------------------------------------------------------
    # SSE formatting
    # ------------------------------------------------------------------

    @staticmethod
    def to_sse(event: StreamEvent) -> str:
        """
        Format a :class:`StreamEvent` as a Server-Sent Event (SSE) string.

        The payload is serialised as a compact JSON object on a single
        ``data:`` line, terminated by the double newline required by the SSE
        specification.

        Parameters
        ----------
        event:
            The stream event to serialise.

        Returns
        -------
        str
            An SSE-formatted string ready to be written to an HTTP response,
            e.g. ``"data: {\"type\": \"output\", \"token\": \"Hello\"}\\n\\n"``.

        Examples
        --------
        >>> engine = StreamingEngine()
        >>> ev = StreamEvent(type="output", token="Hello", backend="groq")
        >>> StreamingEngine.to_sse(ev)
        'data: {"type": "output", "content": null, "token": "Hello", ...}\\n\\n'
        """
        payload = json.dumps(event.__dict__, ensure_ascii=False)
        return f"data: {payload}\n\n"

    # ------------------------------------------------------------------
    # Static construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def from_text(
        text: str,
        backend: str = "",
    ) -> AsyncIterator[StreamEvent]:
        """
        Create a minimal async stream from a plain text string.

        This helper is useful for wrapping non-streaming backends in the
        streaming interface, or for constructing test fixtures without
        spinning up a real inference server.

        The generator yields one ``"output"`` event per word followed by a
        single ``"complete"`` event containing the full *text*.

        Parameters
        ----------
        text:
            The text to emit as a stream.
        backend:
            Optional backend identifier to embed in each event.

        Yields
        ------
        StreamEvent
            Word-level output events followed by a complete event.

        Examples
        --------
        >>> import asyncio
        >>> async def run():
        ...     engine = StreamingEngine()
        ...     _, output = await engine.collect(
        ...         await StreamingEngine.from_text("hello world", backend="test")
        ...     )
        ...     print(output)
        >>> asyncio.run(run())
        hello world
        """
        return StreamingEngine._text_generator(text, backend)

    @staticmethod
    async def _text_generator(
        text: str,
        backend: str,
    ) -> AsyncIterator[StreamEvent]:
        words = text.split(" ")
        for i, word in enumerate(words):
            # Re-add the space that split() removed, except after the last word.
            token = word if i == len(words) - 1 else word + " "
            yield StreamEvent(type="output", token=token, backend=backend)

        yield StreamEvent(type="complete", content=text, backend=backend)
