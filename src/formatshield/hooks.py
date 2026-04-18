"""FormatShield hook system — lifecycle callbacks for generation events.

Inspired by Instructor's Hooks class, this module provides a lightweight
event-driven callback mechanism that fires at key points in the generation
pipeline without coupling the core routing logic to observability concerns.

Usage::

    import formatshield as fs
    from formatshield.hooks import Hooks, HOOK_COMPLETION_RESPONSE

    hooks = Hooks()
    hooks.on(HOOK_COMPLETION_RESPONSE, lambda resp: print(f"Got: {resp[:80]}"))

    shield = fs.FormatShield(model="dryrun/test", hooks=hooks)
    result = shield.generate_sync("Hello", schema={"type": "object"})
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
from collections.abc import Callable
from typing import Any, Final

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hook name constants
# ---------------------------------------------------------------------------

HOOK_COMPLETION_KWARGS: Final[str] = "completion:kwargs"
"""Fired before the backend API call.

Handler signature: ``(kwargs: dict[str, Any]) -> None``

The handler receives the kwargs dict and may mutate it in-place to modify
what is sent to the backend (e.g. inject headers, override temperature).
"""

HOOK_COMPLETION_RESPONSE: Final[str] = "completion:response"
"""Fired after a successful backend response.

Handler signature: ``(response: str) -> None``

The handler receives the raw string output returned by the backend.
"""

HOOK_COMPLETION_ERROR: Final[str] = "completion:error"
"""Fired when the backend call raises an exception.

Handler signature: ``(error: Exception) -> None``

The handler receives the exception.  Raising from this handler is safe —
``emit()`` will catch and log it without propagating.
"""

HOOK_PARSE_ERROR: Final[str] = "parse:error"
"""Fired when schema validation or JSON parsing fails.

Handler signature: ``(error: Exception, raw_output: str) -> None``

The handler receives the validation/parse error and the raw output string
that failed validation.
"""

HOOK_REQUEST_BEFORE_ROUTE: Final[str] = "request:before_route"
"""Fired before complexity scoring and route selection.

Handler signature: ``(context: dict[str, Any]) -> None``

The context includes at least: ``prompt``, ``schema``, ``model``, and ``backend``.
Handlers may add policy metadata (e.g. ``blocked=True`` with ``reason``).
"""

HOOK_REQUEST_POLICY_CHECK: Final[str] = "request:policy_check"
"""Fired for policy enforcement checkpoints.

Handler signature: ``(context: dict[str, Any]) -> None``

The context includes ``phase`` (e.g. ``"pre_route"`` or ``"post_output"``)
plus request/result metadata for policy logging or enforcement.
"""

HOOK_ROUTING_DECISION: Final[str] = "routing:decision"
"""Fired immediately after the final routing decision is made.

Handler signature: ``(context: dict[str, Any]) -> None``

The context includes route strategy, confidence, feature score, and failure modes.
"""

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

HandlerFn = Callable[..., Any]
"""Type alias for a hook handler callable."""

# ---------------------------------------------------------------------------
# Hooks class
# ---------------------------------------------------------------------------

_KNOWN_EVENTS: frozenset[str] = frozenset(
    [
        HOOK_COMPLETION_KWARGS,
        HOOK_COMPLETION_RESPONSE,
        HOOK_COMPLETION_ERROR,
        HOOK_PARSE_ERROR,
        HOOK_REQUEST_BEFORE_ROUTE,
        HOOK_REQUEST_POLICY_CHECK,
        HOOK_ROUTING_DECISION,
    ]
)


class Hooks:
    """Thread-safe event hook registry for FormatShield lifecycle callbacks.

    Handlers are called in registration order.  Exceptions raised by
    handlers are caught and logged — they never propagate out of ``emit()``.
    Both sync and async handlers are supported.

    Example::

        hooks = Hooks()

        # Register a handler
        def log_response(response: str) -> None:
            print(f"Response length: {len(response)}")

        hooks.on("completion:response", log_response)

        # Remove a specific handler
        hooks.off("completion:response", log_response)

        # Remove all handlers for an event
        hooks.clear("completion:response")

        # Fire all handlers for an event
        hooks.emit("completion:response", "some output text")
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[HandlerFn]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def on(self, event: str, handler: HandlerFn) -> None:
        """Register a handler for *event*.

        Handlers are invoked in the order they are registered.  The same
        handler may be registered multiple times; each registration adds
        an independent invocation.

        Args:
            event: The event name (e.g. ``"completion:response"``).
                Use the ``HOOK_*`` constants defined in this module.
            handler: Callable to invoke when the event fires.
                May be a regular function or a coroutine function.

        Example::

            hooks.on(HOOK_COMPLETION_RESPONSE, lambda r: print(r))
        """
        if event not in _KNOWN_EVENTS:
            logger.debug(
                "FormatShield hooks: registering handler for unknown event %r — known events: %s",
                event,
                sorted(_KNOWN_EVENTS),
            )
        with self._lock:
            self._handlers.setdefault(event, []).append(handler)

    def off(self, event: str, handler: HandlerFn) -> None:
        """Remove the first matching registration of *handler* for *event*.

        If *handler* was registered multiple times, only the first
        occurrence is removed.  If *handler* is not registered, this
        is a no-op.

        Args:
            event: The event name.
            handler: The exact callable previously passed to :meth:`on`.

        Example::

            hooks.off(HOOK_COMPLETION_RESPONSE, my_handler)
        """
        with self._lock:
            handlers = self._handlers.get(event)
            if handlers is None:
                return
            try:
                handlers.remove(handler)
            except ValueError:
                pass  # not registered — silently ignore

    def clear(self, event: str) -> None:
        """Remove all handlers for *event*.

        Args:
            event: The event name whose handlers should be cleared.

        Example::

            hooks.clear(HOOK_COMPLETION_RESPONSE)
        """
        with self._lock:
            self._handlers.pop(event, None)

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    def emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Invoke all handlers registered for *event*.

        Handlers are called in registration order with the positional and
        keyword arguments supplied to ``emit()``.  Exceptions from any
        handler are caught and logged at WARNING level — they never
        propagate out of this method.

        Async handlers are detected via :func:`asyncio.iscoroutinefunction`.
        When called from a synchronous context (no running event loop),
        async handlers are executed via :func:`asyncio.run`.  When called
        from an async context (running loop), async handlers are scheduled
        as fire-and-forget tasks via :meth:`asyncio.ensure_future`.

        Args:
            event: The event name to fire.
            *args: Positional arguments forwarded to each handler.
            **kwargs: Keyword arguments forwarded to each handler.

        Example::

            hooks.emit(HOOK_COMPLETION_RESPONSE, output_str)
            hooks.emit(HOOK_COMPLETION_ERROR, exc)
            hooks.emit(HOOK_PARSE_ERROR, exc, raw_output)
        """
        with self._lock:
            handlers = list(self._handlers.get(event, []))

        for handler in handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    self._invoke_async(handler, *args, **kwargs)
                else:
                    handler(*args, **kwargs)
            except Exception:
                logger.warning(
                    "FormatShield hooks: unhandled exception in handler for event %r",
                    event,
                    exc_info=True,
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _invoke_async(self, handler: HandlerFn, *args: Any, **kwargs: Any) -> None:
        """Execute an async handler, adapting to the current event-loop state.

        Args:
            handler: Coroutine function to invoke.
            *args: Positional arguments forwarded to the handler.
            **kwargs: Keyword arguments forwarded to the handler.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Inside an async context — schedule as a fire-and-forget task.
            loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(handler(*args, **kwargs), loop=loop)
            )
        else:
            # No running loop — safe to use asyncio.run().
            asyncio.run(handler(*args, **kwargs))

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def handler_count(self, event: str) -> int:
        """Return the number of handlers registered for *event*.

        Args:
            event: The event name to query.

        Returns:
            Integer count of registered handlers.  Returns 0 if no
            handlers are registered for the event.
        """
        with self._lock:
            return len(self._handlers.get(event, []))

    def events(self) -> list[str]:
        """Return a list of all event names that have at least one handler.

        Returns:
            List of event name strings in insertion order.
        """
        with self._lock:
            return [e for e, h in self._handlers.items() if h]
