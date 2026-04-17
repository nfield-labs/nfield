"""Unit tests for the FormatShield hook system (GROUP F — Stage 4)."""

from __future__ import annotations

from typing import Any

from formatshield.hooks import (
    HOOK_COMPLETION_ERROR,
    HOOK_COMPLETION_KWARGS,
    HOOK_COMPLETION_RESPONSE,
    HOOK_PARSE_ERROR,
    Hooks,
)


def _noop(_: Any) -> None:
    """No-op handler used in registration tests."""


class TestHooksRegistration:
    def test_on_registers_handler(self) -> None:
        hooks = Hooks()
        hooks.on(HOOK_COMPLETION_RESPONSE, _noop)
        assert hooks.handler_count(HOOK_COMPLETION_RESPONSE) == 1

    def test_on_same_handler_twice(self) -> None:
        hooks = Hooks()
        hooks.on(HOOK_COMPLETION_RESPONSE, _noop)
        hooks.on(HOOK_COMPLETION_RESPONSE, _noop)
        assert hooks.handler_count(HOOK_COMPLETION_RESPONSE) == 2

    def test_off_removes_first_occurrence(self) -> None:
        hooks = Hooks()
        hooks.on(HOOK_COMPLETION_RESPONSE, _noop)
        hooks.on(HOOK_COMPLETION_RESPONSE, _noop)
        hooks.off(HOOK_COMPLETION_RESPONSE, _noop)
        assert hooks.handler_count(HOOK_COMPLETION_RESPONSE) == 1

    def test_off_no_op_when_not_registered(self) -> None:
        hooks = Hooks()
        hooks.off(HOOK_COMPLETION_RESPONSE, _noop)  # should not raise
        assert hooks.handler_count(HOOK_COMPLETION_RESPONSE) == 0

    def test_clear_removes_all_handlers(self) -> None:
        hooks = Hooks()
        for _ in range(3):
            hooks.on(HOOK_COMPLETION_RESPONSE, _noop)
        hooks.clear(HOOK_COMPLETION_RESPONSE)
        assert hooks.handler_count(HOOK_COMPLETION_RESPONSE) == 0

    def test_clear_no_op_for_unknown_event(self) -> None:
        hooks = Hooks()
        hooks.clear("nonexistent:event")  # should not raise

    def test_events_returns_only_events_with_handlers(self) -> None:
        hooks = Hooks()
        hooks.on(HOOK_COMPLETION_RESPONSE, _noop)
        events = hooks.events()
        assert HOOK_COMPLETION_RESPONSE in events
        assert HOOK_COMPLETION_KWARGS not in events


class TestHooksEmit:
    def test_emit_calls_handler_with_args(self) -> None:
        received: list[str] = []

        def capture(r: str) -> None:
            received.append(r)

        hooks = Hooks()
        hooks.on(HOOK_COMPLETION_RESPONSE, capture)
        hooks.emit(HOOK_COMPLETION_RESPONSE, "hello world")
        assert received == ["hello world"]

    def test_emit_calls_handlers_in_order(self) -> None:
        order: list[int] = []
        hooks = Hooks()
        hooks.on(HOOK_COMPLETION_RESPONSE, lambda _: order.append(1))
        hooks.on(HOOK_COMPLETION_RESPONSE, lambda _: order.append(2))
        hooks.on(HOOK_COMPLETION_RESPONSE, lambda _: order.append(3))
        hooks.emit(HOOK_COMPLETION_RESPONSE, "x")
        assert order == [1, 2, 3]

    def test_emit_does_not_raise_on_handler_exception(self) -> None:
        def bad_handler(_: Any) -> None:
            raise RuntimeError("handler exploded")

        hooks = Hooks()
        hooks.on(HOOK_COMPLETION_RESPONSE, bad_handler)
        # Should not raise — exceptions are logged, not propagated
        hooks.emit(HOOK_COMPLETION_RESPONSE, "test")

    def test_emit_continues_after_failing_handler(self) -> None:
        results: list[int] = []

        def bad_handler(_: Any) -> None:
            raise RuntimeError("boom")

        hooks = Hooks()
        hooks.on(HOOK_COMPLETION_RESPONSE, bad_handler)
        hooks.on(HOOK_COMPLETION_RESPONSE, lambda _: results.append(1))
        hooks.emit(HOOK_COMPLETION_RESPONSE, "test")
        assert results == [1]

    def test_emit_no_handlers_is_no_op(self) -> None:
        hooks = Hooks()
        hooks.emit(HOOK_COMPLETION_RESPONSE, "test")  # should not raise

    def test_emit_with_multiple_args(self) -> None:
        received: list[tuple[str, str]] = []

        def capture(e: BaseException, raw: str) -> None:
            received.append((str(e), raw))

        hooks = Hooks()
        hooks.on(HOOK_PARSE_ERROR, capture)
        exc = ValueError("bad json")
        hooks.emit(HOOK_PARSE_ERROR, exc, "raw output text")
        assert received == [("bad json", "raw output text")]

    def test_emit_error_hook_receives_exception(self) -> None:
        errors: list[BaseException] = []

        def capture_error(e: BaseException) -> None:
            errors.append(e)

        hooks = Hooks()
        hooks.on(HOOK_COMPLETION_ERROR, capture_error)
        exc = RuntimeError("api error")
        hooks.emit(HOOK_COMPLETION_ERROR, exc)
        assert errors == [exc]


class TestHooksAsync:
    def test_async_handler_scheduled(self) -> None:
        results: list[str] = []

        async def async_handler(r: str) -> None:
            results.append(r)

        hooks = Hooks()
        hooks.on(HOOK_COMPLETION_RESPONSE, async_handler)

        # Running outside an event loop — asyncio.run should be called internally
        hooks.emit(HOOK_COMPLETION_RESPONSE, "async test")
        assert results == ["async test"]


class TestHooksWithFormatShield:
    def test_hooks_fire_on_generate(self) -> None:
        from formatshield.backends.dryrun_backend import DryRunBackend
        from formatshield.core import FormatShield

        responses: list[str] = []

        def capture(r: str) -> None:
            responses.append(r)

        hooks = Hooks()
        hooks.on(HOOK_COMPLETION_RESPONSE, capture)

        shield = FormatShield(model="dryrun/test", backend=DryRunBackend(), hooks=hooks)
        shield.generate_sync("Hello world")
        assert len(responses) >= 1
        assert all(isinstance(r, str) for r in responses)

    def test_kwargs_hook_fires_before_backend_call(self) -> None:
        from formatshield.backends.dryrun_backend import DryRunBackend
        from formatshield.core import FormatShield

        kwargs_seen: list[dict[str, Any]] = []

        def capture_kw(kw: dict[str, Any]) -> None:
            kwargs_seen.append(dict(kw))

        hooks = Hooks()
        hooks.on(HOOK_COMPLETION_KWARGS, capture_kw)

        shield = FormatShield(model="dryrun/test", backend=DryRunBackend(), hooks=hooks)
        shield.generate_sync("Hello")
        assert len(kwargs_seen) >= 1
        assert "schema" in kwargs_seen[0]
