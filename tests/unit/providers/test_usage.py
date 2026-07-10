"""Per-run usage accounting: the counter, its context isolation, and the provider hook."""

from __future__ import annotations

import asyncio

from nfield.providers._usage import Usage, record_usage, start_usage


class TestUsage:
    """The counter itself."""

    def test_starts_empty(self) -> None:
        usage = Usage()
        assert (usage.prompt_tokens, usage.completion_tokens, usage.calls) == (0, 0, 0)

    def test_cost_is_priced_per_million(self) -> None:
        usage = Usage(prompt_tokens=2_000_000, completion_tokens=1_000_000)
        assert usage.cost((0.5, 1.5)) == 2 * 0.5 + 1 * 1.5

    def test_cost_of_empty_run_is_zero(self) -> None:
        assert Usage().cost((5.0, 15.0)) == 0.0


class TestRecordUsage:
    """Recording into the active run's counter."""

    def test_no_active_run_is_a_noop(self) -> None:
        # Direct provider use outside the engine must not raise or leak state.
        async def scenario() -> None:
            record_usage(10, 5)

        asyncio.run(scenario())

    def test_accumulates_across_calls(self) -> None:
        async def scenario() -> Usage:
            usage = start_usage()
            record_usage(100, 40)
            record_usage(200, 60)
            return usage

        usage = asyncio.run(scenario())
        assert usage.prompt_tokens == 300
        assert usage.completion_tokens == 100
        assert usage.calls == 2

    def test_unreported_usage_is_not_counted_as_a_call(self) -> None:
        async def scenario() -> Usage:
            usage = start_usage()
            record_usage(None, None)
            return usage

        usage = asyncio.run(scenario())
        assert usage.calls == 0

    def test_partial_report_counts_what_it_has(self) -> None:
        async def scenario() -> Usage:
            usage = start_usage()
            record_usage(100, None)
            return usage

        usage = asyncio.run(scenario())
        assert (usage.prompt_tokens, usage.completion_tokens, usage.calls) == (100, 0, 1)

    def test_concurrent_runs_keep_isolated_tallies(self) -> None:
        # Two runs in sibling tasks (the extract_batch shape) must never mix counts:
        # each task's start_usage() binds a fresh counter in its own context.
        async def one_run(prompt: int) -> Usage:
            usage = start_usage()
            await asyncio.sleep(0)  # interleave with the sibling run
            record_usage(prompt, prompt // 2)
            await asyncio.sleep(0)
            record_usage(prompt, prompt // 2)
            return usage

        async def scenario() -> tuple[Usage, Usage]:
            return await asyncio.gather(one_run(100), one_run(1000))

        a, b = asyncio.run(scenario())
        assert (a.prompt_tokens, a.completion_tokens) == (200, 100)
        assert (b.prompt_tokens, b.completion_tokens) == (2000, 1000)

    def test_leaf_tasks_inherit_the_run_counter(self) -> None:
        # Tasks created after start_usage() (the engine's leaf calls) share the tally.
        async def leaf() -> None:
            record_usage(50, 20)

        async def scenario() -> Usage:
            usage = start_usage()
            await asyncio.gather(asyncio.create_task(leaf()), asyncio.create_task(leaf()))
            return usage

        usage = asyncio.run(scenario())
        assert (usage.prompt_tokens, usage.completion_tokens, usage.calls) == (100, 40, 2)


class TestProviderHook:
    """BaseProvider._record_usage feeds calibration and the run tally together."""

    def test_sets_last_prompt_tokens_and_records(self) -> None:
        from nfield.providers._base import BaseProvider

        class _Stub(BaseProvider):
            async def _raw_complete(self, messages, *, max_tokens):  # type: ignore[no-untyped-def]
                return ""

            def _get_client(self):  # type: ignore[no-untyped-def]
                return None

            @property
            def context_window(self) -> int:
                return 8192

            @property
            def max_output_tokens(self) -> int:
                return 1024

        async def scenario() -> tuple[Usage, _Stub]:
            usage = start_usage()
            stub = _Stub("m")
            stub._record_usage(123, 45)
            return usage, stub

        usage, stub = asyncio.run(scenario())
        assert stub.last_prompt_tokens == 123  # calibration input unchanged
        assert (usage.prompt_tokens, usage.completion_tokens) == (123, 45)
