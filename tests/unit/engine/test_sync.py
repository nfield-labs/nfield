"""Tests for the synchronous engine and its Jupyter-safe loop handling."""

from __future__ import annotations

import asyncio

from nfield import NField, nfield
from nfield.config import ExtractionConfig
from nfield.engine._sync import _run_sync

_DOC = "Name: Alice. Age: 30."
_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
}
_ECHO = "name = Alice\nage = 30"


class TestSyncEngine:
    def test_nfield_returns_result(self, install_provider):
        install_provider(_ECHO)
        result = nfield(_DOC, _SCHEMA, "mock/echo")
        assert result.data["name"] == "Alice"

    def test_call_alias_and_context_manager(self, install_provider):
        install_provider(_ECHO)
        with NField("mock/echo", _SCHEMA) as fs:
            result = fs(_DOC)
        assert result.data["age"] == 30


class TestRunSync:
    def test_no_running_loop_uses_asyncio_run(self):
        async def _v() -> int:
            return 7

        assert _run_sync(_v()) == 7

    async def test_called_from_within_running_loop(self, install_provider):
        # Simulates Jupyter: a loop is already running on this thread. A direct
        # blocking call must offload to a worker thread (its own loop) instead
        # of re-entering asyncio.run on the active loop.
        install_provider(_ECHO)
        assert asyncio.get_running_loop().is_running()
        result = NField("mock/echo", _SCHEMA, config=ExtractionConfig(max_retry_rounds=0)).extract(
            _DOC
        )
        assert result.data["name"] == "Alice"
