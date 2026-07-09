"""Tests for extract_batch on the async and sync engines."""

from __future__ import annotations

import asyncio

import pytest

from nfield import AsyncNField, NField
from nfield.types import ExtractionStatus

_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
}
_ECHO = "name = Alice\nage = 30"
_DOCS = ["Name: Alice. Age: 30.", "Name: Bob. Age: 41.", "Name: Cy. Age: 5."]


class TestAsyncBatch:
    async def test_returns_one_result_per_document_in_order(self, install_provider) -> None:
        provider = install_provider(_ECHO)
        engine = AsyncNField("mock/echo", _SCHEMA)
        results = await engine.extract_batch(_DOCS)
        assert len(results) == len(_DOCS)
        assert all(r.data["name"] == "Alice" for r in results)
        # One extraction call per document (single-leaf schema). Stage 0 spends no
        # provider call, so it does not add to this count.
        assert provider.calls == len(_DOCS)

    async def test_empty_batch_returns_empty_list(self, install_provider) -> None:
        install_provider(_ECHO)
        engine = AsyncNField("mock/echo", _SCHEMA)
        assert await engine.extract_batch([]) == []

    async def test_max_concurrent_bounds_in_flight_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A provider that records concurrency: it must never exceed max_concurrent.
        peak = {"now": 0, "max": 0}

        class _CountingProvider:
            model_name = "mock/echo"
            context_window = 8192
            max_output_tokens = 8192

            async def complete(self, messages, *, max_tokens):
                peak["now"] += 1
                peak["max"] = max(peak["max"], peak["now"])
                await asyncio.sleep(0)  # yield so overlap can build up
                peak["now"] -= 1
                return _ECHO

        provider = _CountingProvider()
        monkeypatch.setattr("nfield.engine._async.from_model", lambda _m, **_k: provider)
        engine = AsyncNField("mock/echo", _SCHEMA)
        await engine.extract_batch(_DOCS * 4, max_concurrent=2)
        assert peak["max"] <= 2

    async def test_provider_failure_yields_failed_result_not_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A provider blowup is caught inside the pipeline and surfaces as a FAILED
        # result (with the call-failed fields reported), not as a raised exception -
        # so the batch keeps its one-result-per-document contract.
        class _AlwaysFails:
            model_name = "mock/echo"
            context_window = 8192
            max_output_tokens = 8192

            async def complete(self, messages, *, max_tokens):
                raise RuntimeError("synthetic failure")

        provider = _AlwaysFails()
        monkeypatch.setattr("nfield.engine._async.from_model", lambda _m, **_k: provider)
        engine = AsyncNField("mock/echo", _SCHEMA)
        results = await engine.extract_batch(_DOCS)
        assert len(results) == len(_DOCS)
        assert all(r.status is ExtractionStatus.FAILED for r in results)
        # The failure reason is surfaced on the result, not only in the logs, so a bad
        # model or missing key is not a silent empty result.
        assert all(r.metadata.fields_call_failed > 0 for r in results)
        assert all("synthetic failure" in (r.metadata.error or "") for r in results)

    async def test_return_exceptions_keeps_escaping_errors_in_place(
        self, install_provider
    ) -> None:
        # When an error truly escapes extract() (here forced via a patched extract),
        # return_exceptions=True keeps it in that document's slot; others still resolve.
        install_provider(_ECHO)
        engine = AsyncNField("mock/echo", _SCHEMA)
        real_extract = engine.extract

        async def maybe_fail(document: str, schema: object | None = None):
            if document == "boom":
                raise RuntimeError("synthetic failure")
            return await real_extract(document, schema)

        engine.extract = maybe_fail  # type: ignore[method-assign]
        results = await engine.extract_batch(
            ["Name: Alice. Age: 30.", "boom"], return_exceptions=True
        )
        assert not isinstance(results[0], BaseException)
        assert results[0].data["name"] == "Alice"
        assert isinstance(results[1], RuntimeError)

    async def test_default_re_raises_an_escaping_error(self, install_provider) -> None:
        install_provider(_ECHO)
        engine = AsyncNField("mock/echo", _SCHEMA)

        async def always_fail(document: str, schema: object | None = None):
            raise RuntimeError("synthetic failure")

        engine.extract = always_fail  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="synthetic failure"):
            await engine.extract_batch(_DOCS)


class TestSyncBatch:
    def test_sync_extract_batch(self, install_provider) -> None:
        install_provider(_ECHO)
        fs = NField("mock/echo", _SCHEMA)
        results = fs.extract_batch(_DOCS)
        assert [r.data["age"] for r in results] == [30, 30, 30]
