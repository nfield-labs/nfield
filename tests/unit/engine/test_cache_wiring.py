"""Engine-level wiring for the response cache.

Two things must hold: the config's ``cache`` reaches the provider factory, and a
warm run over an identical document reuses stored responses instead of re-calling
the model. The first is asserted by capturing the factory kwargs; the second by a
cache-aware mock provider driven end-to-end through S0-S6.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nfield import AsyncNField
from nfield.config import ExtractionConfig
from nfield.providers._cache import MemoryCache, make_cache_key

if TYPE_CHECKING:
    import pytest

_DOC = "Name: Alice. Age: 30."
_OTHER_DOC = "Name: Bob. Age: 41."
_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
}
_ECHO = "name = Alice\nage = 30"


class _CachingMock:
    """Mock provider that honors an attached cache exactly as BaseProvider does."""

    model_name = "mock/echo"

    def __init__(self, sfep_text: str, cache: object | None) -> None:
        self._sfep = sfep_text
        self.cache = cache
        self.raw_calls = 0
        self.context_window = 8192
        self.max_output_tokens = 8192

    async def complete(self, messages: list[dict[str, str]], *, max_tokens: int) -> str:
        if self.cache is not None:
            key = make_cache_key(self.model_name, messages, max_tokens)
            hit = self.cache.get(key)
            if hit is not None:
                return hit
            self.raw_calls += 1
            self.cache.set(key, self._sfep)
            return self._sfep
        self.raw_calls += 1
        return self._sfep


class TestCacheReachesFactory:
    """ExtractionConfig.cache is resolved and passed to from_model."""

    def test_default_passes_no_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, object] = {}

        def fake(_model: str, **kwargs: object) -> _CachingMock:
            seen.update(kwargs)
            return _CachingMock(_ECHO, kwargs.get("cache"))

        monkeypatch.setattr("nfield.engine._async.from_model", fake)
        AsyncNField("mock/echo", _SCHEMA)
        assert seen["cache"] is None

    def test_true_passes_memory_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, object] = {}

        def fake(_model: str, **kwargs: object) -> _CachingMock:
            seen.update(kwargs)
            return _CachingMock(_ECHO, kwargs.get("cache"))

        monkeypatch.setattr("nfield.engine._async.from_model", fake)
        AsyncNField("mock/echo", _SCHEMA, config=ExtractionConfig(cache=True))
        assert isinstance(seen["cache"], MemoryCache)


class TestWarmRun:
    """A repeated identical extraction reuses cached responses."""

    async def test_identical_document_makes_no_new_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = _CachingMock(_ECHO, None)

        def fake(_model: str, **kwargs: object) -> _CachingMock:
            provider.cache = kwargs.get("cache")
            return provider

        monkeypatch.setattr("nfield.engine._async.from_model", fake)
        engine = AsyncNField(
            "mock/echo", _SCHEMA, config=ExtractionConfig(cache=True, max_retry_rounds=0)
        )
        first = await engine.extract(_DOC)
        cold_calls = provider.raw_calls
        second = await engine.extract(_DOC)

        assert first.data == second.data
        assert cold_calls > 0
        assert provider.raw_calls == cold_calls  # every leaf of the rerun hit the cache

    async def test_different_document_misses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = _CachingMock(_ECHO, None)

        def fake(_model: str, **kwargs: object) -> _CachingMock:
            provider.cache = kwargs.get("cache")
            return provider

        monkeypatch.setattr("nfield.engine._async.from_model", fake)
        engine = AsyncNField(
            "mock/echo", _SCHEMA, config=ExtractionConfig(cache=True, max_retry_rounds=0)
        )
        await engine.extract(_DOC)
        cold_calls = provider.raw_calls
        await engine.extract(_OTHER_DOC)
        assert provider.raw_calls > cold_calls  # a new document is a cache miss
