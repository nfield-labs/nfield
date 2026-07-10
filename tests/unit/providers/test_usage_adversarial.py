"""Hard cases for usage accounting: cache interplay, old payloads, engine chains."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, ClassVar

from nfield.providers._base import BaseProvider
from nfield.providers._cache import MemoryCache
from nfield.providers._usage import Usage, start_usage
from nfield.types import ExtractionResult, ExtractionStatus, Metadata

if TYPE_CHECKING:
    import pytest


class _CountingProvider(BaseProvider):
    """Real BaseProvider subclass that reports fixed usage per raw call."""

    def __init__(self) -> None:
        super().__init__("mock/counting")
        self.raw_calls = 0

    async def _raw_complete(self, messages: list[dict[str, str]], *, max_tokens: int) -> str:
        self.raw_calls += 1
        self._record_usage(100, 10)
        return "ok"

    def _get_client(self) -> object:
        return None

    @property
    def context_window(self) -> int:
        return 8192

    @property
    def max_output_tokens(self) -> int:
        return 1024


class TestCacheHitRecordsNothing:
    """A cache hit skips the API call, so it must add zero to the run's tally."""

    def test_miss_records_hit_does_not(self) -> None:
        async def scenario() -> tuple[Usage, _CountingProvider]:
            usage = start_usage()
            provider = _CountingProvider()
            provider.cache = MemoryCache()
            messages = [{"role": "user", "content": "hi"}]
            await provider.complete(messages, max_tokens=8)  # miss -> raw call
            await provider.complete(messages, max_tokens=8)  # hit -> no call
            return usage, provider

        usage, provider = asyncio.run(scenario())
        assert provider.raw_calls == 1
        assert (usage.prompt_tokens, usage.completion_tokens, usage.calls) == (100, 10, 1)


class TestOldPayloadCompatibility:
    """Results saved before token fields existed must still load."""

    _OLD_METADATA: ClassVar[dict[str, object]] = {
        "K": 1,
        "K_min": 1,
        "optimality_gap": 0.0,
        "quality_score": 1.0,
        "confidence_level": "HIGH",
        "fields_extracted": 1,
        "fields_total": 1,
        "fields_missing": 0,
        "fields_conflicted": 0,
        "fields_needs_revalidation": 0,
        "per_field_confidence": {},
        "retry_rounds": 0,
    }

    def test_metadata_defaults_the_new_fields(self) -> None:
        meta = Metadata(**self._OLD_METADATA)  # type: ignore[arg-type]
        assert (meta.tokens_prompt, meta.tokens_completion, meta.cost) == (0, 0, None)

    def test_result_round_trips_the_new_fields(self) -> None:
        meta = Metadata(**self._OLD_METADATA, tokens_prompt=7, tokens_completion=3, cost=0.5)  # type: ignore[arg-type]
        result = ExtractionResult(data={}, metadata=meta, status=ExtractionStatus.SUCCESS)
        restored = ExtractionResult.from_dict(result.to_dict())
        assert restored.metadata.tokens_prompt == 7
        assert restored.metadata.tokens_completion == 3
        assert restored.metadata.cost == 0.5


class TestEngineBuildsTheChain:
    """fallback_model as a list constructs one provider per entry, in order."""

    def test_list_builds_providers_in_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from nfield import AsyncNField
        from nfield.config import ExtractionConfig

        built: list[str] = []

        class _Mock:
            model_name = "mock"
            context_window = 8192
            max_output_tokens = 1024

            async def complete(self, messages: object, *, max_tokens: int) -> str:
                return ""

        def fake(model: str, **_kwargs: object) -> _Mock:
            built.append(model)
            return _Mock()

        monkeypatch.setattr("nfield.engine._async.from_model", fake)
        AsyncNField(
            "mock/primary",
            {"type": "object", "properties": {"a": {"type": "string"}}},
            config=ExtractionConfig(fallback_model=["mock/mid", "mock/strong"]),
        )
        assert built == ["mock/primary", "mock/mid", "mock/strong"]

    def test_single_string_builds_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from nfield import AsyncNField
        from nfield.config import ExtractionConfig

        built: list[str] = []

        class _Mock:
            model_name = "mock"
            context_window = 8192
            max_output_tokens = 1024

            async def complete(self, messages: object, *, max_tokens: int) -> str:
                return ""

        def fake(model: str, **_kwargs: object) -> _Mock:
            built.append(model)
            return _Mock()

        monkeypatch.setattr("nfield.engine._async.from_model", fake)
        AsyncNField(
            "mock/primary",
            {"type": "object", "properties": {"a": {"type": "string"}}},
            config=ExtractionConfig(fallback_model="mock/strong"),
        )
        assert built == ["mock/primary", "mock/strong"]
