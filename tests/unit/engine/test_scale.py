"""Aggressive width tests through the public engine API.

The deterministic full-pipeline scale test exercises the raw stages; this one
drives the same widths (200 / 500 / 1000 fields) through ``nfield`` and
``AsyncFormatShield`` so the public surface is proven to thread hundreds of
leaves and reassemble every field. The mock provider's small context window
forces capacity packing into many leaves, so the multi-leaf path is real.
"""

from __future__ import annotations

import pytest

from formatshield import AsyncFormatShield, nfield
from formatshield.config import ExtractionConfig

from .conftest import MockProvider


def _wide_schema(n: int) -> dict:
    return {
        "type": "object",
        "properties": {
            f"field_{i:04d}": {"type": "string", "description": f"synthetic field {i}"}
            for i in range(n)
        },
    }


def _full_sfep(n: int) -> str:
    return "\n".join(f"field_{i:04d} = value{i:04d}" for i in range(n))


def _install(monkeypatch, n: int) -> MockProvider:
    provider = MockProvider(_full_sfep(n))
    monkeypatch.setattr("formatshield.engine._async.from_model", lambda _m: provider)
    return provider


class TestPublicApiScale:
    @pytest.mark.parametrize("n", [200, 500, 1000])
    async def test_async_engine_reassembles_every_field(self, monkeypatch, n):
        _install(monkeypatch, n)
        engine = AsyncFormatShield(
            "mock/echo", _wide_schema(n), config=ExtractionConfig(max_retry_rounds=0)
        )
        result = await engine.extract("short document")

        assert result.metadata.fields_total == n
        assert result.metadata.fields_extracted == n
        assert len(result.data) == n
        # Spot-check boundary values survived packing → extraction → assembly.
        assert result.data[f"field_{0:04d}"] == "value0000"
        assert result.data[f"field_{n - 1:04d}"] == f"value{n - 1:04d}"

    async def test_many_leaves_one_call_each(self, monkeypatch):
        provider = _install(monkeypatch, 500)
        engine = AsyncFormatShield(
            "mock/echo", _wide_schema(500), config=ExtractionConfig(max_retry_rounds=0)
        )
        result = await engine.extract("short document")
        # 500 fields cannot fit one 8K-context call → many leaves, one call per leaf.
        assert result.metadata.K > 1
        assert provider.calls == result.metadata.K
        assert result.metadata.K_min >= 1
        assert result.metadata.K_min <= result.metadata.K

    def test_sync_nfield_at_scale(self, monkeypatch):
        _install(monkeypatch, 200)
        result = nfield(
            "short document",
            _wide_schema(200),
            "mock/echo",
            config=ExtractionConfig(max_retry_rounds=0),
        )
        assert result.metadata.fields_total == 200
        assert len(result.data) == 200
