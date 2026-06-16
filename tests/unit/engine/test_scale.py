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
    monkeypatch.setattr("formatshield.engine._async.from_model", lambda _m, **_kw: provider)
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

    async def test_context_window_drives_leaf_count(self, monkeypatch):
        # The public context_window / max_output_tokens must reach capacity
        # planning: a bigger window packs the same schema into fewer calls.
        sfep = _full_sfep(300)

        def factory(_model, *, context_window=None, max_output_tokens=None, **_kwargs):
            return MockProvider(
                sfep,
                context_window=context_window or 8192,
                max_output_tokens=max_output_tokens or 8192,
            )

        monkeypatch.setattr("formatshield.engine._async.from_model", factory)
        # Raise the field cap so the CONTEXT WINDOW is the binding constraint here
        # (this test isolates context-window → leaf-count; the field cap is its own
        # test, TestMaxFieldsPerCall).
        cfg = ExtractionConfig(max_retry_rounds=0, max_fields_per_call=1000)

        small = await AsyncFormatShield(
            "groq/x", _wide_schema(300), context_window=8192, max_output_tokens=8192, config=cfg
        ).extract("doc")
        big = await AsyncFormatShield(
            "groq/x",
            _wide_schema(300),
            context_window=131_072,
            max_output_tokens=131_072,
            config=cfg,
        ).extract("doc")

        assert big.metadata.fields_total == small.metadata.fields_total == 300
        assert big.metadata.K < small.metadata.K  # bigger window → fewer calls
        assert len(big.data) == len(small.data) == 300


class TestInstructions:
    """Caller instructions reach the provider and are charged to leaf overhead."""

    async def test_instructions_reach_provider(self, install_provider):
        provider = install_provider("name = Alice\nage = 30")
        engine = AsyncFormatShield(
            "mock/echo",
            {"type": "object", "properties": {"name": {"type": "string"}}},
            instructions="DOMAIN: clinical trial records. Prefer ISO dates.",
            config=ExtractionConfig(max_retry_rounds=0),
        )
        await engine.extract("doc")
        system_msg = provider.last_messages[0]["content"]
        user_msg = provider.last_messages[1]["content"]
        # Instructions reach the model in the USER turn (better Llama adherence);
        # the system message stays the SFEP contract.
        assert "DOMAIN: clinical trial records. Prefer ISO dates." in user_msg
        assert "OUTPUT FORMAT" in system_msg  # SFEP contract preserved

    async def test_large_instructions_increases_leaf_count(self, monkeypatch):
        # A big instructions string eats the per-leaf budget, so the same schema
        # must split into more calls — proof it is counted in overhead, not ignored.
        sfep = _full_sfep(50)

        def factory(_model, *, context_window=None, max_output_tokens=None, **_kwargs):
            return MockProvider(
                sfep,
                context_window=context_window or 8192,
                max_output_tokens=max_output_tokens or 8192,
            )

        monkeypatch.setattr("formatshield.engine._async.from_model", factory)
        cfg = ExtractionConfig(max_retry_rounds=0)
        schema = _wide_schema(50)

        empty = await AsyncFormatShield(
            "groq/x", schema, context_window=8192, max_output_tokens=8192, config=cfg
        ).extract("doc")
        huge = await AsyncFormatShield(
            "groq/x",
            schema,
            context_window=8192,
            max_output_tokens=8192,
            instructions="X" * 20_000,  # ~5000 tokens of caller context
            config=cfg,
        ).extract("doc")

        assert huge.metadata.K > empty.metadata.K
        assert huge.metadata.fields_total == empty.metadata.fields_total == 50
