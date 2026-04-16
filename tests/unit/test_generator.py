"""Unit tests for FormatShieldGenerator and AsyncFormatShieldGenerator — no API keys."""

from __future__ import annotations

import pytest

from formatshield.backends.dryrun_backend import DryRunBackend
from formatshield.core import FormatShield
from formatshield.generator import AsyncFormatShieldGenerator, FormatShieldGenerator

# ---------------------------------------------------------------------------
# FormatShieldGenerator — sync tests
# ---------------------------------------------------------------------------


def _make_shield() -> FormatShield:
    return FormatShield(model="dryrun/test", backend=DryRunBackend())


def test_generator_construction() -> None:
    """FormatShieldGenerator is constructed without errors."""
    shield = _make_shield()
    gen = FormatShieldGenerator(shield)
    assert gen.schema is None
    assert gen.output_type is None


def test_generator_with_schema() -> None:
    """FormatShieldGenerator stores schema at construction."""
    shield = _make_shield()
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    gen = FormatShieldGenerator(shield, schema=schema)
    assert gen.schema == schema


def test_generator_with_output_type() -> None:
    """FormatShieldGenerator stores output_type at construction."""
    shield = _make_shield()
    gen = FormatShieldGenerator(shield, output_type=int)
    assert gen.output_type is int


def test_generator_callable() -> None:
    """FormatShieldGenerator is callable and returns GenerationResult."""
    shield = _make_shield()
    gen = FormatShieldGenerator(shield, output_type=int)
    result = gen("What is 2+2?")
    assert result is not None
    assert isinstance(result.output, str)


def test_generator_batch() -> None:
    """FormatShieldGenerator.batch returns results in input order."""
    shield = _make_shield()
    gen = FormatShieldGenerator(shield)
    prompts = ["Q1", "Q2", "Q3"]
    results = gen.batch(prompts)
    assert len(results) == 3
    assert all(r is not None for r in results)


def test_generator_batch_empty() -> None:
    """Empty batch returns empty list."""
    shield = _make_shield()
    gen = FormatShieldGenerator(shield)
    results = gen.batch([])
    assert results == []


def test_generator_via_shield_factory() -> None:
    """FormatShield.generator() returns a FormatShieldGenerator."""
    shield = _make_shield()
    gen = shield.generator(output_type=int)
    assert isinstance(gen, FormatShieldGenerator)
    assert gen.output_type is int


def test_generator_schema_cached() -> None:
    """Schema is retained across multiple calls."""
    shield = _make_shield()
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    gen = FormatShieldGenerator(shield, schema=schema)
    # Call twice — schema should be consistent
    r1 = gen("First prompt")
    r2 = gen("Second prompt")
    assert r1.schema_valid is True
    assert r2.schema_valid is True


# ---------------------------------------------------------------------------
# AsyncFormatShieldGenerator — async tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_generator_callable() -> None:
    """AsyncFormatShieldGenerator is awaitable and returns GenerationResult."""
    shield = _make_shield()
    gen = AsyncFormatShieldGenerator(shield)
    result = await gen("What is 2+2?")
    assert result is not None
    assert isinstance(result.output, str)


@pytest.mark.asyncio
async def test_async_generator_batch() -> None:
    """AsyncFormatShieldGenerator.batch returns all results."""
    shield = _make_shield()
    gen = AsyncFormatShieldGenerator(shield)
    prompts = ["Q1", "Q2", "Q3"]
    results = await gen.batch(prompts)
    assert len(results) == 3
    assert all(r is not None for r in results)


@pytest.mark.asyncio
async def test_async_generator_batch_with_concurrency() -> None:
    """AsyncFormatShieldGenerator.batch respects max_concurrency."""
    shield = _make_shield()
    gen = AsyncFormatShieldGenerator(shield)
    prompts = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    # max_concurrency=2 should still return all results
    results = await gen.batch(prompts, max_concurrency=2)
    assert len(results) == 5


@pytest.mark.asyncio
async def test_async_generator_batch_empty() -> None:
    """Empty async batch returns empty list."""
    shield = _make_shield()
    gen = AsyncFormatShieldGenerator(shield)
    results = await gen.batch([])
    assert results == []


@pytest.mark.asyncio
async def test_async_generator_with_output_type() -> None:
    """AsyncFormatShieldGenerator passes output_type through."""
    shield = _make_shield()
    gen = AsyncFormatShieldGenerator(shield, output_type=int)
    result = await gen("What is 2+2?")
    assert result is not None
    assert isinstance(result.parsed, int)


def test_async_generator_via_shield_factory() -> None:
    """FormatShield.async_generator() returns AsyncFormatShieldGenerator."""
    shield = _make_shield()
    gen = shield.async_generator(output_type=int)
    assert isinstance(gen, AsyncFormatShieldGenerator)
    assert gen.output_type is int


def test_async_generator_properties() -> None:
    """AsyncFormatShieldGenerator properties are accessible."""
    shield = _make_shield()
    schema = {"type": "string"}
    gen = AsyncFormatShieldGenerator(shield, schema=schema, output_type=str)
    assert gen.schema == schema
    assert gen.output_type is str


# ---------------------------------------------------------------------------
# from_provider() factory
# ---------------------------------------------------------------------------


def test_from_provider_creates_shield() -> None:
    """from_provider() creates a FormatShield instance."""
    import formatshield as fs

    shield = fs.from_provider("dryrun/test")
    assert isinstance(shield, fs.FormatShield)


def test_from_provider_passes_kwargs() -> None:
    """from_provider() passes kwargs to FormatShield."""
    import formatshield as fs

    shield = fs.from_provider("dryrun/test", debug=True)
    assert shield._debug is True
