"""Integration tests for GroqBackend — requires GROQ_API_KEY environment variable."""

from __future__ import annotations

import json
import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set — skipping Groq integration tests",
)


@pytest.fixture
def groq_backend():
    from formatshield.backends.groq_backend import GroqBackend

    return GroqBackend(model="llama-3.1-8b-instant")  # Use smaller model for tests


@pytest.mark.asyncio
async def test_groq_generate_returns_string(groq_backend) -> None:
    result = await groq_backend.generate("Say 'hello' in one word.")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_groq_generate_with_json_constraints(groq_backend) -> None:
    result = await groq_backend.generate(
        "Return a JSON object with key 'value' set to 42.",
        constraints="json",
    )
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert isinstance(parsed, dict)


@pytest.mark.asyncio
async def test_groq_stream_yields_events(groq_backend) -> None:
    events = []
    stream = await groq_backend.stream("Count to 3.")
    async for event in stream:
        events.append(event)
        if len(events) > 20:
            break

    assert len(events) >= 1
    types = {e.type for e in events}
    assert types <= {"output", "complete"}


def test_groq_backend_name(groq_backend) -> None:
    assert groq_backend.name == "groq"


def test_groq_supports_kv_cache_false(groq_backend) -> None:
    assert groq_backend.supports_kv_cache_reuse is False


def test_groq_missing_api_key() -> None:
    from formatshield.backends.groq_backend import GroqBackend

    with pytest.raises((ValueError, Exception)):
        GroqBackend(api_key="invalid_key_that_wont_work", model="llama-3.1-8b-instant")
