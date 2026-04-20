"""
Unit tests for ReplayBackend and RecordingBackend.

All tests are offline — no API keys, no network calls.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from formatshield.backends.replay_backend import (
    RecordingBackend,
    ReplayBackend,
    _hash_request,
)

# ---------------------------------------------------------------------------
# _hash_request helpers
# ---------------------------------------------------------------------------


def test_hash_request_is_deterministic() -> None:
    h1 = _hash_request("hello", None, None)
    h2 = _hash_request("hello", None, None)
    assert h1 == h2


def test_hash_request_differs_on_different_prompt() -> None:
    h1 = _hash_request("hello", None, None)
    h2 = _hash_request("world", None, None)
    assert h1 != h2


def test_hash_request_differs_on_schema_change() -> None:
    h1 = _hash_request("p", {"type": "object"}, None)
    h2 = _hash_request("p", {"type": "string"}, None)
    assert h1 != h2


def test_hash_request_differs_on_constraints() -> None:
    h1 = _hash_request("p", None, "json")
    h2 = _hash_request("p", None, None)
    assert h1 != h2


def test_hash_request_is_64_hex_chars() -> None:
    h = _hash_request("test", None, None)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# ReplayBackend — construction
# ---------------------------------------------------------------------------


def test_replay_backend_default_name() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = ReplayBackend(Path(tmpdir) / "fixture.jsonl")
        assert backend.name == "replay"


def test_replay_backend_custom_name() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = ReplayBackend(Path(tmpdir) / "fixture.jsonl", name="groq-replay")
        assert backend.name == "groq-replay"


def test_replay_backend_empty_on_missing_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = ReplayBackend(Path(tmpdir) / "nonexistent.jsonl")
        assert len(backend) == 0


def test_replay_backend_supports_kv_cache_reuse_false() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = ReplayBackend(Path(tmpdir) / "f.jsonl")
        assert backend.supports_kv_cache_reuse is False


def test_replay_backend_accuracy_loss_baseline_none() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = ReplayBackend(Path(tmpdir) / "f.jsonl")
        assert backend.accuracy_loss_baseline is None


# ---------------------------------------------------------------------------
# ReplayBackend — fixture loading
# ---------------------------------------------------------------------------


def _write_fixture(path: Path, entries: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")


def test_replay_backend_loads_fixture_len() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        fixture = Path(tmpdir) / "fixture.jsonl"
        key = _hash_request("hello", None, None)
        _write_fixture(
            fixture,
            [
                {
                    "key": key,
                    "prompt": "hello",
                    "schema": None,
                    "constraints": None,
                    "response": "hi",
                }
            ],
        )
        backend = ReplayBackend(fixture)
        assert len(backend) == 1


@pytest.mark.asyncio
async def test_replay_backend_returns_stored_response() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        fixture = Path(tmpdir) / "fixture.jsonl"
        key = _hash_request("What is 2+2?", None, "json")
        _write_fixture(
            fixture,
            [
                {
                    "key": key,
                    "prompt": "What is 2+2?",
                    "schema": None,
                    "constraints": "json",
                    "response": '{"answer": 4}',
                }
            ],
        )
        backend = ReplayBackend(fixture)
        result = await backend.generate("What is 2+2?", constraints="json")
        assert result == '{"answer": 4}'


@pytest.mark.asyncio
async def test_replay_backend_cache_miss_returns_string() -> None:
    """On cache miss, falls back to DryRunBackend — result is still a string."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = ReplayBackend(Path(tmpdir) / "empty.jsonl")
        result = await backend.generate("unknown prompt")
        assert isinstance(result, str)


def test_replay_backend_skips_malformed_lines() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        fixture = Path(tmpdir) / "fixture.jsonl"
        good_key = _hash_request("ok", None, None)
        with fixture.open("w") as fh:
            fh.write("not-json\n")
            fh.write('{"missing_key": true}\n')
            fh.write(json.dumps({"key": good_key, "response": "good"}) + "\n")
        backend = ReplayBackend(fixture)
        assert len(backend) == 1


# ---------------------------------------------------------------------------
# RecordingBackend
# ---------------------------------------------------------------------------


def _make_fake_backend(response: str = "fake response", name: str = "groq") -> MagicMock:
    fake = MagicMock()
    fake.name = name
    fake.supports_kv_cache_reuse = False
    fake.accuracy_loss_baseline = 0.18
    fake.generate = AsyncMock(return_value=response)
    return fake


@pytest.mark.asyncio
async def test_recording_backend_name_delegates() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        fake = _make_fake_backend(name="mybackend")
        recorder = RecordingBackend(fake, fixture_path=Path(tmpdir) / "out.jsonl")
        assert recorder.name == "mybackend"


@pytest.mark.asyncio
async def test_recording_backend_returns_original_response() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        fake = _make_fake_backend(response='{"value": 42}')
        recorder = RecordingBackend(fake, fixture_path=Path(tmpdir) / "out.jsonl")
        result = await recorder.generate("test prompt", constraints="json")
        assert result == '{"value": 42}'


@pytest.mark.asyncio
async def test_recording_backend_writes_fixture_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        fixture = Path(tmpdir) / "out.jsonl"
        fake = _make_fake_backend(response="hello")
        recorder = RecordingBackend(fake, fixture_path=fixture)

        await recorder.generate("prompt one")
        await recorder.generate("prompt two")

        lines = fixture.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2


@pytest.mark.asyncio
async def test_recording_backend_fixture_has_correct_fields() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        fixture = Path(tmpdir) / "out.jsonl"
        fake = _make_fake_backend(response='{"x": 1}')
        recorder = RecordingBackend(fake, fixture_path=fixture)

        await recorder.generate("my prompt", schema={"type": "object"}, constraints="json")

        record = json.loads(fixture.read_text(encoding="utf-8").strip())
        assert record["prompt"] == "my prompt"
        assert record["constraints"] == "json"
        assert record["response"] == '{"x": 1}'
        assert len(record["key"]) == 64


@pytest.mark.asyncio
async def test_recording_backend_replay_roundtrip() -> None:
    """Record with RecordingBackend, replay with ReplayBackend — responses match."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fixture = Path(tmpdir) / "out.jsonl"
        fake = _make_fake_backend(response='{"result": "yes"}')
        recorder = RecordingBackend(fake, fixture_path=fixture)

        original = await recorder.generate("roundtrip test", constraints="json")

        replay = ReplayBackend(fixture)
        replayed = await replay.generate("roundtrip test", constraints="json")

        assert replayed == original


def test_recording_backend_creates_parent_dirs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        deep_path = Path(tmpdir) / "a" / "b" / "c" / "fixture.jsonl"
        fake = _make_fake_backend()
        # Should not raise even though parent dirs don't exist yet
        RecordingBackend(fake, fixture_path=deep_path)
        assert deep_path.parent.exists()
