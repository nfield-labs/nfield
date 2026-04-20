"""
ReplayBackend and RecordingBackend — golden-fixture infrastructure for offline testing.

``RecordingBackend`` wraps any real backend and saves every ``generate()``
request/response pair to a JSONL fixture file keyed by SHA256 hash of the
request parameters.  ``ReplayBackend`` reads that file and replays responses
deterministically — no API key required after the initial recording run.

Typical workflow::

    # 1. Record once (requires a real API key)
    GROQ_API_KEY=xxx python scripts/record_fixtures.py --tasks gsm,medical_ner --quick

    # 2. Replay forever (no API key needed)
    from formatshield.backends.replay_backend import ReplayBackend

    backend = ReplayBackend("tests/fixtures/groq_responses.jsonl")
    result = await backend.generate("What is 2+2?")
    # returns the exact string that Groq returned when the fixture was recorded
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from formatshield.scorer.features import StreamEvent

_DEFAULT_FIXTURE = Path("tests/fixtures/groq_responses.jsonl")


def _hash_request(
    prompt: str,
    schema: dict[str, Any] | None,
    constraints: str | None,
    **kwargs: Any,
) -> str:
    """Return SHA256 hex of the canonical request representation.

    The hash is computed over a sorted JSON serialisation of all request
    parameters so that the same logical request always maps to the same key
    regardless of dict insertion order.
    """
    payload = json.dumps(
        {"prompt": prompt, "schema": schema, "constraints": constraints, **kwargs},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class RecordingBackend:
    """Proxy backend that intercepts ``generate()`` calls and saves them to JSONL.

    Wrap any real backend with ``RecordingBackend`` and run the benchmark once.
    All request/response pairs are appended to the fixture file so that
    :class:`ReplayBackend` can replay them in future runs with no API calls.

    Parameters
    ----------
    backend:
        Any object implementing the Backend protocol (e.g. ``GroqBackend``).
    fixture_path:
        Path to the JSONL file to append records to.  Parent directories are
        created automatically.

    Example::

        from formatshield.backends.groq_backend import GroqBackend
        from formatshield.backends.replay_backend import RecordingBackend

        real = GroqBackend()
        recorder = RecordingBackend(real, fixture_path="tests/fixtures/groq_responses.jsonl")
        result = await recorder.generate("What is 2+2?")
        # fixture file now contains one entry
    """

    def __init__(
        self,
        backend: Any,
        fixture_path: str | Path = _DEFAULT_FIXTURE,
    ) -> None:
        self._backend = backend
        self._fixture_path = Path(fixture_path)
        self._fixture_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return self._backend.name  # type: ignore[no-any-return]

    @property
    def supports_kv_cache_reuse(self) -> bool:
        return bool(getattr(self._backend, "supports_kv_cache_reuse", False))

    @property
    def accuracy_loss_baseline(self) -> float | None:
        return getattr(self._backend, "accuracy_loss_baseline", None)  # type: ignore[no-any-return]

    @property
    def supports_logit_bias(self) -> bool:
        """This backend does not support token-level logit biasing."""
        return False

    async def generate(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        constraints: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Forward to the real backend and record the exchange to the fixture file."""
        response: str = await self._backend.generate(
            prompt, schema=schema, constraints=constraints, **kwargs
        )
        key = _hash_request(prompt, schema, constraints, **kwargs)
        record = {
            "key": key,
            "prompt": prompt,
            "schema": schema,
            "constraints": constraints,
            "response": response,
        }
        with self._fixture_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return response

    async def stream(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        constraints: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Delegate streaming to the real backend (stream events are not recorded)."""
        return await self._backend.stream(  # type: ignore[no-any-return]
            prompt, schema=schema, constraints=constraints, **kwargs
        )


class ReplayBackend:
    """Replay recorded fixture responses without any API calls.

    Loads a JSONL fixture file written by :class:`RecordingBackend` and returns
    stored responses for matching requests (matched by SHA256 hash).  On a
    cache miss, falls back to
    :class:`~formatshield.backends.dryrun_backend.DryRunBackend` so tests never
    fail with ``KeyError``.

    Parameters
    ----------
    fixture_path:
        Path to the JSONL fixture file to load.  If the file does not exist,
        every call falls back to ``DryRunBackend``.
    name:
        Backend name reported to the harness.  Default ``"replay"``.

    Example::

        from formatshield.backends.replay_backend import ReplayBackend

        backend = ReplayBackend("tests/fixtures/groq_responses.jsonl")
        result = await backend.generate("What is 2+2?")
        assert isinstance(result, str)
    """

    def __init__(
        self,
        fixture_path: str | Path = _DEFAULT_FIXTURE,
        name: str = "replay",
    ) -> None:
        self._name = name
        self._fixture_path = Path(fixture_path)
        self._cache: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        """Read all records from the fixture file into the in-memory cache."""
        if not self._fixture_path.exists():
            return
        with self._fixture_path.open(encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    self._cache[record["key"]] = record["response"]
                except (json.JSONDecodeError, KeyError):
                    continue

    @property
    def name(self) -> str:
        return self._name

    @property
    def supports_kv_cache_reuse(self) -> bool:
        return False

    @property
    def accuracy_loss_baseline(self) -> float | None:
        return None

    @property
    def supports_logit_bias(self) -> bool:
        """This backend does not support token-level logit biasing."""
        return False

    def __len__(self) -> int:
        """Return the number of fixture entries loaded into the cache."""
        return len(self._cache)

    async def generate(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        constraints: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Return the recorded response, or a DryRun fallback on cache miss."""
        key = _hash_request(prompt, schema, constraints, **kwargs)
        if key in self._cache:
            return self._cache[key]
        # Cache miss — fall back to DryRunBackend so tests remain green
        from formatshield.backends.dryrun_backend import DryRunBackend

        fallback = DryRunBackend(seed=42)
        return await fallback.generate(prompt, schema=schema, constraints=constraints, **kwargs)

    async def stream(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        constraints: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming is not supported — delegates to DryRunBackend."""
        from formatshield.backends.dryrun_backend import DryRunBackend

        fallback = DryRunBackend(seed=42)
        return await fallback.stream(prompt, schema=schema, constraints=constraints, **kwargs)
