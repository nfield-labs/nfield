"""Exact-match response cache for provider completions.

Keys a completed model response on the full request (model, messages, max_tokens).
An identical request returns the stored text instead of calling the model, so a
benchmark rerun or a repeated document costs nothing. Exact-match by construction:
any change to the request yields a different key, so a hit is always the same text
the model would have produced - never a similar request's answer.

Two backends ship here (in-memory LRU, on-disk). Either can be swapped for a custom
store satisfying :class:`ResponseCache` (e.g. Redis-backed).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Protocol, runtime_checkable

__all__ = ["DiskCache", "MemoryCache", "ResponseCache", "make_cache_key", "resolve_cache"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Folded into every key; bump to invalidate all entries when the key scheme changes.
_CACHE_VERSION: int = 1
# Holds several documents' worth of leaf calls for reruns while bounding memory.
_DEFAULT_MEMORY_CACHE_SIZE: int = 1024
# On-disk entry suffix; the filename stem is the hex digest, already filesystem-safe.
_DISK_ENTRY_SUFFIX: str = ".txt"


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ResponseCache(Protocol):
    """A store of completed model responses, keyed by an opaque request key.

    Implement ``get``/``set`` to plug a custom backend (Redis, a database, ...).
    ``get`` returns the stored text or ``None`` on a miss; ``set`` records a
    successful completion. A cache is best-effort: it never raises into the caller.
    """

    def get(self, key: str) -> str | None:
        """Return the cached response for *key*, or ``None`` on a miss."""
        ...

    def set(self, key: str, value: str) -> None:
        """Store *value* as the response for *key*."""
        ...


# ---------------------------------------------------------------------------
# Key hashing
# ---------------------------------------------------------------------------


def make_cache_key(model_name: str, messages: list[dict[str, str]], max_tokens: int) -> str:
    """Hash a completion request into a stable, filesystem-safe cache key.

    The key covers everything that determines the output for a fixed provider - the
    model name, the messages (order preserved), the output ceiling - plus a format
    version so a scheme change invalidates old entries. Message dict keys are sorted
    so ``{"role", "content"}`` ordering never changes the key.

    Args:
        model_name: The provider's model name (e.g. "llama-3.3-70b-versatile").
        messages: The chat messages sent to the model.
        max_tokens: The output-token ceiling for the call.

    Returns:
        A 64-character hex SHA-256 digest of the canonical request.

    Example:
        >>> make_cache_key("m", [{"role": "user", "content": "hi"}], 8) != ""
        True
    """
    payload = json.dumps(
        {"v": _CACHE_VERSION, "model": model_name, "messages": messages, "max_tokens": max_tokens},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class MemoryCache:
    """In-process LRU cache of responses, bounded to ``max_size`` entries.

    Least-recently-used eviction keeps memory bounded across a long-running or batch
    session. Thread-safe: get/set are guarded so the concurrent leaf calls the engine
    fires under its semaphore cannot corrupt the ordering.

    Attributes:
        max_size: Maximum number of entries retained before LRU eviction.
    """

    def __init__(self, *, max_size: int = _DEFAULT_MEMORY_CACHE_SIZE) -> None:
        """Initialize the cache.

        Args:
            max_size: Maximum entries before least-recently-used eviction. Must be > 0.

        Raises:
            ValueError: If ``max_size`` is not positive.
        """
        if max_size <= 0:
            raise ValueError(f"max_size must be > 0, got {max_size}")
        self.max_size = max_size
        self._store: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        """Return the cached response for *key* and mark it most-recently-used."""
        with self._lock:
            if key not in self._store:
                return None
            self._store.move_to_end(key)
            return self._store[key]

    def set(self, key: str, value: str) -> None:
        """Store *value* for *key*, evicting the least-recently-used entry if full."""
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def clear(self) -> None:
        """Drop every entry from the cache."""
        with self._lock:
            self._store.clear()


class DiskCache:
    """Persistent response cache: one file per entry under ``directory``.

    Survives process restarts, so re-running the same extraction reads the stored
    responses instead of calling the model. Writes are atomic (temp file +
    ``os.replace``), so a crash mid-write never leaves a torn entry, and distinct
    keys touch distinct files, so concurrent writers do not contend.

    The key covers the model and request but not other provider settings, so a single
    directory should serve one model configuration; use separate directories for runs
    that differ in, say, reasoning-model handling.

    Attributes:
        directory: The directory holding one file per cached entry.
    """

    def __init__(self, directory: str | Path) -> None:
        """Initialize the cache, creating *directory* if needed.

        Args:
            directory: Where cache entries are stored (created with parents).
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._resolved_dir = self.directory.resolve()

    def _path(self, key: str) -> Path:
        """Return the on-disk path for *key*, rejecting any key that escapes the directory."""
        candidate = (self.directory / f"{key}{_DISK_ENTRY_SUFFIX}").resolve()
        if candidate.parent != self._resolved_dir:
            raise ValueError("cache key must name a file inside the cache directory")
        return candidate

    def get(self, key: str) -> str | None:
        """Return the cached response for *key*, or ``None`` on a miss or read error."""
        try:
            return self._path(key).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            # Fail-open: a missing, unreadable, or non-UTF-8 entry is a miss, not a crash.
            return None

    def set(self, key: str, value: str) -> None:
        """Store *value* for *key* via an atomic temp-file replace."""
        path = self._path(key)
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(value, encoding="utf-8")
        os.replace(tmp, path)

    def clear(self) -> None:
        """Delete every entry file in the cache directory."""
        for entry in self.directory.glob(f"*{_DISK_ENTRY_SUFFIX}"):
            entry.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def resolve_cache(spec: bool | ResponseCache) -> ResponseCache | None:
    """Turn an ``ExtractionConfig.cache`` value into a cache instance, or ``None``.

    Args:
        spec: ``False`` disables caching; ``True`` builds an in-memory LRU cache; a
            ``ResponseCache`` instance is used as given (disk, Redis, ...).

    Returns:
        A cache to attach to the provider, or ``None`` when caching is off.

    Example:
        >>> resolve_cache(False) is None
        True
        >>> isinstance(resolve_cache(True), MemoryCache)
        True
    """
    if spec is False:
        return None
    if spec is True:
        return MemoryCache()
    return spec
