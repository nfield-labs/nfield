"""
Cross-schema TTF trace cache.

Caches Pass 1 reasoning traces keyed on schema structure rather than the full
prompt+schema+backend hash used by the main response cache.  This allows
structurally identical schemas (same field names, types, and nesting — just
different prompts) to share a reasoning scaffold, dramatically reducing Pass 1
inference cost in multi-schema pipelines.

Key design decisions
--------------------
* **Schema-structure key** — the cache key is a deterministic hash of the
  schema's field-graph topology: property names, types, required flags, and
  nesting depth.  Enum values, descriptions, and title are excluded so that
  minor metadata differences do not prevent cache hits.

* **Scaffold injection** — a cached trace is not blindly replayed.  It is
  injected as a *scaffold* prefix inside the think prompt so the model can
  adapt it to the current prompt rather than copying it verbatim.

* **LRU eviction** — bounded by ``max_size`` (default 256 schemas).  When the
  cache is full the least-recently-used schema trace is evicted.

* **TTL expiry** — entries older than ``ttl_seconds`` (default 3600 s) are
  treated as stale and not returned, though they remain in the LRU until
  evicted naturally.

* **Thread-safe** — uses a ``threading.Lock`` so the cache is safe under
  concurrent async event-loop threads (each asyncio event loop runs in one
  OS thread, but multiple loops can share a cache instance).

Public API
----------
- :class:`TraceCache` — the cache class
- :func:`build_schema_cache_key` — compute the schema-structure hash
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_SIZE: int = 256
DEFAULT_TTL_SECONDS: float = 3600.0


# ---------------------------------------------------------------------------
# Schema-structure key
# ---------------------------------------------------------------------------


def build_schema_cache_key(schema: dict[str, Any]) -> str:
    """Return a stable hex digest keyed on the *structural* shape of *schema*.

    Only the following are included in the hash:
    - property names (sorted)
    - property types
    - required flags
    - nesting structure (recursive)

    Excluded: descriptions, titles, examples, enum values, format strings,
    pattern strings, minimum/maximum constraints.  This maximises cache hit
    rate for schemas that share the same logical structure but differ in
    documentation or validation rules.

    Parameters
    ----------
    schema:
        JSON Schema dict.

    Returns
    -------
    str
        12-character lowercase hex digest.
    """
    canonical = _extract_structural_shape(schema)
    serialised = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialised.encode()).hexdigest()[:12]


def _extract_structural_shape(node: Any, depth: int = 0) -> Any:
    """Recursively extract structural shape from a JSON Schema node."""
    if not isinstance(node, dict) or depth > 10:
        return None

    shape: dict[str, Any] = {}

    node_type = node.get("type")
    if node_type:
        shape["t"] = node_type

    props = node.get("properties")
    if isinstance(props, dict):
        shape["p"] = {k: _extract_structural_shape(v, depth + 1) for k, v in sorted(props.items())}

    required = node.get("required")
    if isinstance(required, list):
        shape["r"] = sorted(required)

    items = node.get("items")
    if isinstance(items, dict):
        shape["i"] = _extract_structural_shape(items, depth + 1)

    any_of = node.get("anyOf") or node.get("oneOf")
    if isinstance(any_of, list):
        shape["u"] = [_extract_structural_shape(s, depth + 1) for s in any_of]

    return shape


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------


class _CacheEntry:
    """Internal record stored per cache key."""

    __slots__ = ("created_at", "hit_count", "trace")

    def __init__(self, trace: str) -> None:
        self.trace = trace
        self.created_at: float = time.monotonic()
        self.hit_count: int = 0


# ---------------------------------------------------------------------------
# TraceCache
# ---------------------------------------------------------------------------


class TraceCache:
    """LRU cache for Pass 1 reasoning traces, keyed on schema structure.

    Parameters
    ----------
    max_size:
        Maximum number of distinct schema structures to cache.  When the cache
        is full the LRU entry is evicted.
    ttl_seconds:
        Maximum age (seconds) of a cached trace before it is considered stale.
        Stale entries are not returned but remain in the LRU until naturally
        evicted.  Pass ``0`` or ``float("inf")`` to disable TTL.

    Example
    -------
    .. code-block:: python

        cache = TraceCache()
        key = build_schema_cache_key(schema)

        cached = cache.get(key)
        if cached is not None:
            think_prompt += f"\\n\\n[Prior reasoning scaffold]\\n{cached}"
        else:
            # run Pass 1 normally
            raw_thinking = await backend.generate(think_prompt, ...)
            thinking_text = extract_thinking(raw_thinking)
            cache.put(key, thinking_text)
    """

    def __init__(
        self,
        max_size: int = DEFAULT_MAX_SIZE,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
    ) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be ≥ 1, got {max_size}")
        self._max_size = max_size
        self._ttl = ttl_seconds if ttl_seconds > 0 else float("inf")
        self._store: collections.OrderedDict[str, _CacheEntry] = collections.OrderedDict()
        self._lock = threading.Lock()
        self._total_hits = 0
        self._total_misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> str | None:
        """Return cached trace for *key*, or ``None`` on miss / stale entry.

        A hit moves the entry to the most-recently-used position.

        Parameters
        ----------
        key:
            Schema-structure key from :func:`build_schema_cache_key`.

        Returns
        -------
        str | None
            Cached reasoning trace text, or ``None``.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._total_misses += 1
                return None

            age = time.monotonic() - entry.created_at
            if age > self._ttl:
                # Stale — remove and report miss
                del self._store[key]
                self._total_misses += 1
                logger.debug("TraceCache: stale entry evicted (age=%.1fs key=%s)", age, key)
                return None

            # Move to end (most recently used)
            self._store.move_to_end(key)
            entry.hit_count += 1
            self._total_hits += 1
            logger.debug(
                "TraceCache: HIT key=%s hits=%d age=%.1fs",
                key,
                entry.hit_count,
                age,
            )
            return entry.trace

    def put(self, key: str, trace: str) -> None:
        """Store *trace* under *key*.

        If the cache is full the LRU entry is evicted first.

        Parameters
        ----------
        key:
            Schema-structure key from :func:`build_schema_cache_key`.
        trace:
            Pass 1 reasoning text to cache.
        """
        if not trace or not trace.strip():
            return  # never cache empty traces

        with self._lock:
            if key in self._store:
                # Update existing entry in-place and refresh LRU position
                self._store[key].trace = trace
                self._store[key].created_at = time.monotonic()
                self._store.move_to_end(key)
                return

            if len(self._store) >= self._max_size:
                evicted_key, _ = self._store.popitem(last=False)
                logger.debug("TraceCache: LRU eviction key=%s", evicted_key)

            self._store[key] = _CacheEntry(trace)
            logger.debug("TraceCache: stored key=%s size=%d", key, len(self._store))

    def invalidate(self, key: str) -> bool:
        """Remove a specific entry.  Returns ``True`` if it existed.

        Parameters
        ----------
        key:
            Schema-structure key to remove.
        """
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        """Remove all cached entries and reset statistics."""
        with self._lock:
            self._store.clear()
            self._total_hits = 0
            self._total_misses = 0

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        with self._lock:
            return len(self._store)

    @property
    def total_hits(self) -> int:
        """Cumulative cache hits since creation or last :meth:`clear`."""
        return self._total_hits

    @property
    def total_misses(self) -> int:
        """Cumulative cache misses since creation or last :meth:`clear`."""
        return self._total_misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate ∈ [0, 1].  Returns 0.0 if no requests yet."""
        total = self._total_hits + self._total_misses
        return self._total_hits / total if total > 0 else 0.0

    def stats(self) -> dict[str, Any]:
        """Return a stats snapshot dict for logging / observability."""
        return {
            "size": self.size,
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
            "total_hits": self._total_hits,
            "total_misses": self._total_misses,
            "hit_rate": round(self.hit_rate, 4),
        }
