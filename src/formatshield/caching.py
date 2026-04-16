"""
FormatShield caching layer — persistent disk cache for oracle decisions and LLM responses.

Uses diskcache (CloudpickleDisk for sklearn-compatible serialization) with optional
zlib compression. The cache version-stamps itself — any package version bump auto-clears
stale entries so users never see serialization mismatches after upgrades.

Environment::

    FORMATSHIELD_CACHE_DIR   Override the default cache directory.
                             Default: ~/.cache/formatshield

Usage::

    from formatshield.caching import cache, clear_cache, disable_cache

    @cache(expire=3600)
    def expensive_call(prompt: str, schema: dict) -> str:
        ...

    # Clear everything:
    clear_cache()

    # Temporarily disable (e.g., in tests):
    with disable_cache():
        result = expensive_call(...)
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import logging
import os
import zlib
from collections.abc import Callable, Generator
from typing import Any, TypeVar

from formatshield._version import __version__

logger = logging.getLogger(__name__)

_F = TypeVar("_F", bound=Callable[..., Any])

# Module-level cache singleton — reset with _reset_cache_instance() in tests
_cache_instance: Any | None = None
_caching_enabled: bool = True

# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

_CACHE_DIR_ENV = "FORMATSHIELD_CACHE_DIR"
_CACHE_VERSION_KEY = "__fs_version__"

try:
    import importlib.util as _util

    _DISKCACHE_AVAILABLE = bool(_util.find_spec("diskcache") and _util.find_spec("cloudpickle"))
except Exception:
    _DISKCACHE_AVAILABLE = False


def _default_cache_dir() -> str:
    """Resolve the default cache directory.

    Priority order:

    1. ``FORMATSHIELD_CACHE_DIR`` env var
    2. ``$XDG_CACHE_HOME/formatshield``
    3. ``~/.cache/formatshield``
    """
    if env_val := os.environ.get(_CACHE_DIR_ENV):
        return env_val
    xdg = os.environ.get("XDG_CACHE_HOME", "")
    if xdg:
        return os.path.join(xdg, "formatshield")
    return os.path.expanduser("~/.cache/formatshield")


# ---------------------------------------------------------------------------
# CloudpickleDisk — built lazily so the class only exists when diskcache is present
# ---------------------------------------------------------------------------


def _make_cloudpickle_disk_class() -> type:
    """Factory: build the CloudpickleDisk subclass only when diskcache is importable.

    Returns:
        A ``diskcache.Disk`` subclass that serializes with cloudpickle + zlib.

    Raises:
        ImportError: If diskcache or cloudpickle are not installed.
    """
    try:
        import cloudpickle as cp  # type: ignore[import-untyped]
        import diskcache  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "diskcache and cloudpickle are required for FormatShield caching. "
            "Install with: pip install formatshield[cache]"
        ) from exc

    _compress_level = 1

    class CloudpickleDisk(diskcache.Disk):
        """diskcache.Disk with cloudpickle serialization and zlib compression."""

        def put(self, key: Any) -> Any:  # type: ignore[override]
            return cp.dumps(key)

        def get(self, key: Any, raw: bool) -> Any:  # type: ignore[override]
            return cp.loads(key)  # type: ignore[arg-type]

        def store(self, value: Any, read: bool, key: Any = diskcache.Disk.Unknown) -> Any:  # type: ignore[override]
            if not read:
                serialized: bytes = cp.dumps(value)
                if _compress_level > 0:
                    serialized = zlib.compress(serialized, _compress_level)
                return super().store(serialized, read, key=key)
            return super().store(value, read, key=key)

        def fetch(self, mode: Any, filename: str | None, value: Any, read: bool) -> Any:  # type: ignore[override]
            fetched = super().fetch(mode, filename, value, read)
            if not read:
                as_bytes: bytes = fetched  # type: ignore[assignment]
                decompressed = zlib.decompress(as_bytes) if _compress_level > 0 else as_bytes
                return cp.loads(decompressed)  # type: ignore[arg-type]
            return fetched

    return CloudpickleDisk


# ---------------------------------------------------------------------------
# Cache singleton management
# ---------------------------------------------------------------------------


def get_cache() -> Any:
    """Return the singleton diskcache.Cache instance.

    Creates the cache on first call. Auto-clears if the FormatShield version
    changed (prevents deserialization errors after upgrades).

    Returns:
        A ``diskcache.Cache`` instance backed by ``CloudpickleDisk``.

    Raises:
        ImportError: If diskcache or cloudpickle are not installed.
    """
    global _cache_instance

    if not _DISKCACHE_AVAILABLE:
        raise ImportError(
            "diskcache and cloudpickle are required. Install with: pip install formatshield[cache]"
        )

    if _cache_instance is None:
        import diskcache  # type: ignore[import-untyped]

        cloudpickle_disk = _make_cloudpickle_disk_class()
        cache_dir = _default_cache_dir()
        _cache_instance = diskcache.Cache(
            directory=cache_dir,
            disk=cloudpickle_disk,
            size_limit=2**30,  # 1 GiB
        )
        # Version stamp — clear on package upgrade to avoid deserialization mismatches
        if _cache_instance.get(_CACHE_VERSION_KEY) != __version__:
            _cache_instance.clear()
            _cache_instance[_CACHE_VERSION_KEY] = __version__
            logger.debug("FormatShield cache cleared (version bump → %s)", __version__)

    return _cache_instance


def _reset_cache_instance() -> None:
    """Reset the cache singleton. For use in tests only."""
    global _cache_instance
    _cache_instance = None


def clear_cache() -> None:
    """Delete all cached entries.

    Removes every key/value stored under the current cache directory,
    then re-stamps the version key so subsequent reads start fresh.

    Raises:
        ImportError: If diskcache or cloudpickle are not installed.
    """
    disk_cache = get_cache()
    disk_cache.clear()
    disk_cache[_CACHE_VERSION_KEY] = __version__
    logger.info("FormatShield cache cleared")


@contextlib.contextmanager
def disable_cache() -> Generator[None, None, None]:
    """Context manager — disables caching for the duration of the block.

    Useful in tests and for benchmarking real latency without cache hits.

    Example::

        with disable_cache():
            result = my_cached_function(prompt)  # goes to real implementation
    """
    global _caching_enabled
    _caching_enabled = False
    try:
        yield
    finally:
        _caching_enabled = True


# ---------------------------------------------------------------------------
# Cache key construction
# ---------------------------------------------------------------------------


def make_cache_key(*args: Any, **kwargs: Any) -> str:
    """Build a deterministic SHA-256 cache key from arbitrary arguments.

    The key is derived from the JSON-serialized representation of ``args``
    and ``kwargs``. Non-JSON-serializable values (e.g., Pydantic models,
    dataclasses) are converted via their ``str()`` representation.

    Args:
        *args: Positional values contributing to the key.
        **kwargs: Keyword values contributing to the key.

    Returns:
        64-character lowercase hex string (SHA-256 digest).

    Example::

        key = make_cache_key("groq/llama-3.3-70b", prompt, schema)
        assert len(key) == 64
    """

    def _serialize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in sorted(obj.items())}
        if isinstance(obj, (list, tuple)):
            return [_serialize(v) for v in obj]
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)

    payload = {"args": [_serialize(a) for a in args], "kwargs": _serialize(kwargs)}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# @cache decorator
# ---------------------------------------------------------------------------


def cache(
    expire: float | None = None,
    typed: bool = False,
    ignore: tuple[str, ...] = (),
) -> Callable[[_F], _F]:
    """Decorator — cache function results on disk via diskcache.

    Applies no caching when ``_caching_enabled`` is ``False`` (inside
    :func:`disable_cache`) or when diskcache is not installed (warns once,
    then falls back to calling the function directly).

    Args:
        expire: TTL in seconds. ``None`` = never expire.
        typed: If ``True``, ``fn(1)`` and ``fn(1.0)`` are distinct cache entries.
        ignore: Parameter names to exclude from the cache key.

    Returns:
        Decorated function. Adds ``.cache_clear()`` and ``.cache_info()`` helpers.

    Example::

        @cache(expire=3600)
        def oracle_decision(prompt: str, schema: str, backend: str) -> str:
            return expensive_oracle(prompt, schema, backend)
    """

    def decorator(fn: _F) -> _F:
        _warned: list[bool] = [False]

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _caching_enabled:
                return fn(*args, **kwargs)

            if not _DISKCACHE_AVAILABLE:
                if not _warned[0]:
                    logger.warning(
                        "FormatShield caching disabled — diskcache/cloudpickle not installed. "
                        "Install with: pip install formatshield[cache]"
                    )
                    _warned[0] = True
                return fn(*args, **kwargs)

            disk_cache = get_cache()

            # Exclude ignored params from key
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ignore}

            # Use make_cache_key for a stable SHA-256 key
            key = make_cache_key(fn.__module__, fn.__qualname__, *args, **filtered_kwargs)

            result = disk_cache.get(key)
            if result is None:
                result = fn(*args, **kwargs)
                disk_cache.set(key, result, expire=expire)

            return result

        def cache_clear() -> None:
            """Remove all entries for this function from the global cache."""
            if not _DISKCACHE_AVAILABLE:
                return
            disk_cache = get_cache()
            prefix = make_cache_key(fn.__module__, fn.__qualname__)[:16]
            # Evict keys whose prefix matches this function
            for disk_key in list(disk_cache.iterkeys()):
                try:
                    full_key = str(disk_key)
                    if prefix in full_key:
                        del disk_cache[disk_key]
                except (KeyError, TypeError):
                    pass

        wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]
        wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
        wrapper.__memory__ = lambda: get_cache()  # type: ignore[attr-defined]

        return wrapper  # type: ignore[return-value]

    return decorator
