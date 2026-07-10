"""Cache introspection: hit/miss counters and size reporting on both backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nfield.providers._cache import DiskCache, MemoryCache

if TYPE_CHECKING:
    from pathlib import Path


class TestMemoryCacheStats:
    def test_starts_at_zero(self) -> None:
        assert MemoryCache().stats() == {"hits": 0, "misses": 0, "entries": 0}

    def test_counts_hits_misses_and_entries(self) -> None:
        cache = MemoryCache()
        assert cache.get("a") is None  # miss
        cache.set("a", "value")
        assert cache.get("a") == "value"  # hit
        assert cache.get("b") is None  # miss
        assert cache.stats() == {"hits": 1, "misses": 2, "entries": 1}

    def test_clear_keeps_the_counters(self) -> None:
        cache = MemoryCache()
        cache.set("a", "value")
        cache.get("a")
        cache.clear()
        stats = cache.stats()
        assert stats["entries"] == 0
        assert stats["hits"] == 1  # history survives; only entries are dropped


class TestDiskCacheStats:
    def test_counts_hits_misses_entries_and_bytes(self, tmp_path: Path) -> None:
        cache = DiskCache(tmp_path)
        key = "a" * 64
        assert cache.get(key) is None  # miss
        cache.set(key, "hello")
        assert cache.get(key) == "hello"  # hit
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entries"] == 1
        assert stats["size_bytes"] == len(b"hello")

    def test_entries_reflect_other_writers(self, tmp_path: Path) -> None:
        # entries/bytes come from the directory, so a second instance sees them.
        DiskCache(tmp_path).set("b" * 64, "shared")
        stats = DiskCache(tmp_path).stats()
        assert stats["entries"] == 1
        assert stats["hits"] == 0  # counters are per-instance


class TestDiskCacheSetIsBestEffort:
    """set() never raises into the caller - a failed write is a future miss."""

    def test_same_key_hammer_never_raises(self, tmp_path: Path) -> None:
        # On Windows, os.replace onto a file another thread is reading raises
        # PermissionError; the cache must absorb it (the caller already paid the
        # API call for this value).
        import threading

        cache = DiskCache(tmp_path)
        key = "c" * 64
        errors: list[BaseException] = []

        def hammer() -> None:
            try:
                for _ in range(200):
                    cache.set(key, "value")
                    cache.get(key)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=hammer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert cache.get(key) == "value"  # the entry still lands

    def test_unwritable_directory_is_swallowed(self, tmp_path: Path) -> None:
        cache = DiskCache(tmp_path)
        # Simulate a write failure by removing the directory out from under it.
        import shutil

        shutil.rmtree(tmp_path)
        cache.set("d" * 64, "value")  # must not raise
        assert cache.get("d" * 64) is None  # simply a miss afterwards
