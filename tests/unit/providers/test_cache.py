"""Tests for providers._cache - exact-match response caching.

The correctness contract of an exact-match cache is twofold: an identical request
must hit (reuse), and any change to the request must miss (never serve a different
request's answer). Exact-match trades a lower hit rate for that guarantee, unlike
semantic caches (GPTCache). The suite pins both directions plus the production
concerns: bounded memory, persistence, atomicity, fail-open reads, version
invalidation, and thread safety.
"""

from __future__ import annotations

import threading

import pytest

from nfield.providers._cache import (
    DiskCache,
    MemoryCache,
    ResponseCache,
    make_cache_key,
    resolve_cache,
)

_MESSAGES: list[dict[str, str]] = [
    {"role": "system", "content": "Extract fields."},
    {"role": "user", "content": "INVOICE 42"},
]


# ---------------------------------------------------------------------------
# make_cache_key
# ---------------------------------------------------------------------------


class TestCacheKey:
    """The key must be stable for an identical request and sensitive to any change."""

    def test_identical_request_same_key(self) -> None:
        assert make_cache_key("m", _MESSAGES, 128) == make_cache_key("m", _MESSAGES, 128)

    def test_model_change_changes_key(self) -> None:
        assert make_cache_key("m1", _MESSAGES, 128) != make_cache_key("m2", _MESSAGES, 128)

    def test_max_tokens_change_changes_key(self) -> None:
        assert make_cache_key("m", _MESSAGES, 128) != make_cache_key("m", _MESSAGES, 256)

    def test_message_content_change_changes_key(self) -> None:
        other = [{"role": "user", "content": "INVOICE 43"}]
        assert make_cache_key("m", _MESSAGES, 128) != make_cache_key("m", other, 128)

    def test_message_order_changes_key(self) -> None:
        reversed_messages = list(reversed(_MESSAGES))
        assert make_cache_key("m", _MESSAGES, 128) != make_cache_key("m", reversed_messages, 128)

    def test_dict_key_order_is_irrelevant(self) -> None:
        a = [{"role": "user", "content": "hi"}]
        b = [{"content": "hi", "role": "user"}]
        assert make_cache_key("m", a, 128) == make_cache_key("m", b, 128)

    def test_key_is_hex_digest(self) -> None:
        key = make_cache_key("m", _MESSAGES, 128)
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_version_bump_invalidates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        before = make_cache_key("m", _MESSAGES, 128)
        monkeypatch.setattr("nfield.providers._cache._CACHE_VERSION", 999)
        assert make_cache_key("m", _MESSAGES, 128) != before


# ---------------------------------------------------------------------------
# MemoryCache
# ---------------------------------------------------------------------------


class TestMemoryCache:
    """In-process LRU: hit/miss, bounded size, eviction order, clear, validation."""

    def test_miss_returns_none(self) -> None:
        assert MemoryCache().get("absent") is None

    def test_set_then_get(self) -> None:
        cache = MemoryCache()
        cache.set("k", "v")
        assert cache.get("k") == "v"

    def test_overwrite_replaces_value(self) -> None:
        cache = MemoryCache()
        cache.set("k", "v1")
        cache.set("k", "v2")
        assert cache.get("k") == "v2"

    def test_evicts_least_recently_used(self) -> None:
        cache = MemoryCache(max_size=2)
        cache.set("a", "1")
        cache.set("b", "2")
        cache.set("c", "3")  # "a" is the oldest and is evicted
        assert cache.get("a") is None
        assert cache.get("b") == "2"
        assert cache.get("c") == "3"

    def test_get_refreshes_recency(self) -> None:
        cache = MemoryCache(max_size=2)
        cache.set("a", "1")
        cache.set("b", "2")
        cache.get("a")  # "a" is now most-recently-used, so "b" is evicted next
        cache.set("c", "3")
        assert cache.get("a") == "1"
        assert cache.get("b") is None

    def test_clear_empties(self) -> None:
        cache = MemoryCache()
        cache.set("k", "v")
        cache.clear()
        assert cache.get("k") is None

    def test_non_positive_size_raises(self) -> None:
        with pytest.raises(ValueError, match="max_size must be > 0"):
            MemoryCache(max_size=0)

    def test_is_response_cache(self) -> None:
        assert isinstance(MemoryCache(), ResponseCache)

    def test_thread_safe_under_concurrent_writes(self) -> None:
        cache = MemoryCache(max_size=1000)

        def worker(start: int) -> None:
            for i in range(start, start + 100):
                cache.set(f"k{i}", f"v{i}")

        threads = [threading.Thread(target=worker, args=(base * 100,)) for base in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert cache.get("k0") == "v0"
        assert cache.get("k799") == "v799"


# ---------------------------------------------------------------------------
# DiskCache
# ---------------------------------------------------------------------------


class TestDiskCache:
    """On-disk: persistence, atomic writes, fail-open reads, clear."""

    def test_miss_returns_none(self, tmp_path) -> None:
        assert DiskCache(tmp_path).get("absent") is None

    def test_set_then_get(self, tmp_path) -> None:
        cache = DiskCache(tmp_path)
        cache.set("k", "value with\nnewlines")
        assert cache.get("k") == "value with\nnewlines"

    def test_persists_across_instances(self, tmp_path) -> None:
        DiskCache(tmp_path).set("k", "v")
        # A fresh instance on the same directory reads the stored entry.
        assert DiskCache(tmp_path).get("k") == "v"

    def test_creates_directory(self, tmp_path) -> None:
        nested = tmp_path / "a" / "b" / "cache"
        DiskCache(nested).set("k", "v")
        assert nested.is_dir()

    def test_write_leaves_no_temp_files(self, tmp_path) -> None:
        cache = DiskCache(tmp_path)
        cache.set("k", "v")
        assert list(tmp_path.glob("*.tmp")) == []

    def test_overwrite_replaces_value(self, tmp_path) -> None:
        cache = DiskCache(tmp_path)
        cache.set("k", "v1")
        cache.set("k", "v2")
        assert cache.get("k") == "v2"

    def test_unicode_roundtrip(self, tmp_path) -> None:
        cache = DiskCache(tmp_path)
        cache.set("k", "café naïve 日本語 €")
        assert cache.get("k") == "café naïve 日本語 €"

    def test_corrupt_entry_reads_as_miss(self, tmp_path) -> None:
        cache = DiskCache(tmp_path)
        # A non-UTF-8 file at the key path must fail open (recompute), not raise.
        cache._path("k").write_bytes(b"\xff\xfe\x00bad")
        assert cache.get("k") is None

    def test_rejects_traversing_key_on_set(self, tmp_path) -> None:
        cache = DiskCache(tmp_path / "cache")
        escaped = tmp_path / "ESCAPED.txt"
        with pytest.raises(ValueError, match="inside the cache directory"):
            cache.set("../ESCAPED", "pwned")
        assert not escaped.exists()

    def test_rejects_traversing_key_on_get(self, tmp_path) -> None:
        cache = DiskCache(tmp_path / "cache")
        secret = tmp_path / "secret.txt"
        secret.write_text("top secret", encoding="utf-8")
        with pytest.raises(ValueError, match="inside the cache directory"):
            cache.get("../secret")

    def test_hex_key_is_accepted(self, tmp_path) -> None:
        cache = DiskCache(tmp_path)
        key = make_cache_key("m", _MESSAGES, 128)
        cache.set(key, "ok")
        assert cache.get(key) == "ok"

    def test_clear_deletes_entries(self, tmp_path) -> None:
        cache = DiskCache(tmp_path)
        cache.set("a", "1")
        cache.set("b", "2")
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_is_response_cache(self, tmp_path) -> None:
        assert isinstance(DiskCache(tmp_path), ResponseCache)

    def test_thread_safe_across_distinct_keys(self, tmp_path) -> None:
        cache = DiskCache(tmp_path)

        def worker(start: int) -> None:
            for i in range(start, start + 50):
                cache.set(f"k{i}", f"v{i}")

        threads = [threading.Thread(target=worker, args=(base * 50,)) for base in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert cache.get("k0") == "v0"
        assert cache.get("k299") == "v299"


# ---------------------------------------------------------------------------
# resolve_cache
# ---------------------------------------------------------------------------


class TestResolveCache:
    """Maps a config spec to a cache instance or None."""

    def test_false_disables(self) -> None:
        assert resolve_cache(False) is None

    def test_true_builds_memory_cache(self) -> None:
        assert isinstance(resolve_cache(True), MemoryCache)

    def test_instance_passthrough(self, tmp_path) -> None:
        given = DiskCache(tmp_path)
        assert resolve_cache(given) is given
