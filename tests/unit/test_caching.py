"""Unit tests for formatshield.caching — no API keys required."""

from __future__ import annotations

import os
import tempfile

import pytest

from formatshield.caching import (
    _DISKCACHE_AVAILABLE,
    _reset_cache_instance,
    cache,
    clear_cache,
    disable_cache,
    make_cache_key,
)

# ---------------------------------------------------------------------------
# make_cache_key
# ---------------------------------------------------------------------------


def test_make_cache_key_returns_64_char_hex() -> None:
    key = make_cache_key("a", "b", x=1)
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)


def test_make_cache_key_deterministic() -> None:
    k1 = make_cache_key("groq/llama", "hello", schema={"type": "string"})
    k2 = make_cache_key("groq/llama", "hello", schema={"type": "string"})
    assert k1 == k2


def test_make_cache_key_different_for_different_args() -> None:
    k1 = make_cache_key("hello")
    k2 = make_cache_key("world")
    assert k1 != k2


def test_make_cache_key_kwargs_order_independent() -> None:
    k1 = make_cache_key(a=1, b=2)
    k2 = make_cache_key(b=2, a=1)
    assert k1 == k2


def test_make_cache_key_handles_non_json_serializable() -> None:
    """Non-JSON-serializable values fall back to str() representation."""

    class Unserializable:
        def __repr__(self) -> str:
            return "Unserializable()"

    key = make_cache_key(Unserializable())
    assert len(key) == 64


def test_make_cache_key_dict_key_order_normalized() -> None:
    """Nested dicts should produce the same key regardless of insertion order."""
    k1 = make_cache_key({"z": 1, "a": 2})
    k2 = make_cache_key({"a": 2, "z": 1})
    assert k1 == k2


# ---------------------------------------------------------------------------
# disable_cache context manager
# ---------------------------------------------------------------------------


def test_disable_cache_bypasses_caching() -> None:
    call_count = 0

    @cache()
    def counted(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 2

    with disable_cache():
        counted(1)
        counted(1)
        counted(1)

    # All three calls should have gone through (no caching)
    assert call_count == 3


def test_disable_cache_restores_on_exit() -> None:
    from formatshield import caching

    assert caching._caching_enabled is True
    with disable_cache():
        assert caching._caching_enabled is False
    assert caching._caching_enabled is True


def test_disable_cache_restores_on_exception() -> None:
    from formatshield import caching

    try:
        with disable_cache():
            raise ValueError("test error")
    except ValueError:
        pass
    assert caching._caching_enabled is True


# ---------------------------------------------------------------------------
# @cache decorator — basic behaviour
# ---------------------------------------------------------------------------


def test_cache_decorator_calls_function_once_for_same_args() -> None:
    if not _DISKCACHE_AVAILABLE:
        pytest.skip("diskcache not available")

    _reset_cache_instance()

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["FORMATSHIELD_CACHE_DIR"] = tmpdir
        try:
            call_count = 0

            @cache()
            def expensive(x: int) -> int:
                nonlocal call_count
                call_count += 1
                return x * 10

            result1 = expensive(5)
            result2 = expensive(5)
            assert result1 == 50
            assert result2 == 50
            # Second call hits the cache — function body executes once
            assert call_count == 1
        finally:
            del os.environ["FORMATSHIELD_CACHE_DIR"]
            _reset_cache_instance()


def test_cache_decorator_different_args_call_separately() -> None:
    if not _DISKCACHE_AVAILABLE:
        pytest.skip("diskcache not available")

    _reset_cache_instance()

    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["FORMATSHIELD_CACHE_DIR"] = tmpdir
        try:
            call_count = 0

            @cache()
            def fn(x: int) -> int:
                nonlocal call_count
                call_count += 1
                return x

            fn(1)
            fn(2)
            fn(3)
            assert call_count == 3
        finally:
            del os.environ["FORMATSHIELD_CACHE_DIR"]
            _reset_cache_instance()


def test_cache_decorator_falls_back_gracefully_when_unavailable() -> None:
    """When diskcache is not installed, @cache should still call the function."""
    import formatshield.caching as caching_module

    orig = caching_module._DISKCACHE_AVAILABLE
    caching_module._DISKCACHE_AVAILABLE = False
    try:
        call_count = 0

        @cache()
        def fn() -> int:
            nonlocal call_count
            call_count += 1
            return 42

        result = fn()
        assert result == 42
        assert call_count == 1
    finally:
        caching_module._DISKCACHE_AVAILABLE = orig


# ---------------------------------------------------------------------------
# clear_cache
# ---------------------------------------------------------------------------


def test_clear_cache_works_when_available() -> None:
    if not _DISKCACHE_AVAILABLE:
        pytest.skip("diskcache not available")

    _reset_cache_instance()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["FORMATSHIELD_CACHE_DIR"] = tmpdir
        try:
            clear_cache()  # Should not raise
        finally:
            del os.environ["FORMATSHIELD_CACHE_DIR"]
            _reset_cache_instance()


def test_clear_cache_raises_without_diskcache() -> None:
    import formatshield.caching as caching_module

    orig = caching_module._DISKCACHE_AVAILABLE
    caching_module._DISKCACHE_AVAILABLE = False
    _reset_cache_instance()
    try:
        with pytest.raises(ImportError, match="diskcache"):
            clear_cache()
    finally:
        caching_module._DISKCACHE_AVAILABLE = orig


# ---------------------------------------------------------------------------
# FORMATSHIELD_CACHE_DIR env var
# ---------------------------------------------------------------------------


def test_default_cache_dir_uses_env_var() -> None:
    from formatshield.caching import _default_cache_dir

    os.environ["FORMATSHIELD_CACHE_DIR"] = "/tmp/my_cache"
    try:
        result = _default_cache_dir()
        assert result == "/tmp/my_cache"
    finally:
        del os.environ["FORMATSHIELD_CACHE_DIR"]


def test_default_cache_dir_falls_back_to_home() -> None:
    from formatshield.caching import _default_cache_dir

    for var in ["FORMATSHIELD_CACHE_DIR", "XDG_CACHE_HOME"]:
        os.environ.pop(var, None)

    result = _default_cache_dir()
    assert "formatshield" in result
    assert result.endswith("formatshield")


def test_default_cache_dir_uses_xdg() -> None:
    from formatshield.caching import _default_cache_dir

    os.environ.pop("FORMATSHIELD_CACHE_DIR", None)
    os.environ["XDG_CACHE_HOME"] = "/tmp/xdg"
    try:
        result = _default_cache_dir()
        # Use os.path.join to be cross-platform (Windows uses backslash)
        expected = os.path.join("/tmp/xdg", "formatshield")
        assert result == expected
    finally:
        del os.environ["XDG_CACHE_HOME"]
