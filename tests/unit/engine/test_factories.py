"""Tests for ``from_model`` resolution and the public import surface.

These confirm the provider factory routes by prefix without importing any
optional SDK, and that the documented MVP names are importable from the
top-level package.
"""

from __future__ import annotations

import sys

import pytest

from formatshield import from_model
from formatshield.exceptions import ProviderError


class TestFromModel:
    def test_groq_prefix_returns_provider(self):
        provider = from_model("groq/llama-3.1-8b")
        assert provider.model_name == "llama-3.1-8b"
        assert provider.context_window > 0
        assert provider.max_output_tokens > 0

    def test_groq_construction_does_not_import_sdk(self):
        # The groq SDK must be imported lazily (inside the client), never at
        # provider construction. This keeps `import formatshield` dependency-free.
        sys.modules.pop("groq", None)
        from_model("groq/llama-3.1-8b")
        assert "groq" not in sys.modules

    def test_unknown_prefix_raises_provider_error(self):
        with pytest.raises(ProviderError):
            from_model("nope/some-model")

    def test_malformed_model_string_raises(self):
        with pytest.raises(ValueError, match="provider/model-name"):
            from_model("no-slash-here")


class TestPublicImports:
    def test_mvp_surface_is_importable(self):
        import formatshield as fs

        for name in (
            "nfield",
            "nfield_async",
            "FormatShield",
            "AsyncFormatShield",
            "from_model",
            "ExtractionConfig",
            "ExtractionResult",
            "FormatShieldError",
            "__version__",
        ):
            assert hasattr(fs, name), f"missing public name: {name}"
