"""Tests for ``from_model`` resolution and the public import surface.

These confirm the provider factory routes by prefix without importing any
optional SDK, and that the documented MVP names are importable from the
top-level package.
"""

from __future__ import annotations

import sys

import pytest

from nfield import from_model
from nfield.exceptions import ProviderError


class TestFromModel:
    def test_groq_prefix_returns_provider(self):
        provider = from_model("groq/llama-3.1-8b")
        assert provider.model_name == "llama-3.1-8b"
        assert provider.context_window > 0
        assert provider.max_output_tokens > 0

    def test_groq_construction_does_not_import_sdk(self):
        # The groq SDK must be imported lazily (inside the client), never at
        # provider construction. This keeps `import nfield` dependency-free.
        sys.modules.pop("groq", None)
        from_model("groq/llama-3.1-8b")
        assert "groq" not in sys.modules

    def test_unknown_prefix_raises_provider_error(self):
        with pytest.raises(ProviderError):
            from_model("nope/some-model")

    def test_malformed_model_string_raises(self):
        with pytest.raises(ValueError, match="provider/model-name"):
            from_model("no-slash-here")

    def test_caller_supplies_model_specs(self):
        # No specs → provider's conservative default.
        default = from_model("groq/llama-3.1-8b-instant")
        assert default.context_window == 8192

        # The caller supplies the model's real window/output (per ModelSpec).
        provider = from_model(
            "groq/llama-3.1-8b-instant", context_window=131_072, max_output_tokens=32_768
        )
        assert provider.context_window == 131_072
        assert provider.max_output_tokens == 32_768

    def test_partial_spec_keeps_default_for_the_other(self):
        provider = from_model("groq/llama-3.1-8b-instant", context_window=131_072)
        assert provider.context_window == 131_072
        assert provider.max_output_tokens == 8192  # untouched default


class TestPublicImports:
    def test_mvp_surface_is_importable(self):
        import nfield as fs

        for name in (
            "nfield",
            "nfield_async",
            "NField",
            "AsyncNField",
            "from_model",
            "ExtractionConfig",
            "ExtractionResult",
            "NFieldError",
            "__version__",
        ):
            assert hasattr(fs, name), f"missing public name: {name}"
