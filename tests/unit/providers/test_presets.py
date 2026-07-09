"""Tests for providers._presets - OpenAI-compatible preset routing."""

from __future__ import annotations

import pytest

from nfield.exceptions import ProviderError
from nfield.providers._presets import (
    _LOCAL_PLACEHOLDER_KEY,
    OPENAI_COMPATIBLE_PRESETS,
    build_preset_provider,
)
from nfield.providers._registry import from_model
from nfield.providers.openai import OpenAIProvider

ALL_PREFIXES = sorted(OPENAI_COMPATIBLE_PRESETS)


class TestPresetTable:
    """The preset table is well-formed."""

    def test_expected_presets_present(self) -> None:
        assert set(OPENAI_COMPATIBLE_PRESETS) == {
            "openrouter",
            "deepseek",
            "together",
            "fireworks",
            "mistral",
            "xai",
            "perplexity",
            "cerebras",
            "ollama",
        }

    @pytest.mark.parametrize("prefix", ALL_PREFIXES)
    def test_base_url_is_https_or_localhost(self, prefix: str) -> None:
        base, _env = OPENAI_COMPATIBLE_PRESETS[prefix]
        assert base.startswith("https://") or base.startswith("http://localhost")


class TestPresetRouting:
    """Each preset routes through the factory to a configured OpenAIProvider."""

    @pytest.mark.parametrize("prefix", ALL_PREFIXES)
    def test_prefix_routes_to_openai_provider_with_preset_base(self, prefix: str) -> None:
        base, _env = OPENAI_COMPATIBLE_PRESETS[prefix]
        provider = from_model(f"{prefix}/vendor/model-x", api_key="explicit")
        assert isinstance(provider, OpenAIProvider)
        assert provider._base_url == base
        # First slash only: the vendor/model form is preserved as the model name.
        assert provider.model_name == "vendor/model-x"

    def test_explicit_api_key_wins(self) -> None:
        provider = from_model("deepseek/deepseek-chat", api_key="dk-explicit")
        assert provider._api_key == "dk-explicit"

    def test_base_url_override_wins(self) -> None:
        provider = from_model("together/x", api_key="k", base_url="https://proxy/v1")
        assert provider._base_url == "https://proxy/v1"

    @pytest.mark.parametrize(
        "prefix,env_var",
        [(p, e) for p, (_b, e) in OPENAI_COMPATIBLE_PRESETS.items() if e is not None],
    )
    def test_env_var_is_read(self, prefix: str, env_var: str, monkeypatch) -> None:
        monkeypatch.setenv(env_var, f"key-for-{prefix}")
        provider = from_model(f"{prefix}/some-model")
        assert provider._api_key == f"key-for-{prefix}"

    def test_ollama_uses_placeholder_key_without_env(self, monkeypatch) -> None:
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        provider = from_model("ollama/llama3.2")
        # Local server needs a non-empty key it ignores.
        assert provider._api_key == _LOCAL_PLACEHOLDER_KEY


class TestBuildPresetProvider:
    """The builder resolves key and base URL with the right precedence."""

    def test_missing_env_key_leaves_none(self, monkeypatch) -> None:
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        provider = build_preset_provider("xai", "grok-2")
        assert provider._api_key is None

    def test_unknown_prefix_still_lists_presets(self) -> None:
        with pytest.raises(ProviderError) as info:
            from_model("nope/model")
        message = str(info.value)
        assert "deepseek" in message and "openrouter" in message
