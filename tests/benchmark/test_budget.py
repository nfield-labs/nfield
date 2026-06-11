"""Budget-profile tests — native limits from the registry, constrained constants."""

from __future__ import annotations

import pytest

from benchmark.budget import BUDGET_MODES, resolve_budget
from benchmark.models import native_limits


def test_native_limits_strip_provider_prefix():
    with_prefix = native_limits("groq/llama-3.3-70b-versatile")
    bare = native_limits("llama-3.3-70b-versatile")
    assert with_prefix == bare
    assert with_prefix.context_window == 131_072
    assert with_prefix.max_output_tokens == 32_768
    # The reliable single-call output is below the published ceiling (120s wall).
    assert with_prefix.reliable_output_tokens == 24_000
    assert with_prefix.reliable_output_tokens < with_prefix.max_output_tokens


def test_native_limits_unknown_model_raises():
    with pytest.raises(KeyError, match="no native limits registered"):
        native_limits("nonexistent-model")


def test_resolve_native_uses_reliable_output_not_published_max():
    model = "groq/llama-3.3-70b-versatile"
    limits = native_limits(model)
    budget = resolve_budget("native", model)
    # Native context is the model's real window, but output is the RELIABLE ceiling
    # (the published 32k max deterministically 502s past the ~120s wall).
    assert budget.context_window == limits.context_window
    assert budget.max_output_tokens == limits.reliable_output_tokens == 24_000


def test_resolve_constrained_is_fixed_and_model_independent():
    a = resolve_budget("constrained", "groq/llama-3.3-70b-versatile")
    b = resolve_budget("constrained", "some-other-model")  # not in the registry, still fine
    assert a == b
    assert a.context_window == 40_000
    assert a.max_output_tokens == 8_000


def test_budget_modes_are_the_two_we_expose():
    assert set(BUDGET_MODES) == {"native", "constrained"}


def test_resolve_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown budget mode"):
        resolve_budget("turbo", "groq/llama-3.3-70b-versatile")  # type: ignore[arg-type]
