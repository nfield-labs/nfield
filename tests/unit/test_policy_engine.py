"""Unit tests for the built-in default policy engine."""

from __future__ import annotations

import pytest

from formatshield.backends.dryrun_backend import DryRunBackend
from formatshield.core import FormatShield
from formatshield.governance.policy import DefaultPolicyEngine


def test_default_policy_profile_strict_requires_schema() -> None:
    policy = DefaultPolicyEngine.from_profile("strict")
    shield = FormatShield(
        model="dryrun/test",
        backend=DryRunBackend(),
        policy_engine=policy,
    )

    with pytest.raises(PermissionError, match="requires a schema"):
        shield.generate_sync("Hello without schema")


def test_default_policy_profile_permissive_does_not_force_direct() -> None:
    policy = DefaultPolicyEngine.from_profile("permissive")
    assert policy.force_direct_on_short_prompt_chars is None
    assert policy.force_ttf_on_schema_depth == 4


def test_default_policy_blocks_denied_keyword() -> None:
    policy = DefaultPolicyEngine(deny_keywords=("forbidden",))
    shield = FormatShield(
        model="dryrun/test",
        backend=DryRunBackend(),
        policy_engine=policy,
    )

    with pytest.raises(PermissionError, match="denied keyword"):
        shield.generate_sync("This prompt contains forbidden text")


def test_default_policy_forces_ttf_on_deep_schema() -> None:
    policy = DefaultPolicyEngine(
        force_ttf_on_schema_depth=3,
        force_direct_on_short_prompt_chars=None,
    )
    shield = FormatShield(
        model="dryrun/test",
        backend=DryRunBackend(),
        policy_engine=policy,
    )

    schema = {
        "type": "object",
        "properties": {
            "a": {
                "type": "object",
                "properties": {
                    "b": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string"},
                        },
                        "required": ["value"],
                    }
                },
            }
        },
        "required": ["a"],
    }

    result = shield.generate_sync("Hi", schema=schema)
    assert result.routing.strategy == "ttf"
    assert "Policy forced route" in result.routing.explanation


def test_default_policy_forces_direct_on_short_shallow_prompt() -> None:
    policy = DefaultPolicyEngine(
        force_ttf_on_schema_depth=None,
        force_direct_on_short_prompt_chars=12,
    )
    shield = FormatShield(
        model="dryrun/test",
        backend=DryRunBackend(),
        policy_engine=policy,
    )

    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    result = shield.generate_sync("Hi", schema=schema)
    assert result.routing.strategy == "direct"
    assert "Policy forced route" in result.routing.explanation
