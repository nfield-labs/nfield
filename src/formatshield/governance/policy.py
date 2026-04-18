"""Default policy engine for pre-route and post-output enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from formatshield.hooks import HOOK_REQUEST_BEFORE_ROUTE, Hooks


@runtime_checkable
class PolicyEngineProtocol(Protocol):
    """Protocol for policy engines attachable to FormatShield hooks."""

    def attach_to_hooks(self, hooks: Hooks) -> None:
        """Register policy handlers on the provided Hooks instance."""


def _schema_depth(schema: Any, depth: int = 0) -> int:
    if not isinstance(schema, dict):
        return depth

    candidates = [depth]

    properties = schema.get("properties")
    if isinstance(properties, dict):
        for child in properties.values():
            candidates.append(_schema_depth(child, depth + 1))

    items = schema.get("items")
    if isinstance(items, dict):
        candidates.append(_schema_depth(items, depth + 1))

    for key in ("anyOf", "oneOf", "allOf"):
        group = schema.get(key)
        if isinstance(group, list):
            for child in group:
                candidates.append(_schema_depth(child, depth + 1))

    return max(candidates)


@dataclass
class DefaultPolicyEngine:
    """Built-in policy engine with controllable route forcing and blocking.

    This policy engine is intentionally deterministic and lightweight:
    it mutates pre-route hook context with decisions like:
    - block request
    - force strategy (direct/ttf)
    """

    max_prompt_chars: int | None = None
    deny_keywords: tuple[str, ...] = ()
    require_schema: bool = False
    force_direct_on_short_prompt_chars: int | None = 96
    force_ttf_on_schema_depth: int | None = 3

    @classmethod
    def from_profile(
        cls,
        profile: Literal["strict", "balanced", "permissive"] = "balanced",
        **overrides: Any,
    ) -> DefaultPolicyEngine:
        """Construct a policy engine from a named profile."""
        presets: dict[str, dict[str, Any]] = {
            "strict": {
                "max_prompt_chars": 4000,
                "deny_keywords": (
                    "ignore previous instructions",
                    "leak system prompt",
                ),
                "require_schema": True,
                "force_direct_on_short_prompt_chars": 64,
                "force_ttf_on_schema_depth": 2,
            },
            "balanced": {
                "max_prompt_chars": None,
                "deny_keywords": (),
                "require_schema": False,
                "force_direct_on_short_prompt_chars": 96,
                "force_ttf_on_schema_depth": 3,
            },
            "permissive": {
                "max_prompt_chars": None,
                "deny_keywords": (),
                "require_schema": False,
                "force_direct_on_short_prompt_chars": None,
                "force_ttf_on_schema_depth": 4,
            },
        }

        if profile not in presets:
            raise ValueError(f"Unknown policy profile: {profile}")

        config = dict(presets[profile])
        config.update(overrides)
        return cls(**config)

    def attach_to_hooks(self, hooks: Hooks) -> None:
        hooks.on(HOOK_REQUEST_BEFORE_ROUTE, self._on_before_route)

    def _on_before_route(self, context: dict[str, Any]) -> None:
        flags = context.setdefault("policy_flags", [])
        if not isinstance(flags, list):
            flags = []
            context["policy_flags"] = flags

        prompt = str(context.get("prompt") or "")
        schema = context.get("schema")

        if self.require_schema and not isinstance(schema, dict):
            context["blocked"] = True
            context["reason"] = "Policy requires a schema for generation"
            flags.append("blocked:require_schema")
            return

        if self.max_prompt_chars is not None and len(prompt) > self.max_prompt_chars:
            context["blocked"] = True
            context["reason"] = (
                f"Prompt length {len(prompt)} exceeds policy max_prompt_chars="
                f"{self.max_prompt_chars}"
            )
            flags.append("blocked:max_prompt_chars")
            return

        prompt_lower = prompt.lower()
        for keyword in self.deny_keywords:
            if keyword and keyword.lower() in prompt_lower:
                context["blocked"] = True
                context["reason"] = f"Prompt contains denied keyword: {keyword}"
                flags.append("blocked:deny_keyword")
                return

        depth = _schema_depth(schema) if isinstance(schema, dict) else 0
        context["schema_depth"] = depth

        if context.get("forced_strategy") in {"direct", "ttf"}:
            flags.append("forced_strategy:preserved")
            return

        if self.force_ttf_on_schema_depth is not None and depth >= self.force_ttf_on_schema_depth:
            context["forced_strategy"] = "ttf"
            context["reason"] = (
                f"Policy forced TTF for schema depth {depth} >= {self.force_ttf_on_schema_depth}"
            )
            flags.append("forced_strategy:ttf")
            return

        if (
            self.force_direct_on_short_prompt_chars is not None
            and len(prompt) <= self.force_direct_on_short_prompt_chars
            and depth <= 1
        ):
            context["forced_strategy"] = "direct"
            context["reason"] = (
                f"Policy forced direct for short prompt length {len(prompt)} "
                f"and shallow schema depth {depth}"
            )
            flags.append("forced_strategy:direct")
