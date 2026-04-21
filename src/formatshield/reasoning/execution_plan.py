"""
Reasoning Execution Plan Builder

Generates step-by-step BINDING execution protocols from JSON schema structure.
Unlike soft suggestions, execution plans are ORDERED and MANDATORY —
the model must complete each step before proceeding to the next.

Core insight: Schema structure reveals exactly what the model must do:
- Arrays with minItems  → must iterate and evaluate each item
- Enum arrays           → must restrict to exact allowed values only
- Boolean verdict flags → must maintain consistency with sub-results
- Numeric ranges        → must compute and validate bounds
- Required fields       → must populate all before finalizing

This converts "suggest fields to consider" into "execution protocol that
controls cognition."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from formatshield.oracle.routing_score import RoutingScore


@dataclass
class ExecutionStep:
    """A single binding step in the reasoning execution protocol."""

    step_number: int
    instruction: str
    binding: bool = True  # True = MUST execute, False = optional guidance


@dataclass
class ExecutionPlan:
    """
    Complete step-by-step execution protocol derived from schema structure.

    Unlike reasoning suggestions, execution plans are:
    - ORDERED: steps must be followed in sequence
    - BINDING: each step is mandatory (marked ★)
    - CONSISTENT: post-execution checks prevent contradictory outputs
    """

    steps: list[ExecutionStep] = field(default_factory=list)
    consistency_rules: list[str] = field(default_factory=list)
    enforcement_level: str = "guided"  # "strict" | "guided" | "flexible"

    def is_empty(self) -> bool:
        """Return True if plan has no actionable content."""
        return not self.steps and not self.consistency_rules


class ExecutionPlanBuilder:
    """Builds deterministic execution plans from schema + Φ routing signals."""

    def __init__(self, routing_score: RoutingScore) -> None:
        self.tau = routing_score.tau
        self.lambda2 = routing_score.lambda2

    def build(self, schema: dict[str, Any]) -> ExecutionPlan:
        """
        Build execution plan from schema structure.

        Analyzes schema for:
        1. Array fields with minItems → forced iteration steps
        2. Enum values (array items or top-level) → restricted selection steps
        3. Boolean verdict fields + sub-arrays → consistency enforcement
        4. Numeric range fields → bounds validation steps
        5. Required fields → final completion check

        Returns
        -------
        ExecutionPlan
            Ordered steps + consistency rules + enforcement level
        """
        steps: list[ExecutionStep] = []
        consistency_rules: list[str] = []
        n = 1

        properties: dict[str, Any] = schema.get("properties") or {}
        required: list[str] = schema.get("required") or []

        # ─── 1. Array fields: iterate and evaluate each item ─────────────
        for field_name, spec in properties.items():
            if not isinstance(spec, dict) or spec.get("type") != "array":
                continue

            min_items: int = spec.get("minItems", 0)
            item_spec: dict = spec.get("items") or {}
            if not isinstance(item_spec, dict):
                item_spec = {}
            item_props: dict = item_spec.get("properties") or {}

            # Minimum count enforcement
            if min_items > 0:
                steps.append(
                    ExecutionStep(
                        step_number=n,
                        instruction=(
                            f"Extract ALL '{field_name}' — MINIMUM {min_items} required. "
                            f"Do NOT proceed to the next step until you have identified "
                            f"at least {min_items} distinct {field_name}."
                        ),
                    )
                )
                n += 1

            # Per-item evaluation when items have sub-fields
            if item_props:
                item_req = item_spec.get("required") or []
                req_note = (
                    f" Required sub-fields for each: {', '.join(item_req)}." if item_req else ""
                )
                singular = field_name.rstrip("s") if field_name.endswith("s") else field_name
                steps.append(
                    ExecutionStep(
                        step_number=n,
                        instruction=(
                            f"Evaluate EACH {singular} in '{field_name}' INDEPENDENTLY. "
                            f"For every item assess: {', '.join(item_props.keys())}.{req_note} "
                            f"Complete ALL sub-fields for one item before moving to the next."
                        ),
                    )
                )
                n += 1

                # Boolean sub-fields inside items → no item may omit them
                bool_sub = [
                    k
                    for k, v in item_props.items()
                    if isinstance(v, dict) and v.get("type") == "boolean"
                ]
                for bf in bool_sub:
                    consistency_rules.append(
                        f"Every item in '{field_name}' MUST have '{bf}' set to true or false — "
                        f"no item may omit this field."
                    )

            # Enum-constrained array items
            if "enum" in item_spec:
                enum_vals: list = item_spec["enum"]
                steps.append(
                    ExecutionStep(
                        step_number=n,
                        instruction=(
                            f"Values in '{field_name}' MUST come ONLY from this exact list: "
                            f"{', '.join(repr(v) for v in enum_vals)}. "
                            f"Any value NOT in this list is INVALID — do not include it."
                        ),
                    )
                )
                n += 1

        # ─── 2. Top-level enum fields (non-array) ────────────────────────
        enum_fields = [
            (name, spec)
            for name, spec in properties.items()
            if isinstance(spec, dict) and "enum" in spec and spec.get("type") != "array"
        ]
        if enum_fields:
            enum_clauses = [
                f"'{fname}' → exactly one of {{{', '.join(repr(v) for v in fspec['enum'])}}}"
                for fname, fspec in enum_fields[:6]
            ]
            steps.append(
                ExecutionStep(
                    step_number=n,
                    instruction=(
                        "Assign enumerated fields — each MUST be exactly one allowed value: "
                        + "; ".join(enum_clauses)
                        + ". DO NOT invent values outside these lists."
                    ),
                )
            )
            n += 1

        # ─── 3. Numeric range fields ─────────────────────────────────────
        range_fields = [
            (name, spec)
            for name, spec in properties.items()
            if isinstance(spec, dict) and ("minimum" in spec or "maximum" in spec)
        ]
        if range_fields:
            range_clauses = [
                f"'{fname}' ∈ [{spec.get('minimum', '−∞')}, {spec.get('maximum', '+∞')}]"
                for fname, spec in range_fields
            ]
            steps.append(
                ExecutionStep(
                    step_number=n,
                    instruction=(
                        "Compute numeric fields within required bounds: "
                        + "; ".join(range_clauses)
                        + ". Any value outside these ranges is INVALID."
                    ),
                )
            )
            n += 1

        # ─── 4. Boolean verdict fields + array consistency ───────────────
        bool_top = [
            name
            for name, spec in properties.items()
            if isinstance(spec, dict) and spec.get("type") == "boolean"
        ]
        array_top = [
            name
            for name, spec in properties.items()
            if isinstance(spec, dict) and spec.get("type") == "array"
        ]

        for bfield in bool_top:
            bfield_lower = bfield.lower()
            is_verdict = (
                "valid" in bfield_lower
                or "correct" in bfield_lower
                or "match" in bfield_lower
                or "result" in bfield_lower
            )
            if not is_verdict:
                continue

            for afield in array_top:
                aspec = properties.get(afield) or {}
                aitem = aspec.get("items") or {}
                if not isinstance(aitem, dict):
                    continue
                aitem_props = aitem.get("properties") or {}
                # Cross-field consistency: boolean verdict ↔ array sub-item booleans
                if any("valid" in k.lower() for k in aitem_props):
                    consistency_rules.append(
                        f"CONSISTENCY: '{bfield}' MUST reflect the results in '{afield}'. "
                        f"If ANY item in '{afield}' has a negative/false evaluation, "
                        f"'{bfield}' CANNOT be true — it must be false."
                    )

        # ─── 5. Required field completion check ──────────────────────────
        if required:
            steps.append(
                ExecutionStep(
                    step_number=n,
                    instruction=(
                        f"Verify ALL required fields are populated before finalizing: "
                        f"{', '.join(required)}. "
                        f"Any missing required field makes the entire response INVALID."
                    ),
                )
            )
            n += 1

        # Enforcement level derived from τ (constraint tightness)
        if self.tau >= 0.7:
            enforcement = "strict"
        elif self.tau >= 0.4:
            enforcement = "guided"
        else:
            enforcement = "flexible"

        return ExecutionPlan(
            steps=steps,
            consistency_rules=consistency_rules,
            enforcement_level=enforcement,
        )

    @staticmethod
    def render(plan: ExecutionPlan) -> str:
        """
        Render execution plan as a prompt-injection ready string.

        Parameters
        ----------
        plan
            ExecutionPlan to render

        Returns
        -------
        str
            Formatted execution protocol ready for injection into Pass 1 prompt.
            Empty string if plan has no content.
        """
        if plan.is_empty():
            return ""

        lines: list[str] = [
            "## EXECUTION PROTOCOL",
            (
                f"[Enforcement: {plan.enforcement_level.upper()}] "
                "Follow these steps IN ORDER. Do NOT skip any step. "
                "Each ★ step is MANDATORY."
            ),
            "",
        ]

        for step in plan.steps:
            marker = "★" if step.binding else "○"
            lines.append(f"Step {step.step_number} {marker}: {step.instruction}")

        if plan.consistency_rules:
            lines.append("")
            lines.append("### POST-EXECUTION CONSISTENCY CHECKS")
            lines.append("Before finalizing your response, verify ALL of the following:")
            for rule in plan.consistency_rules:
                lines.append(f"  • {rule}")

        return "\n".join(lines)


def build_execution_plan(
    schema: dict[str, Any],
    routing_score: RoutingScore,
) -> ExecutionPlan:
    """
    Public API: build execution plan from schema.

    Parameters
    ----------
    schema
        JSON Schema dict
    routing_score
        RoutingScore for enforcement-level derivation

    Returns
    -------
    ExecutionPlan
        Ordered binding execution protocol
    """
    return ExecutionPlanBuilder(routing_score).build(schema)


def render_execution_plan(plan: ExecutionPlan) -> str:
    """
    Public API: render execution plan to injection-ready string.

    Parameters
    ----------
    plan
        ExecutionPlan to render

    Returns
    -------
    str
        Formatted protocol string, empty if no content
    """
    return ExecutionPlanBuilder.render(plan)
