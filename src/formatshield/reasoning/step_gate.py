"""
Step gating engine for execution plan enforcement.

Forward-steering enforcement layer that verifies execution step completeness
before permitting the next step to begin.

Mechanism: forward-steering (NOT backtracking — backtracking is expensive and
destroys KV cache). When a step is detected as incomplete, a focused injection
token-sequence steers the model back to complete the step.

Called after Pass 1 reasoning trace is generated but before Pass 2 formatting.
Examines partial structured output (if the model produced any) or trace text
to determine whether each step in the ExecutionPlan was satisfied.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from formatshield.reasoning.execution_plan import ExecutionPlan, ExecutionStep

# ---------------------------------------------------------------------------
# Public data contracts
# ---------------------------------------------------------------------------


@dataclass
class StepCheckResult:
    """
    Result of checking one execution step against partial output.

    Attributes
    ----------
    step:
        The ExecutionStep that was checked.
    complete:
        True when the step's requirements are satisfied.
    missing_description:
        Human-readable description of what is missing (empty when complete).
    injection_text:
        Forward-steering injection text to append to the generation context
        when ``complete=False``. Empty when the step is already complete.
    """

    step: ExecutionStep
    complete: bool
    missing_description: str = ""
    injection_text: str = ""


@dataclass
class GateResult:
    """
    Aggregated result of checking all steps in an execution plan.

    Attributes
    ----------
    all_complete:
        True when every step in the plan is satisfied.
    step_results:
        Per-step check results in plan order.
    combined_injection:
        All forward-steering injections joined for prompt prepend.
        Empty when all_complete=True.
    incomplete_count:
        Number of steps that failed their completeness check.
    """

    all_complete: bool
    step_results: list[StepCheckResult] = field(default_factory=list)
    combined_injection: str = ""
    incomplete_count: int = 0

    def has_injections(self) -> bool:
        """Return True if any forward-steering injections are pending."""
        return bool(self.combined_injection)


# ---------------------------------------------------------------------------
# Step-type detectors
# ---------------------------------------------------------------------------

#: Pattern to detect "Extract ALL '<field>' — MINIMUM N required" steps
_ARRAY_STEP_PATTERN = re.compile(
    r"extract\s+all\s+'?(\w+)'?\s+.*minimum\s+(\d+)\s+required",
    re.IGNORECASE,
)

#: Pattern to detect "Evaluate EACH <item> in '<field>'" steps
_EVAL_STEP_PATTERN = re.compile(
    r"evaluate\s+each\s+\w+\s+in\s+'?(\w+)'?",
    re.IGNORECASE,
)

#: Pattern to detect required field completion steps
_REQUIRED_STEP_PATTERN = re.compile(
    r"verify\s+all\s+required\s+fields",
    re.IGNORECASE,
)

#: Pattern to detect enum assignment steps
_ENUM_STEP_PATTERN = re.compile(
    r"assign\s+enumerated\s+fields",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Core gate engine
# ---------------------------------------------------------------------------


class TemporalStepGate:
    """
    Check execution plan step completion and emit forward-steering injections.

    Usage
    -----
    ::

        gate = TemporalStepGate()
        result = gate.gate_plan(plan, partial_output, trace_text)
        if not result.all_complete:
            # Prepend result.combined_injection to the next generation context
            corrected_context = result.combined_injection + existing_context
    """

    def gate_plan(
        self,
        plan: ExecutionPlan,
        partial_output: dict[str, Any] | None = None,
        trace_text: str = "",
    ) -> GateResult:
        """
        Check every step in the plan and collect forward-steering injections.

        Parameters
        ----------
        plan:
            The ExecutionPlan generated from the schema.
        partial_output:
            Partial or full structured output dict from Pass 1 / Pass 2.
            May be None when only trace text is available.
        trace_text:
            Raw Pass 1 reasoning trace text (for heuristic content checks).

        Returns
        -------
        GateResult
            Aggregated check result with per-step details and any injections.
        """
        if plan.is_empty():
            return GateResult(all_complete=True)

        results: list[StepCheckResult] = []
        injections: list[str] = []

        for step in plan.steps:
            if not step.binding:
                # Non-binding steps are guidance only — always marked complete
                results.append(StepCheckResult(step=step, complete=True))
                continue

            check = self.check_step(step, partial_output or {}, trace_text)
            results.append(check)
            if not check.complete and check.injection_text:
                injections.append(check.injection_text)

        incomplete = sum(1 for r in results if not r.complete)
        return GateResult(
            all_complete=incomplete == 0,
            step_results=results,
            combined_injection="\n\n".join(injections),
            incomplete_count=incomplete,
        )

    def check_step(
        self,
        step: ExecutionStep,
        partial_output: dict[str, Any],
        trace_text: str = "",
    ) -> StepCheckResult:
        """
        Check a single execution step for completeness.

        Dispatches to the appropriate check based on the instruction content.

        Parameters
        ----------
        step:
            Execution step to verify.
        partial_output:
            Current structured output dict (may be empty).
        trace_text:
            Pass 1 reasoning trace (used as fallback when output is absent).

        Returns
        -------
        StepCheckResult
            Complete=True when the step is satisfied; otherwise includes
            injection_text for forward-steering.
        """
        instruction = step.instruction

        # Detect step type by instruction content
        array_match = _ARRAY_STEP_PATTERN.search(instruction)
        if array_match:
            return self._check_array_step(step, array_match, partial_output, trace_text)

        eval_match = _EVAL_STEP_PATTERN.search(instruction)
        if eval_match:
            return self._check_eval_step(step, eval_match, partial_output, trace_text)

        if _REQUIRED_STEP_PATTERN.search(instruction):
            return self._check_required_step(step, instruction, partial_output)

        if _ENUM_STEP_PATTERN.search(instruction):
            return self._check_enum_step(step, instruction, partial_output)

        # Generic binding step — check for mention in trace as heuristic
        return self._check_generic_step(step, trace_text)

    # ------------------------------------------------------------------
    # Step-type specific checks
    # ------------------------------------------------------------------

    def _check_array_step(
        self,
        step: ExecutionStep,
        match: re.Match[str],
        partial_output: dict[str, Any],
        trace_text: str,
    ) -> StepCheckResult:
        """Check that an array field meets its minItems requirement."""
        field_name = match.group(1)
        min_items = int(match.group(2))

        actual_items = partial_output.get(field_name)
        if isinstance(actual_items, list):
            actual_count = len(actual_items)
            if actual_count >= min_items:
                return StepCheckResult(step=step, complete=True)
            missing = min_items - actual_count
            desc = (
                f"'{field_name}' has {actual_count} item(s) but needs at least {min_items}. "
                f"{missing} more required."
            )
            injection = self.inject_continuation(step, desc, field_name, missing)
            return StepCheckResult(
                step=step,
                complete=False,
                missing_description=desc,
                injection_text=injection,
            )

        # Field not yet in output — check trace text for mentions
        if field_name.lower() in trace_text.lower():
            # Field was mentioned but not yet structured — treat as in-progress
            # Use heuristic: count list-like patterns near field name in trace
            count = _count_trace_items(trace_text, field_name)
            if count >= min_items:
                return StepCheckResult(step=step, complete=True)

        desc = (
            f"'{field_name}' not found or insufficient in output. "
            f"Need at least {min_items} distinct {field_name} items."
        )
        injection = self.inject_continuation(step, desc, field_name, min_items)
        return StepCheckResult(
            step=step,
            complete=False,
            missing_description=desc,
            injection_text=injection,
        )

    def _check_eval_step(
        self,
        step: ExecutionStep,
        match: re.Match[str],
        partial_output: dict[str, Any],
        trace_text: str,
    ) -> StepCheckResult:
        """Check that each item in an array was individually evaluated."""
        field_name = match.group(1)
        items = partial_output.get(field_name)

        if not isinstance(items, list) or len(items) == 0:
            # Array not yet populated — check trace for evidence
            if field_name.lower() in trace_text.lower():
                return StepCheckResult(step=step, complete=True)
            desc = f"Individual evaluation of each '{field_name}' item not detected."
            injection = self.inject_continuation(step, desc, field_name, 0)
            return StepCheckResult(
                step=step,
                complete=False,
                missing_description=desc,
                injection_text=injection,
            )

        # Check that each item has at least one non-empty field
        incomplete_items = [
            i
            for i, item in enumerate(items)
            if not isinstance(item, dict) or not any(item.values())
        ]
        if incomplete_items:
            desc = (
                f"Item(s) at index {incomplete_items} in '{field_name}' "
                f"appear to be incomplete (empty or non-dict)."
            )
            injection = self.inject_continuation(step, desc, field_name, len(incomplete_items))
            return StepCheckResult(
                step=step,
                complete=False,
                missing_description=desc,
                injection_text=injection,
            )

        return StepCheckResult(step=step, complete=True)

    def _check_required_step(
        self,
        step: ExecutionStep,
        instruction: str,
        partial_output: dict[str, Any],
    ) -> StepCheckResult:
        """Check that all required fields mentioned in the step are present."""
        # Extract field names from the instruction text
        # Pattern: "Verify ALL required fields are populated before finalizing: a, b, c."
        colon_pos = instruction.find(":")
        if colon_pos == -1:
            return StepCheckResult(step=step, complete=True)

        fields_text = instruction[colon_pos + 1 :].strip().rstrip(".")
        required_fields = [f.strip() for f in fields_text.split(",") if f.strip()]

        missing = [f for f in required_fields if f not in partial_output]
        if not missing:
            return StepCheckResult(step=step, complete=True)

        desc = f"Required field(s) missing from output: {', '.join(missing)}."
        injection = (
            f"[Step {step.step_number} incomplete] Required field(s) not yet populated: "
            f"{', '.join(missing)}. Add these fields before finalizing."
        )
        return StepCheckResult(
            step=step,
            complete=False,
            missing_description=desc,
            injection_text=injection,
        )

    def _check_enum_step(
        self,
        step: ExecutionStep,
        instruction: str,
        partial_output: dict[str, Any],
    ) -> StepCheckResult:
        """Check that enum fields have been assigned valid values."""
        # Parse "field_name → exactly one of {v1, v2, v3}" patterns
        enum_pattern = re.compile(r"'(\w+)'\s+→\s+exactly one of\s+\{([^}]+)\}")
        violations: list[str] = []

        for enum_match in enum_pattern.finditer(instruction):
            fname = enum_match.group(1)
            allowed_str = enum_match.group(2)
            allowed = {v.strip().strip("'\"") for v in allowed_str.split(",")}

            if fname not in partial_output:
                continue  # Field not yet generated — skip

            actual = str(partial_output[fname]).strip("'\"")
            if actual not in allowed:
                violations.append(
                    f"'{fname}'={partial_output[fname]!r} not in allowed set {{{allowed_str}}}"
                )

        if not violations:
            return StepCheckResult(step=step, complete=True)

        desc = "Enum violations detected: " + "; ".join(violations)
        injection = (
            f"[Step {step.step_number} incomplete] Enum constraint violations: "
            + "; ".join(violations)
            + ". Correct the listed field(s) to use only allowed values."
        )
        return StepCheckResult(
            step=step,
            complete=False,
            missing_description=desc,
            injection_text=injection,
        )

    def _check_generic_step(self, step: ExecutionStep, trace_text: str) -> StepCheckResult:
        """
        Heuristic check for generic binding steps.

        When no structured content is available, checks whether the step
        instruction keywords appear in the trace text as a proxy for
        completion.
        """
        if not trace_text:
            # No trace available — cannot verify; assume complete to avoid
            # false-positive injections that confuse the model
            return StepCheckResult(step=step, complete=True)

        # Extract key nouns from the instruction and check for trace coverage
        keywords = _extract_keywords(step.instruction)
        trace_lower = trace_text.lower()
        found = sum(1 for kw in keywords if kw in trace_lower)

        # Require at least half the key terms to be present
        threshold = max(1, len(keywords) // 2)
        if found >= threshold:
            return StepCheckResult(step=step, complete=True)

        desc = (
            f"Step {step.step_number} keywords not sufficiently covered in trace. "
            f"Found {found}/{len(keywords)} key terms."
        )
        injection = (
            f"[Step {step.step_number} may be incomplete] "
            f"Please ensure you have addressed: {step.instruction}"
        )
        return StepCheckResult(
            step=step,
            complete=False,
            missing_description=desc,
            injection_text=injection,
        )

    # ------------------------------------------------------------------
    # Injection builder
    # ------------------------------------------------------------------

    @staticmethod
    def inject_continuation(
        step: ExecutionStep,
        missing_description: str,
        field_name: str,
        missing_count: int,
    ) -> str:
        """
        Build a forward-steering injection text for an incomplete step.

        The injection is designed to be prepended to the next generation
        window context to guide the model back to completing the step.

        Parameters
        ----------
        step:
            The incomplete execution step.
        missing_description:
            Human-readable description of what is missing.
        field_name:
            Primary field that is incomplete.
        missing_count:
            Number of additional items/fields needed (0 = unknown).

        Returns
        -------
        str
            Forward-steering injection string.
        """
        if missing_count > 0:
            count_hint = f" Need {missing_count} more."
        else:
            count_hint = ""

        return (
            f"[Step {step.step_number} ★ INCOMPLETE — {missing_description}{count_hint} "
            f"Continue completing '{field_name}' before proceeding to the next step.]"
        )


# ---------------------------------------------------------------------------
# Trace analysis helpers
# ---------------------------------------------------------------------------


def _count_trace_items(trace: str, field_name: str) -> int:
    """
    Heuristically count how many items for a field appear in the trace.

    Looks for numbered/bulleted list items near the field name. Used as
    a fallback when structured output is not yet available.
    """
    # Find the section of the trace near the field name
    idx = trace.lower().find(field_name.lower())
    if idx == -1:
        return 0

    # Extract ~500 chars after first mention
    section = trace[idx : idx + 500]

    # Count numbered list items (1. 2. 3.) or bullet items (- or •)
    numbered = len(re.findall(r"^\s*\d+\.", section, re.MULTILINE))
    bulleted = len(re.findall(r"^\s*[-•*]", section, re.MULTILINE))
    return max(numbered, bulleted)


def _extract_keywords(instruction: str) -> list[str]:
    """Extract meaningful keyword tokens from a step instruction."""
    # Remove punctuation and split
    cleaned = re.sub(r"[^\w\s]", " ", instruction.lower())
    words = cleaned.split()
    # Filter out common stop words and very short tokens
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "be",
        "been",
        "all",
        "each",
        "any",
        "every",
        "do",
        "not",
        "must",
        "will",
        "this",
        "that",
        "it",
        "its",
        "has",
        "have",
        "you",
        "your",
        "step",
        "before",
        "after",
        "proceed",
        "next",
    }
    return [w for w in words if len(w) > 2 and w not in stop]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_execution_steps(
    plan: ExecutionPlan,
    partial_output: dict[str, Any] | None = None,
    trace_text: str = "",
) -> GateResult:
    """
    Check execution plan step completeness and emit forward-steering injections.

    This is the primary entry point for the Temporal Step Gating Engine.
    Call after Pass 1 generation to detect incomplete steps and obtain
    injection text for forward-steering before Pass 2.

    Parameters
    ----------
    plan:
        Execution plan built from the schema by ExecutionPlanBuilder.
    partial_output:
        Partial or full structured output dict. Pass None when only trace
        text is available.
    trace_text:
        Raw Pass 1 reasoning trace for heuristic coverage checks.

    Returns
    -------
    GateResult
        all_complete=True when all binding steps are satisfied.
        combined_injection contains forward-steering text when any step fails.
    """
    return TemporalStepGate().gate_plan(plan, partial_output, trace_text)


def parse_partial_output(raw_text: str) -> dict[str, Any]:
    """
    Attempt to extract a partial JSON dict from raw generation text.

    Tries JSON parsing first; falls back to extracting the last complete
    JSON object found in the text. Returns an empty dict on failure.

    Parameters
    ----------
    raw_text:
        Raw generation output that may contain partial or complete JSON.

    Returns
    -------
    dict[str, Any]
        Extracted dict, or empty dict if no valid JSON is found.
    """
    text = raw_text.strip()

    # Try direct parse first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find the last complete JSON object with regex
    json_pattern = re.compile(r"\{[^{}]*\}", re.DOTALL)
    matches = json_pattern.findall(text)
    for candidate in reversed(matches):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue

    return {}
