"""
Pass 1 Quality Gate.

Scores a TTF Pass 1 reasoning trace against three deterministic heuristics
before allowing it to feed into Pass 2.  Bad traces are caught here rather
than silently corrupting the final structured output.

Heuristic Set
-------------
1. **Required-field coverage** — every field name marked ``required`` in the
   JSON schema must appear (as a token) in the thinking text.  A model that
   never mentions ``order_id`` in its reasoning is unlikely to populate it
   correctly in Pass 2.

2. **Contradiction detector** — looks for unresolved self-contradictions in
   the reasoning: phrases like "wait, no", "actually that's wrong", or
   "I made an error" without a subsequent correction signal.  A single
   mid-trace correction is normal; multiple unresolved contradictions are not.

3. **Vocabulary-bridge coverage** (ΔK-gated) — when a routing score is
   provided and ``ΔK > 0.50``, the schema fields most likely to be
   vocabulary-mismatched (those not present in the prompt) should appear
   in the thinking text.  If the model never mentions ``parties[0]`` when
   that field is absent from the prompt, it will likely output the wrong
   field name in Pass 2.

Gate decision
-------------
``passed = score ≥ QUALITY_GATE_PASS_THRESHOLD``

The score is the fraction of checks that passed (0.0–1.0).  The default
threshold is 0.67, meaning at least 2 of 3 checks must pass.

On failure the engine retries Pass 1 once.  After a retry failure the trace
continues to Pass 2 with a warning logged — the gate is a quality signal, not
a hard blocker.

Public API
----------
- :class:`QualityGateResult` — frozen dataclass with the gate verdict
- :func:`score_thinking_trace` — the main scoring function
- :data:`QUALITY_GATE_PASS_THRESHOLD` — threshold constant (overridable for
  testing / tuning)
"""

from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from formatshield.oracle.routing_score import RoutingScore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Minimum fraction of heuristic checks that must pass.
QUALITY_GATE_PASS_THRESHOLD: float = 0.67

#: ΔK threshold above which vocabulary-bridge coverage check is activated.
_DK_VOCAB_CHECK_THRESHOLD: float = 0.50

#: Regex patterns that indicate an *unresolved* self-contradiction in the trace.
#: Each pattern is applied case-insensitively.
_CONTRADICTION_PATTERNS: list[str] = [
    r"wait[,\s]+no\b",
    r"actually\s+(?:that.s\s+)?(?:wrong|incorrect|not\s+right)",
    r"i\s+made\s+an?\s+(?:error|mistake)",
    r"that\s+(?:is|was)\s+(?:wrong|incorrect|not\s+right)",
    r"i\s+(?:was|am)\s+(?:wrong|incorrect)\s+(?:about|here|there)",
]
_CONTRADICTION_RE = re.compile(
    "|".join(f"(?:{p})" for p in _CONTRADICTION_PATTERNS),
    re.IGNORECASE,
)

#: Regex patterns that indicate the model *resolved* the contradiction inline.
_RESOLUTION_PATTERNS: list[str] = [
    r"(?:so|therefore|thus|actually|correction)[,:\s]+(?:the\s+)?(?:correct|right|answer\s+is)",
    r"let\s+me\s+(?:recalculate|reconsider|redo|try\s+again)",
    r"(?:the|my)\s+(?:correct|final|revised)\s+(?:answer|value|result)\s+is",
]
_RESOLUTION_RE = re.compile(
    "|".join(f"(?:{p})" for p in _RESOLUTION_PATTERNS),
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class QualityGateResult:
    """Verdict from :func:`score_thinking_trace`.

    Attributes
    ----------
    passed:
        ``True`` when ``score ≥ QUALITY_GATE_PASS_THRESHOLD``.
    score:
        Fraction of checks that passed, ∈ [0.0, 1.0].
    failed_checks:
        Human-readable names of the checks that failed.
    details:
        Per-check diagnostic information for logging and calibration.
    """

    passed: bool
    score: float
    failed_checks: list[str]
    details: dict[str, Any]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_required_field_coverage(
    thinking: str,
    schema: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Check 1: all required schema field names appear in the thinking text.

    Returns ``(passed, details)`` where ``details`` contains the set of
    missing required fields (empty when passed).
    """
    required: list[str] = schema.get("required", [])
    if not required:
        # No required fields — check trivially passes
        return True, {"required_fields": [], "missing_fields": []}

    thinking_lower = thinking.lower()
    missing: list[str] = []

    for field_name in required:
        # Decompose snake_case / camelCase field names into individual tokens
        tokens = re.sub(r"([a-z])([A-Z])", r"\1 \2", field_name).lower()
        tokens_list = tokens.replace("_", " ").replace("-", " ").split()

        # Check if any meaningful token (>2 chars) from the field name appears
        meaningful = [t for t in tokens_list if len(t) > 2]
        if not meaningful:
            meaningful = tokens_list  # short fields like "id", "to"

        found = any(t in thinking_lower for t in meaningful)
        if not found:
            missing.append(field_name)

    passed = len(missing) == 0
    return passed, {
        "required_fields": required,
        "missing_fields": missing,
        "coverage_ratio": (len(required) - len(missing)) / max(len(required), 1),
    }


def _check_contradiction_free(
    thinking: str,
) -> tuple[bool, dict[str, Any]]:
    """Check 2: no unresolved self-contradictions in the reasoning trace.

    A trace is considered contradiction-free when:
    - It contains no contradiction phrases, OR
    - Every contradiction phrase is followed (within 200 chars) by a
      resolution phrase indicating the model corrected itself.
    """
    contradiction_matches = list(_CONTRADICTION_RE.finditer(thinking))
    if not contradiction_matches:
        return True, {"contradictions_found": 0, "unresolved": 0}

    unresolved = 0
    for m in contradiction_matches:
        # Look for a resolution within the next 200 characters
        window = thinking[m.start() : m.start() + 200]
        if not _RESOLUTION_RE.search(window):
            unresolved += 1

    passed = unresolved == 0
    return passed, {
        "contradictions_found": len(contradiction_matches),
        "unresolved": unresolved,
    }


def _check_vocab_bridge_coverage(
    thinking: str,
    schema: dict[str, Any],
    routing_score: RoutingScore | None,
) -> tuple[bool, dict[str, Any]]:
    """Check 3 (ΔK-gated): schema fields absent from prompt appear in thinking.

    Only active when ``routing_score.delta_k > _DK_VOCAB_CHECK_THRESHOLD``.
    When inactive, the check trivially passes and does not penalize the score.

    For each schema leaf field whose name is not typically found in natural
    language prompts, verify it appears at least once in the thinking text.
    """
    if routing_score is None or routing_score.delta_k <= _DK_VOCAB_CHECK_THRESHOLD:
        return True, {"dk_check_active": False}

    props = schema.get("properties", {})
    if not props:
        return True, {"dk_check_active": True, "schema_fields": 0}

    thinking_lower = thinking.lower()
    uncovered: list[str] = []

    for field_name in props:
        # Same tokenization as vocabulary bridge
        tokens = re.sub(r"([a-z])([A-Z])", r"\1 \2", field_name).lower()
        tokens_list = tokens.replace("_", " ").replace("-", " ").split()
        meaningful = [t for t in tokens_list if len(t) > 2]
        if not meaningful:
            meaningful = tokens_list

        found = any(t in thinking_lower for t in meaningful)
        if not found:
            uncovered.append(field_name)

    # Pass if ≥50% of schema fields are mentioned
    field_count = len(props)
    covered_count = field_count - len(uncovered)
    coverage_ratio = covered_count / max(field_count, 1)
    passed = coverage_ratio >= 0.50

    return passed, {
        "dk_check_active": True,
        "delta_k": routing_score.delta_k,
        "field_count": field_count,
        "uncovered_fields": uncovered,
        "coverage_ratio": coverage_ratio,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_thinking_trace(
    thinking: str,
    schema: dict[str, Any] | None,
    routing_score: RoutingScore | None = None,
) -> QualityGateResult:
    """Score a Pass 1 reasoning trace against the three-heuristic quality gate.

    Parameters
    ----------
    thinking:
        The extracted Pass 1 reasoning text (content of ``<think>…</think>``
        tags, as returned by :func:`~formatshield.ttf.prompts.extract_thinking`).
    schema:
        JSON Schema dict describing the expected output.  When ``None``, only
        the contradiction check is performed.
    routing_score:
        Optional routing score from OracleX.  When provided and
        ``delta_k > 0.50``, the vocabulary-bridge coverage check (Check 3)
        is activated.

    Returns
    -------
    QualityGateResult
        Verdict with ``passed``, ``score``, ``failed_checks``, and ``details``.
    """
    if not thinking or not thinking.strip():
        return QualityGateResult(
            passed=False,
            score=0.0,
            failed_checks=["empty_thinking_trace"],
            details={"reason": "Pass 1 produced an empty reasoning trace"},
        )

    effective_schema: dict[str, Any] = schema if isinstance(schema, dict) else {}

    # Run the three checks
    check1_passed, check1_details = _check_required_field_coverage(thinking, effective_schema)
    check2_passed, check2_details = _check_contradiction_free(thinking)
    check3_passed, check3_details = _check_vocab_bridge_coverage(
        thinking, effective_schema, routing_score
    )

    checks = [
        ("required_field_coverage", check1_passed),
        ("contradiction_free", check2_passed),
        ("vocab_bridge_coverage", check3_passed),
    ]

    passed_count = sum(1 for _, p in checks if p)
    score = passed_count / len(checks)

    failed = [name for name, p in checks if not p]
    gate_passed = score >= QUALITY_GATE_PASS_THRESHOLD

    return QualityGateResult(
        passed=gate_passed,
        score=score,
        failed_checks=failed,
        details={
            "required_field_coverage": check1_details,
            "contradiction_free": check2_details,
            "vocab_bridge_coverage": check3_details,
            "checks_passed": passed_count,
            "total_checks": len(checks),
        },
    )
