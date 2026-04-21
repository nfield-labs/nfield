"""
Semantic Failure Mode Detector (Replacement for length-based detection).

Current problem: FailureModeDetector uses surface-level signals:
  - short_prompt (length < 50 tokens) → assumes extraction
  - schema_depth ≤ 1 → assumes no reasoning needed

This is WRONG because:
  - "Analyze drug interactions" is SHORT but requires DEEP reasoning
  - Email schema is FLAT but current task is REASONING

Solution: Detect failure modes by semantic intent, not surface metrics.

Failure modes (semantic):
  1. simple_extraction — flat schema AND short prompt AND no reasoning verbs
  2. schema_mismatch — schema–prompt alignment score < 0.4 (NEW)
  3. mislabeled_extraction — marked extraction but contains reasoning verbs
  4. reasoning_task_with_flat_schema — reasoning intent but schema can't support it (NEW)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Reasoning intent keywords — presence of ANY indicates semantic complexity
REASONING_VERBS = {
    "analyze", "analyze", "evaluate", "assess", "reason",
    "compare", "contrast", "distinguish", "differentiate",
    "explain", "interpret", "clarify", "elaborate",
    "determine", "decide", "conclude", "infer",
    "calculate", "compute", "derive", "estimate",
    "classify", "categorize", "organize", "structure",
    "identify", "recognize", "detect", "spot",
    "solve", "resolve", "address", "tackle",
    "validate", "verify", "check", "confirm",
    "recommend", "suggest", "advise", "propose",
}

# Exploration keywords — indicate open-ended reasoning
EXPLORATION_KEYWORDS = {
    "discuss", "explore", "elaborate", "expand", "detail",
    "describe", "explain", "interpret", "consider",
    "what if", "how", "why", "how else",
}


@dataclass
class SemanticFailureMode:
    """Detected semantic failure mode.
    
    Attributes
    ----------
    mode_type: str
        Type of failure mode (e.g., "schema_mismatch", "reasoning_with_flat_schema")
    
    severity: str
        "hard" — hard override to direct mode
        "soft" — advisory, included in logs but doesn't block
        "critical" — schema misalignment, fallback to unstructured
    
    confidence: float
        How confident we are this failure mode applies (0–1)
    
    explanation: str
        Human-readable explanation
    """
    mode_type: str
    severity: str
    confidence: float
    explanation: str


def detect_reasoning_intent(prompt: str) -> bool:
    """Detect whether prompt contains reasoning/analysis intent.
    
    Returns True if prompt contains:
    - Reasoning verbs (analyze, evaluate, compare, etc.)
    - Reasoning indicators (multi-step, explain why, etc.)
    - Complex task language
    
    Returns False if prompt is pure extraction:
    - "Extract X from Y"
    - "Get the email address"
    - "List all occurrences"
    
    Parameters
    ----------
    prompt: str
        The prompt text
    
    Returns
    -------
    bool
        True if reasoning intent detected
    """
    prompt_lower = prompt.lower()
    
    # Check for reasoning verbs
    for verb in REASONING_VERBS:
        # Word boundary: avoid matching "analyze" in "analyzed" (ok) but not in "paralyze" (bad)
        pattern = rf'\b{re.escape(verb)}\b'
        if re.search(pattern, prompt_lower):
            logger.debug(f"Reasoning intent detected: found verb '{verb}'")
            return True
    
    # Check for multi-step reasoning indicators
    multi_step_indicators = [
        r'step\s*by\s*step',
        r'break\s*down',
        r'explain\s+why',
        r'explain\s+how',
        r'what\s+is\s+the\s+impact',
        r'why\s+would',
        r'how\s+would',
    ]
    
    for pattern in multi_step_indicators:
        if re.search(pattern, prompt_lower):
            logger.debug(f"Reasoning intent detected: multi-step indicator '{pattern}'")
            return True
    
    # Check for exploration language
    for keyword in EXPLORATION_KEYWORDS:
        pattern = rf'\b{re.escape(keyword)}\b'
        if re.search(pattern, prompt_lower):
            logger.debug(f"Reasoning intent detected: exploration keyword '{keyword}'")
            return True
    
    return False


def detect_extraction_only(prompt: str) -> bool:
    """Detect whether prompt is ONLY extraction (no reasoning).
    
    Returns True if prompt explicitly instructs extraction without analysis:
    - "Extract X from Y"
    - "Get the value of field Z"
    - "List all instances"
    - "Copy these fields"
    
    Parameters
    ----------
    prompt: str
        The prompt text
    
    Returns
    -------
    bool
        True if pure extraction (no reasoning)
    """
    prompt_lower = prompt.lower()
    
    extraction_patterns = [
        r'extract\s+',
        r'get\s+the\s+(?!reason|why|how)',
        r'list\s+all\s+',
        r'copy\s+',
        r'just\s+return\s+',
        r'provide\s+(?!analysis|reasoning)',
    ]
    
    for pattern in extraction_patterns:
        if re.search(pattern, prompt_lower):
            logger.debug(f"Extraction-only detected: matched '{pattern}'")
            return True
    
    return False


def detect_schema_reasoning_mismatch(
    schema: dict,
    prompt: str,
    alignment_score: float,
) -> SemanticFailureMode | None:
    """Detect if schema and prompt have reasoning mismatch.
    
    Failure case: prompt requires reasoning but schema is flat/simple
    
    Example:
        schema = email validation (flat, 1 level)
        prompt = "analyze drug interactions" (requires reasoning)
        → FAIL: schema_reasoning_mismatch
    
    Parameters
    ----------
    schema: dict
        JSON schema
    prompt: str
        Task prompt
    alignment_score: float
        Schema–prompt alignment (0–1)
    
    Returns
    -------
    SemanticFailureMode or None
        Failure mode if detected; None otherwise
    """
    if alignment_score >= 0.4:
        # Reasonable alignment, no mismatch
        return None
    
    # Misaligned schema + reasoning prompt = mismatch
    has_reasoning = detect_reasoning_intent(prompt)
    
    if has_reasoning:
        return SemanticFailureMode(
            mode_type="schema_reasoning_mismatch",
            severity="critical",
            confidence=0.9,
            explanation=(
                f"Prompt requires reasoning (alignment={alignment_score:.2f} < 0.4 threshold), "
                f"but schema appears misaligned. "
                f"Schema designed for different task. "
                f"FALLBACK to unstructured."
            ),
        )
    
    return None


def detect_reasoning_with_flat_schema(
    schema: dict,
    prompt: str,
) -> SemanticFailureMode | None:
    """Detect if prompt needs reasoning but schema is too flat.
    
    Failure case: reasoning task but schema has no dependency structure
    
    Example:
        schema = {"type": "object", "properties": {"field1": ..., "field2": ...}}
            (flat, no nesting)
        prompt = "Analyze dependencies between these concepts"
        → WARNING: schema too flat for reasoning task
    
    Parameters
    ----------
    schema: dict
        JSON schema
    prompt: str
        Task prompt
    
    Returns
    -------
    SemanticFailureMode or None
        Failure mode if detected; None otherwise
    """
    has_reasoning = detect_reasoning_intent(prompt)
    
    if not has_reasoning:
        return None
    
    # Check schema depth
    depth = _compute_schema_depth(schema)
    
    if depth <= 1:
        return SemanticFailureMode(
            mode_type="reasoning_with_flat_schema",
            severity="soft",
            confidence=0.7,
            explanation=(
                f"Prompt has reasoning intent, but schema is flat (depth={depth}). "
                f"Schema may not support complex reasoning output. "
                f"Consider adding structured fields for reasoning components "
                f"(e.g., 'analysis', 'reasoning', 'conclusion')."
            ),
        )
    
    return None


def detect_mislabeled_extraction(
    prompt: str,
) -> SemanticFailureMode | None:
    """Detect if prompt claims extraction but actually needs reasoning.
    
    Failure case: prompt says "extract" but contains reasoning keywords
    
    Example:
        prompt = "Extract and analyze the implications of..."
        → WARNING: mixed extraction + reasoning
    
    Parameters
    ----------
    prompt: str
        Task prompt
    
    Returns
    -------
    SemanticFailureMode or None
        Failure mode if detected; None otherwise
    """
    is_extraction_only = detect_extraction_only(prompt)
    has_reasoning = detect_reasoning_intent(prompt)
    
    if is_extraction_only and has_reasoning:
        return SemanticFailureMode(
            mode_type="mislabeled_extraction",
            severity="soft",
            confidence=0.6,
            explanation=(
                "Prompt says 'extract' but contains reasoning verbs. "
                "Prompt may be ambiguous or mislabeled. "
                "TTF routing may be more appropriate than simple extraction."
            ),
        )
    
    return None


def _compute_schema_depth(schema: dict, _depth: int = 0) -> int:
    """Compute max nesting depth of JSON schema."""
    if not isinstance(schema, dict):
        return _depth
    
    candidates = [_depth]
    
    for key in ("properties", "items", "additionalProperties"):
        child = schema.get(key)
        if isinstance(child, dict):
            for v in child.values():
                candidates.append(_compute_schema_depth(v, _depth + 1))
        elif child is not None:
            candidates.append(_compute_schema_depth(child, _depth + 1))
    
    for key in ("anyOf", "oneOf", "allOf"):
        for item in schema.get(key, []):
            candidates.append(_compute_schema_depth(item, _depth + 1))
    
    return max(candidates) if candidates else _depth
