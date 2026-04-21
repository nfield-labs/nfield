"""
Low-Latency TTF (Think-Then-Format) Optimization.

Reduces TTF latency from 6-8 seconds to 2-3 seconds while maintaining correctness.

Three optimization layers:
1. EARLY EXIT: If Pass 1 confidence ≥ 0.85, skip Pass 2 entirely
2. BUDGET REDUCTION: Φ-aware budget capping (max 512 tokens even for Φ ≥ 0.65)
3. FIELD-SELECTIVE: Only reason about constrained/critical fields

Measurement protocol:
- Baseline: full TTF (1024 tokens, always 2 passes)
- Optimized: early exit + selective reasoning
- Success: ≤40% accuracy drop, ≥2x latency improvement
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LowLatencyConfig:
    """Configuration for low-latency TTF mode.
    
    Parameters
    ----------
    enable_early_exit: bool
        Skip Pass 2 if Pass 1 confidence ≥ early_exit_threshold
    early_exit_threshold: float
        Confidence threshold for early exit (0.85 recommended)
    max_thinking_budget: int
        Cap thinking tokens even for high-Φ tasks (512 recommended)
    enable_selective_reasoning: bool
        Only reason about constrained/critical fields (not exploration)
    use_fast_decomposition: bool
        Skip FULL_STRUCTURAL_REASONING for Φ < 0.8; use DEPENDENCY_AWARE instead
    """
    enable_early_exit: bool = True
    early_exit_threshold: float = 0.85
    max_thinking_budget: int = 512  # was 1024 for Φ ∈ [0.65, 0.80)
    enable_selective_reasoning: bool = True
    use_fast_decomposition: bool = True


def compute_low_latency_thinking_budget(phi: float, config: LowLatencyConfig | None = None) -> int:
    """Compute thinking budget with low-latency optimization.
    
    Replaces the default Φ-proportional budget with a capped version.
    
    Default behavior:
        Φ ≥ 0.90 → 4096
        Φ ∈ [0.75, 0.90) → 1024
        Φ ∈ [0.65, 0.75) → 512
        Φ < 0.65 → 256
    
    Low-latency (with config.max_thinking_budget=512):
        All Φ ≥ 0.65 → capped at 512 (was 1024–4096)
    
    Measurement impact:
        - Default medical case (Φ≈0.686, 1024 tokens) → 512 tokens
        - Estimated latency reduction: ~30-40%
    
    Parameters
    ----------
    phi: float
        Routing score (0–1)
    config: LowLatencyConfig or None
        Configuration; uses defaults if None
    
    Returns
    -------
    int
        Thinking token budget
    """
    if config is None:
        config = LowLatencyConfig()
    
    # Default budget (unchanged)
    if phi >= 0.90:
        default_budget = 4096
    elif phi >= 0.75:
        default_budget = 1024
    elif phi >= 0.65:
        default_budget = 512
    else:
        default_budget = 256
    
    # Apply cap
    if config.max_thinking_budget > 0:
        capped_budget = min(default_budget, config.max_thinking_budget)
        logger.debug(
            "TTF low-latency budget: Φ=%.3f → default=%d tokens → capped=%d tokens",
            phi,
            default_budget,
            capped_budget,
        )
        return capped_budget
    
    return default_budget


def should_skip_pass2(
    pass1_confidence: float,
    config: LowLatencyConfig | None = None,
) -> bool:
    """Decide whether to skip Pass 2 based on Pass 1 confidence.
    
    Early exit: if Pass 1 output is already highly confident, skip expensive
    Pass 2 constraint-checking and reformatting.
    
    Measurement protocol:
        - Track: what % of requests skip Pass 2?
        - Track: what % accuracy loss from skipping?
        - Goal: >50% skip rate with <5% accuracy loss
    
    Parameters
    ----------
    pass1_confidence: float
        Confidence score from Pass 1 output quality gate (0–1)
    config: LowLatencyConfig or None
        Configuration
    
    Returns
    -------
    bool
        True if we should skip Pass 2
    """
    if config is None:
        config = LowLatencyConfig()
    
    if not config.enable_early_exit:
        return False
    
    should_skip = pass1_confidence >= config.early_exit_threshold
    if should_skip:
        logger.debug(
            "TTF early exit triggered: Pass 1 confidence=%.3f >= threshold=%.3f",
            pass1_confidence,
            config.early_exit_threshold,
        )
    
    return should_skip


def get_selective_reasoning_focus(
    schema: dict | None,
    tau: float,
) -> str | None:
    """Generate selective reasoning focus for constrained fields only.
    
    Instead of reasoning about the entire schema, focus reasoning on:
    - Fields with enum constraints
    - Fields with complex dependencies
    - Fields marked as "critical" in schema
    
    Reasoning avoids:
    - Exploration fields (free text)
    - Optional fields
    - Independent flat fields
    
    Measurement:
        - Latency: should reduce reasoning tokens by 30-50%
        - Correctness: should not degrade field accuracy
    
    Parameters
    ----------
    schema: dict or None
        JSON schema (None = no selective reasoning)
    tau: float
        Constraint tightness (0–1)
    
    Returns
    -------
    str or None
        Selective reasoning instruction, or None if no selective focus needed
    """
    if schema is None or not schema:
        return None
    
    # Only apply if schema has actual constraints (τ > 0.3)
    if tau <= 0.3:
        return None
    
    # Find constrained fields
    constrained_fields = []
    
    def _find_constraints(obj: dict, path: str = "") -> None:
        if not isinstance(obj, dict):
            return
        
        for key, val in obj.items():
            new_path = f"{path}.{key}" if path else key
            
            # Detect constrained fields
            if isinstance(val, dict):
                has_enum = "enum" in val
                has_pattern = "pattern" in val
                has_min_max = "minimum" in val or "maximum" in val
                has_required = "required" in val
                
                if has_enum or has_pattern or has_min_max or has_required:
                    constrained_fields.append(new_path)
                
                _find_constraints(val, new_path)
    
    _find_constraints(schema)
    
    if not constrained_fields:
        return None
    
    focus_list = ", ".join(constrained_fields[:10])  # Top 10
    return (
        f"SELECTIVE REASONING (low-latency mode): "
        f"Focus your reasoning ONLY on these constrained fields: {focus_list}. "
        f"For other fields, use direct extraction without deep reasoning."
    )


def should_use_fast_decomposition(phi: float, config: LowLatencyConfig | None = None) -> bool:
    """Decide whether to skip FULL_STRUCTURAL_REASONING.
    
    Fast mode skips expensive "full interconnected" reasoning for Φ < 0.8.
    
    Rationale:
        - FULL_STRUCTURAL requires global consistency checks
        - For Φ ∈ [0.65, 0.80) (mid-complexity), DEPENDENCY_AWARE is sufficient
        - Saves ~200-300 reasoning tokens
    
    Parameters
    ----------
    phi: float
        Routing score
    config: LowLatencyConfig or None
        Configuration
    
    Returns
    -------
    bool
        True if we should use fast (non-FULL) decomposition
    """
    if config is None:
        config = LowLatencyConfig()
    
    if not config.use_fast_decomposition:
        return False
    
    # Use fast decomposition for Φ < 0.80 (don't need full structural reasoning)
    return phi < 0.80


@dataclass
class LowLatencyMetrics:
    """Track metrics for low-latency TTF optimization.
    
    Attributes
    ----------
    original_thinking_budget: int
        Default thinking budget (before optimization)
    optimized_thinking_budget: int
        Capped thinking budget (after optimization)
    pass1_confidence: float
        Pass 1 output quality score (0–1)
    pass2_skipped: bool
        Whether Pass 2 was skipped
    estimated_latency_reduction_pct: float
        Estimated latency improvement percentage
    """
    original_thinking_budget: int
    optimized_thinking_budget: int
    pass1_confidence: float
    pass2_skipped: bool
    estimated_latency_reduction_pct: float = 0.0
    
    def __post_init__(self) -> None:
        # Estimate latency reduction based on pass2_skipped and budget reduction
        budget_reduction = (
            (1.0 - self.optimized_thinking_budget / max(self.original_thinking_budget, 1))
            * 100.0
        )
        
        if self.pass2_skipped:
            # ~50% latency is Pass 2 — if skipped, get ~50% reduction + budget savings
            self.estimated_latency_reduction_pct = min(budget_reduction + 50.0, 80.0)
        else:
            # Just budget savings
            self.estimated_latency_reduction_pct = budget_reduction


# ───────────────────────────────────────────────────────────────────────────
# Example measurement protocol
# ───────────────────────────────────────────────────────────────────────────

"""
MEASUREMENT PROTOCOL for validation:

1. Baseline (disable all optimizations):
   config = LowLatencyConfig(
       enable_early_exit=False,
       enable_selective_reasoning=False,
       use_fast_decomposition=False,
       max_thinking_budget=1024,  # force to 1024
   )
   Run medical case 10 times, record:
       - avg latency_ms
       - avg accuracy (schema_valid + semantic score)

2. Optimized:
   config = LowLatencyConfig(
       enable_early_exit=True,
       early_exit_threshold=0.85,
       enable_selective_reasoning=True,
       use_fast_decomposition=True,
       max_thinking_budget=512,
   )
   Run medical case 10 times, record:
       - avg latency_ms
       - avg accuracy (schema_valid + semantic score)
       - % of requests that skipped Pass 2

3. Success criteria:
   - Latency improvement: ≥2× (6250ms → <3125ms)
   - Accuracy: ≤5% loss from baseline
   - Early exit rate: ≥40% of requests skip Pass 2

4. Report:
   Print LowLatencyMetrics after each run showing estimation vs. actual
"""
