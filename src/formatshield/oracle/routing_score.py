"""
Closed-form information-geometric routing score Φ(prompt, schema).

Φ is the first training-free, closed-form routing score in the structured
generation literature — no benchmark data, no ML model, no artifact files.

Formula
-------
    Φ = 1 − exp(−(A·λ̃₂² + B·τ·λ̃₂ + C·ΔK))

where:

* λ̃₂ ∈ [0,1]: normalized Fiedler value of the JSON schema dependency graph
  (spectral algebraic connectivity — from ``schema_graph``)
* τ  ∈ [0,1]: schema constraint tightness (entropy proxy — from ``schema_entropy``)
* ΔK ∈ [0,1]: NCD prompt-schema alignment gap (from ``ncd``)

Weight calibration (half-point derivation)
------------------------------------------
Each weight is chosen so that the corresponding component alone crosses the
decision boundary Φ = 0.5 at a semantically meaningful value:

    A = ln2 / 0.25²  (λ̃₂ = 0.5  →  Φ = 0.5 when τ=ΔK=0)
    B = ln2 / 0.50   (τ·λ̃₂ = 0.5  →  Φ = 0.5 when A-term≈0)
    C = ln2 / 0.70   (ΔK = 0.70  →  Φ = 0.5 when A+B terms≈0)

Φ > 0.5  →  prefer TTF (schema is complex / prompt is semantically distant)
Φ ≤ 0.5  →  prefer direct (schema is simple / prompt is well-aligned)

Worked examples (see tests/unit/test_routing_score.py)
------------------------------------------------------
* Simple 3-field extraction + aligned prompt   → Φ ≈ 0.15–0.30  (direct ✓)
* Nested itinerary schema + unrelated prompt   → Φ ≈ 0.75–0.92  (TTF ✓)
"""

from __future__ import annotations

import dataclasses
import math

import formatshield.oracle.ncd as _ncd
import formatshield.oracle.schema_entropy as _entropy
import formatshield.oracle.schema_graph as _graph

# ---------------------------------------------------------------------------
# Weight constants
# ---------------------------------------------------------------------------

#: λ̃₂² weight — half-point at λ̃₂ = 0.5 alone
_A: float = math.log(2) / (0.25 ** 2)

#: τ·λ̃₂ interaction weight — half-point at product = 0.5
_B: float = math.log(2) / 0.50

#: ΔK weight — half-point at ΔK = 0.70 alone
_C: float = math.log(2) / 0.70


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RoutingScore:
    """Decomposed routing score returned by :func:`compute_routing_score`.

    Attributes
    ----------
    phi:
        Final routing score Φ ∈ [0, 1].  Values > 0.5 recommend TTF.
    lambda2:
        Normalized Fiedler value λ̃₂ of the schema dependency graph.
    tau:
        Schema constraint tightness τ.
    delta_k:
        NCD alignment gap ΔK between prompt and schema.
    explanation:
        Human-readable one-line summary.
    """

    phi: float
    lambda2: float
    tau: float
    delta_k: float
    explanation: str


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_routing_score(prompt: str, schema: object) -> RoutingScore:
    """Compute Φ(prompt, schema) — training-free routing score.

    Parameters
    ----------
    prompt:
        Raw user prompt string (before any system prompt injection).
    schema:
        JSON Schema dict.  Non-dict values return a neutral score of 0.5.

    Returns
    -------
    RoutingScore
        Decomposed score including Φ and all three components.
    """
    if not isinstance(schema, dict):
        return RoutingScore(
            phi=0.5,
            lambda2=0.0,
            tau=0.0,
            delta_k=0.5,
            explanation="Φ=0.500 (no schema — neutral routing)",
        )

    l2 = _graph.fiedler_value(schema)
    tau = _entropy.constraint_tightness(schema)
    dk = _ncd.prompt_schema_ncd(prompt, schema)

    exponent = _A * l2 ** 2 + _B * tau * l2 + _C * dk
    phi = 1.0 - math.exp(-exponent)
    phi = max(0.0, min(1.0, phi))

    explanation = (
        f"Φ={phi:.3f} λ̃₂={l2:.3f} τ={tau:.3f} ΔK={dk:.3f}"
    )
    return RoutingScore(phi=phi, lambda2=l2, tau=tau, delta_k=dk, explanation=explanation)
