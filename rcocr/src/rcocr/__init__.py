"""
RCOCR — Reasoning-Compatible Output Constraint Recovery.

A minimal, zero-dependency two-pass generation protocol for structured
output from reasoning-capable language models.

The RCOCR architecture solves the grammar-reasoning conflict that occurs
when FSM-based structured output constraints are applied simultaneously
with chain-of-thought / extended thinking:

  Pass 1 (unconstrained): model reasons freely in <think>...</think>
  Pass 2 (constrained):   model formats the final JSON conditioned on its reasoning

This decoupling preserves full reasoning quality while guaranteeing
schema-valid output.

Quick start::

    from rcocr import RCOCREngine

    engine = RCOCREngine(backend=my_backend)
    thinking, output = await engine.generate(
        prompt="Extract order details from this invoice.",
        schema={"type": "object", "properties": {"order_id": {"type": "string"}}},
    )

References
----------
- CRANE (arXiv:2502.09061) — reasoning + structured output
- Self-Consistency CoT (arXiv:2203.11171)
- vLLM issue #34650 — reasoning + structured output conflict
- llama.cpp issue #12204 — grammar disabled with reasoning_format
"""

from __future__ import annotations

from rcocr.engine import RCOCREngine
from rcocr.protocol import RCOCRBackend, StreamingRCOCRBackend

__version__ = "0.1.0"

__all__ = [
    "RCOCRBackend",
    "RCOCREngine",
    "StreamingRCOCRBackend",
    "__version__",
]
