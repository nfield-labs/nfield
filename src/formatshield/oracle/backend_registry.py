"""
FormatShield backend capability registry.

Stores backend-model capabilities and calibration metadata in one JSON file
(``oracle_data/backend_registry.json``) so adding a new provider is **config
plus benchmark, not code surgery**.

Usage::

    from formatshield.oracle.backend_registry import BackendRegistry

    reg = BackendRegistry.load()
    cap = reg.get("groq", "llama-3.1-8b-instant")
    print(cap.ttf_overhead_pct)   # 30.0
    print(cap.native_thinker)     # False

The registry is loaded once and cached.  Call :func:`reload` to pick up
edits without restarting the process.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

#: Default registry path shipped with the package.
_DEFAULT_REGISTRY_PATH: Path = (
    Path(__file__).parent / "oracle_data" / "backend_registry.json"
)


@dataclass
class BackendCapability:
    """Capabilities and calibration metadata for one backend/model combination.

    Parameters
    ----------
    backend_id:
        Lowercase backend name, e.g. ``"groq"``.
    model_id:
        Bare model name.  ``"*"`` for the backend-level defaults.
    supports_ttf:
        Whether the backend supports two-pass Think-Then-Format routing.
    native_thinker:
        If ``True`` the model has built-in chain-of-thought; always route
        direct — TTF overhead would be pure waste.
    ttf_overhead_pct:
        Expected latency overhead of a TTF pass vs direct, as a percentage.
    cost_per_1k_input_tokens:
        Cost in USD per 1 000 input tokens (0.0 for local/self-hosted).
    cost_per_1k_output_tokens:
        Cost in USD per 1 000 output tokens (0.0 for local/self-hosted).
    max_context_tokens:
        Maximum context window in tokens.
    calibration_date:
        ISO date string when these numbers were last measured.
    notes:
        Free-form annotation for humans.
    """

    backend_id: str
    model_id: str
    supports_ttf: bool = True
    native_thinker: bool = False
    ttf_overhead_pct: float = 40.0
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0
    max_context_tokens: int = 32768
    calibration_date: str = ""
    notes: str = ""

    @property
    def cost_per_token(self) -> float:
        """Mean cost per single token (input + output averaged)."""
        return (self.cost_per_1k_input_tokens + self.cost_per_1k_output_tokens) / 2000.0


class BackendRegistry:
    """In-memory view of ``backend_registry.json``.

    Instantiate via :meth:`load` (uses the bundled default) or
    :meth:`load_from` (for custom paths or tests).

    Capability resolution order
    ---------------------------
    1. ``backends[backend_id].models[model_id]`` — most specific
    2. ``backends[backend_id]`` defaults — backend-level fallback
    3. ``backends["default"]`` — universal fallback
    """

    def __init__(self, data: dict) -> None:
        self._data = data
        self._backends: dict[str, dict] = data.get("backends", {})

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def load(cls) -> BackendRegistry:
        """Load the bundled registry from ``oracle_data/backend_registry.json``."""
        return cls.load_from(_DEFAULT_REGISTRY_PATH)

    @classmethod
    def load_from(cls, path: str | Path) -> BackendRegistry:
        """Load the registry from a custom path."""
        p = Path(path)
        if not p.exists():
            logger.warning("BackendRegistry: file not found at %s — using empty registry.", p)
            return cls({"backends": {}})
        try:
            with p.open(encoding="utf-8") as fh:
                data = json.load(fh)
            n = len(data.get("backends", {}))
            logger.debug("BackendRegistry: loaded %d backends from %s", n, p)
            return cls(data)
        except Exception:
            logger.warning(
                "BackendRegistry: failed to parse %s — using empty registry.", p, exc_info=True
            )
            return cls({"backends": {}})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, backend_id: str, model_id: str = "*") -> BackendCapability:
        """Return the :class:`BackendCapability` for a backend/model pair.

        Falls back through model → backend → default in that order.
        """
        backend_id = backend_id.lower()
        backend_data = self._backends.get(backend_id, self._backends.get("default", {}))
        model_data = backend_data.get("models", {}).get(model_id, {})

        # Merge: model overrides backend, backend fills gaps
        merged = {**backend_data, **model_data}
        merged.pop("models", None)  # don't include the sub-dict

        return BackendCapability(
            backend_id=backend_id,
            model_id=model_id,
            supports_ttf=bool(merged.get("supports_ttf", True)),
            native_thinker=bool(merged.get("native_thinker", False)),
            ttf_overhead_pct=float(merged.get("ttf_overhead_pct", 40.0)),
            cost_per_1k_input_tokens=float(merged.get("cost_per_1k_input_tokens", 0.0)),
            cost_per_1k_output_tokens=float(merged.get("cost_per_1k_output_tokens", 0.0)),
            max_context_tokens=int(merged.get("max_context_tokens", 32768)),
            calibration_date=str(merged.get("calibration_date", "")),
            notes=str(merged.get("notes", "")),
        )

    def known_backends(self) -> list[str]:
        """Return all backend IDs present in the registry."""
        return [b for b in self._backends if b != "default"]

    def known_models(self, backend_id: str) -> list[str]:
        """Return all model IDs registered for *backend_id*."""
        return list(self._backends.get(backend_id.lower(), {}).get("models", {}).keys())

    def is_native_thinker(self, backend_id: str, model_id: str) -> bool:
        """Return ``True`` when *model_id* on *backend_id* is a native thinker."""
        return self.get(backend_id, model_id).native_thinker


# ---------------------------------------------------------------------------
# Module-level singleton — loaded lazily
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _cached_registry() -> BackendRegistry:
    return BackendRegistry.load()


def get_registry() -> BackendRegistry:
    """Return the module-level :class:`BackendRegistry` singleton.

    The registry is loaded once from ``backend_registry.json`` and cached.
    Call :func:`reload_registry` to pick up file edits at runtime.
    """
    return _cached_registry()


def reload_registry() -> BackendRegistry:
    """Clear the cache and reload ``backend_registry.json`` from disk."""
    _cached_registry.cache_clear()
    return _cached_registry()
