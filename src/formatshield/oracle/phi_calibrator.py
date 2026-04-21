"""
Self-calibrating routing threshold estimator over (lambda2, tau, delta_k) feature space.

Calibrates the routing threshold as a function of schema feature context rather
than a fixed scalar.  Uses Nadaraya-Watson kernel regression — a non-parametric
estimator equivalent to a Gaussian Process with uniform prior — over a rolling
window of observed routing outcomes.

Key properties
--------------
* **3D feature space**: operates on (lambda2, tau, delta_k) feature vectors, not the
  scalar Phi alone.  Schemas with similar feature profiles share routing lessons
  automatically (cross-schema transfer).

* **RBF kernel**: k(x, x') = exp(-||x - x'||^2 / (2 l^2)).  Similar schemas get
  high weight; dissimilar schemas contribute negligibly.

* **Rolling window**: fixed-size deque (default 500 observations).  Recent
  traffic drives calibration; distribution shift is tracked automatically.

* **Cold-start guard**: calibration is skipped until at least ``min_samples``
  observations (default 20) are collected.  Returns Oracle-X baseline threshold.

* **No pkl / no model file**: the estimator lives entirely in memory.  No
  artifacts, no stale-file issues, no training pipeline.

Public API
----------
- :class:`PhiComponents` — (lambda2, tau, delta_k) feature vector for one request
- :class:`OutcomeRecord` — stored routing outcome observation
- :class:`PhiOracleCalibrator` — the self-calibrating oracle
- :func:`build_phi_calibrator` — factory with sensible defaults
"""

from __future__ import annotations

import collections
import dataclasses
import logging
import math
import threading
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Lambda2-squared weight — mirrors routing_score._A (half-point at lambda2=0.5)
_WEIGHT_A: float = math.log(2) / (0.25**2)

#: Tau * lambda2 interaction weight — mirrors routing_score._B
_WEIGHT_B: float = math.log(2) / 0.50

#: DeltaK weight — mirrors routing_score._C (half-point at delta_k=0.50)
_WEIGHT_C: float = math.log(2) / 0.50

#: Oracle-X baseline routing threshold — fallback during cold start
DEFAULT_THRESHOLD: float = 0.65

#: Safety floor: never push threshold below this (would route nearly everything to TTF)
MIN_THRESHOLD: float = 0.30

#: Safety ceiling: never push threshold above this (would route nearly everything to direct)
MAX_THRESHOLD: float = 0.95

#: Default rolling window — large enough for production traffic diversity
DEFAULT_WINDOW_SIZE: int = 500

#: Minimum observations before calibration activates (cold-start guard)
DEFAULT_MIN_SAMPLES: int = 20

#: Quality outcome threshold defining a "good" routing decision
DEFAULT_TARGET_ACCURACY: float = 0.80

#: RBF length scale l — controls how fast influence decays with feature distance
#: At l=0.30, two schemas must be within ~0.30 units to share >50% weight
DEFAULT_LENGTH_SCALE: float = 0.30

#: Binary search iterations — 30 gives precision ~(0.95-0.30)/2^30 ≈ 6e-10
_BINARY_SEARCH_ITERS: int = 30

#: Kernel weight floor below which unweighted fallback activates
_WEIGHT_FLOOR: float = 1e-10


# ---------------------------------------------------------------------------
# Feature vector
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PhiComponents:
    """Three-dimensional feature vector for a single routing decision.

    Parameters
    ----------
    lambda2:
        Normalized Fiedler value lambda2 in [0, 1].  Measures schema graph
        algebraic connectivity — higher means more cross-field dependencies.
    tau:
        Schema constraint tightness tau in [0, 1].  Entropy proxy for how
        tightly constrained the output space is.
    delta_k:
        NCD prompt-schema alignment gap delta_k in [0, 1].  Measures how
        semantically distant the prompt is from the schema.
    """

    lambda2: float
    tau: float
    delta_k: float

    def as_tuple(self) -> tuple[float, float, float]:
        """Return the feature vector as a plain (lambda2, tau, delta_k) tuple."""
        return (self.lambda2, self.tau, self.delta_k)


# ---------------------------------------------------------------------------
# Observation record
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class OutcomeRecord:
    """One routing outcome observation stored in the calibrator window.

    Parameters
    ----------
    phi_components:
        Feature vector (lambda2, tau, delta_k) for the request.
    phi_score:
        Pre-computed Phi value derived from ``phi_components``.
    used_ttf:
        True if TTF (Think-Then-Format) generation was used for this request.
    quality_outcome:
        Observed generation quality in [0, 1].  Values >= ``target_accuracy``
        are treated as successful routing decisions.
    """

    phi_components: PhiComponents
    phi_score: float
    used_ttf: bool
    quality_outcome: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _phi_from_components(lambda2: float, tau: float, delta_k: float) -> float:
    """Compute Phi directly from (lambda2, tau, delta_k) components.

    Uses the same formula as ``routing_score.compute_routing_score`` so that
    phi_score stored in :class:`OutcomeRecord` matches what Oracle-X would
    compute for the same schema and prompt.
    """
    exponent = _WEIGHT_A * lambda2**2 + _WEIGHT_B * tau * lambda2 + _WEIGHT_C * delta_k
    return max(0.0, min(1.0, 1.0 - math.exp(-exponent)))


def _rbf_kernel(
    x: tuple[float, float, float],
    y: tuple[float, float, float],
    length_scale: float,
) -> float:
    """Radial basis function (squared exponential) kernel between two 3D feature vectors.

    k(x, y) = exp(-||x - y||^2 / (2 * length_scale^2))

    Returns 1.0 when x == y, decaying toward 0.0 as Euclidean distance grows.

    Args:
        x: First feature vector (lambda2, tau, delta_k).
        y: Second feature vector (lambda2, tau, delta_k).
        length_scale: Controls the radius of influence — larger values
            allow more distant schemas to share calibration signal.

    Returns:
        Kernel value in (0, 1].
    """
    sq_dist = sum((a - b) ** 2 for a, b in zip(x, y, strict=True))
    return math.exp(-sq_dist / (2.0 * length_scale**2))


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------


class PhiOracleCalibrator:
    """Self-calibrating routing threshold estimator using Nadaraya-Watson regression.

    Learns the optimal routing threshold as a function of schema feature context
    (lambda2, tau, delta_k) rather than keeping a single global scalar.  Schemas
    with similar features automatically share calibration signal through the RBF
    kernel — no manual grouping or schema tagging required.

    Parameters
    ----------
    window_size:
        Maximum number of recent observations retained in the rolling window.
    min_samples:
        Minimum observations required before calibration activates.
        Below this count, :meth:`calibrate_threshold` returns ``default_threshold``.
    target_accuracy:
        Quality outcome value that defines a "good" routing decision.
        The calibrated threshold is the lowest Phi at which TTF quality
        (kernel-weighted, conditioned on context) reaches this target.
    length_scale:
        RBF kernel length scale.  Smaller values make calibration more local;
        larger values allow distant schemas to share signal.
    default_threshold:
        Oracle-X baseline threshold returned during cold start.
    """

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        target_accuracy: float = DEFAULT_TARGET_ACCURACY,
        length_scale: float = DEFAULT_LENGTH_SCALE,
        default_threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self._window_size = window_size
        self._min_samples = min_samples
        self._target_accuracy = target_accuracy
        self._length_scale = length_scale
        self._default_threshold = default_threshold
        self._window: collections.deque[OutcomeRecord] = collections.deque(maxlen=window_size)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def sample_count(self) -> int:
        """Number of observations currently in the rolling window."""
        with self._lock:
            return len(self._window)

    def record_outcome(
        self,
        phi_components: PhiComponents,
        routing_decision: bool,
        quality_outcome: float,
    ) -> None:
        """Record a routing outcome for a single request.

        Appends the observation to the rolling window.  If the window is full,
        the oldest observation is evicted automatically.

        Args:
            phi_components:
                Feature vector (lambda2, tau, delta_k) for the request.
            routing_decision:
                True if TTF was used; False if direct generation was used.
            quality_outcome:
                Observed output quality in [0, 1].  Values outside this range
                are clamped before storage.
        """
        phi_score = _phi_from_components(
            phi_components.lambda2,
            phi_components.tau,
            phi_components.delta_k,
        )
        record = OutcomeRecord(
            phi_components=phi_components,
            phi_score=phi_score,
            used_ttf=routing_decision,
            quality_outcome=max(0.0, min(1.0, quality_outcome)),
        )
        with self._lock:
            self._window.append(record)

    def calibrate_threshold(
        self,
        lambda2: float,
        tau: float,
        delta_k: float,
    ) -> float:
        """Return the calibrated routing threshold for this schema feature region.

        Performs a binary search over [MIN_THRESHOLD, MAX_THRESHOLD] for the
        lowest Phi threshold at which kernel-weighted TTF quality (conditioned on
        the query features) meets or exceeds ``target_accuracy``.  A lower result
        means TTF is confidently useful in this region; a higher result means TTF
        should be reserved for only the most complex schemas.

        Falls back to ``default_threshold`` when fewer than ``min_samples``
        observations are available (cold-start guard) or when no TTF records
        exist in the window.

        Args:
            lambda2: Normalized Fiedler value for the query schema.
            tau:     Constraint tightness for the query schema.
            delta_k: NCD alignment gap for the query.

        Returns:
            Calibrated Phi threshold in [MIN_THRESHOLD, MAX_THRESHOLD].
        """
        with self._lock:
            records = list(self._window)

        if len(records) < self._min_samples:
            logger.debug(
                "PhiOracleCalibrator: cold start (n=%d < %d) — returning default=%.3f",
                len(records),
                self._min_samples,
                self._default_threshold,
            )
            return self._default_threshold

        context = (lambda2, tau, delta_k)

        # Assess quality at the broadest TTF usage point
        q_at_floor = self._expected_ttf_quality(records, context, MIN_THRESHOLD)
        if q_at_floor is None:
            # No TTF records at all — cannot calibrate
            logger.debug(
                "PhiOracleCalibrator: no TTF records in window — returning default=%.3f",
                self._default_threshold,
            )
            return self._default_threshold

        if q_at_floor >= self._target_accuracy:
            # TTF is good even when used broadly — route everything to TTF
            logger.debug(
                "PhiOracleCalibrator: TTF quality=%.3f >= target=%.3f at floor — threshold=%.3f",
                q_at_floor,
                self._target_accuracy,
                MIN_THRESHOLD,
            )
            return MIN_THRESHOLD

        # Assess quality at the most restrictive point
        q_at_ceil = self._expected_ttf_quality(records, context, MAX_THRESHOLD)
        if q_at_ceil is None or q_at_ceil < self._target_accuracy:
            # TTF quality never reaches target even when used very selectively
            logger.debug(
                "PhiOracleCalibrator: TTF quality below target even at ceiling — threshold=%.3f",
                MAX_THRESHOLD,
            )
            return MAX_THRESHOLD

        # Binary search: find lowest threshold where TTF quality >= target_accuracy
        # Invariant: quality(lo) < target <= quality(hi)
        lo, hi = MIN_THRESHOLD, MAX_THRESHOLD
        for _ in range(_BINARY_SEARCH_ITERS):
            mid = (lo + hi) / 2.0
            q = self._expected_ttf_quality(records, context, mid)
            if q is None or q < self._target_accuracy:
                lo = mid  # quality below target — push threshold up
            else:
                hi = mid  # quality meets target — try lower threshold

        threshold = max(MIN_THRESHOLD, min(MAX_THRESHOLD, hi))
        logger.debug(
            "PhiOracleCalibrator: calibrated threshold=%.4f for (l2=%.3f t=%.3f dk=%.3f) n=%d",
            threshold,
            lambda2,
            tau,
            delta_k,
            len(records),
        )
        return threshold

    def cross_schema_transfer(self, similar_schema_profile: PhiComponents) -> float:
        """Predict expected quality for a schema with the given feature profile.

        Schemas with similar (lambda2, tau, delta_k) feature vectors automatically
        share routing lessons via the RBF kernel — no explicit grouping required.
        If the window is empty, returns ``target_accuracy`` as a neutral prior.
        If all kernel weights are near zero (query far from all observations),
        returns the unweighted global average.

        Args:
            similar_schema_profile:
                Feature vector of the schema to predict quality for.

        Returns:
            Predicted quality outcome in [0, 1].
        """
        with self._lock:
            records = list(self._window)

        if not records:
            return self._target_accuracy

        context = similar_schema_profile.as_tuple()
        total_weight = 0.0
        total_quality = 0.0

        for rec in records:
            w = _rbf_kernel(context, rec.phi_components.as_tuple(), self._length_scale)
            total_weight += w
            total_quality += w * rec.quality_outcome

        if total_weight < _WEIGHT_FLOOR:
            # No nearby observations — fall back to global average
            return sum(r.quality_outcome for r in records) / len(records)

        return total_quality / total_weight

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of calibrator state for logging and observability.

        Returns:
            dict with keys: sample_count, is_active, window_size, min_samples,
            target_accuracy, length_scale, default_threshold.
        """
        with self._lock:
            sample_count = len(self._window)

        return {
            "sample_count": sample_count,
            "is_active": sample_count >= self._min_samples,
            "window_size": self._window_size,
            "min_samples": self._min_samples,
            "target_accuracy": self._target_accuracy,
            "length_scale": self._length_scale,
            "default_threshold": self._default_threshold,
        }

    def reset(self) -> None:
        """Clear all observations and return the calibrator to cold-start state."""
        with self._lock:
            self._window.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _expected_ttf_quality(
        self,
        records: list[OutcomeRecord],
        context: tuple[float, float, float],
        threshold: float,
    ) -> float | None:
        """Kernel-weighted expected quality for TTF observations with Phi >= threshold.

        Args:
            records:   Full observation window snapshot (caller holds no lock).
            context:   Query feature vector (lambda2, tau, delta_k).
            threshold: Phi threshold — only TTF records above this are included.

        Returns:
            Kernel-weighted average quality, or None when no qualifying records exist.
            Falls back to unweighted average when all kernel weights are near zero.
        """
        ttf_records = [r for r in records if r.used_ttf and r.phi_score >= threshold]
        if not ttf_records:
            return None

        total_weight = 0.0
        total_quality = 0.0
        for rec in ttf_records:
            w = _rbf_kernel(context, rec.phi_components.as_tuple(), self._length_scale)
            total_weight += w
            total_quality += w * rec.quality_outcome

        if total_weight < _WEIGHT_FLOOR:
            # Context is far from all TTF observations — unweighted fallback
            return sum(r.quality_outcome for r in ttf_records) / len(ttf_records)

        return total_quality / total_weight


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_phi_calibrator(
    window_size: int = DEFAULT_WINDOW_SIZE,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    target_accuracy: float = DEFAULT_TARGET_ACCURACY,
    length_scale: float = DEFAULT_LENGTH_SCALE,
    default_threshold: float = DEFAULT_THRESHOLD,
) -> PhiOracleCalibrator:
    """Build a :class:`PhiOracleCalibrator` with the given configuration.

    Args:
        window_size:       Maximum observations to retain in the rolling window.
        min_samples:       Observations required before calibration activates.
        target_accuracy:   Quality value defining a "good" routing decision.
        length_scale:      RBF kernel length scale for cross-schema influence radius.
        default_threshold: Fallback threshold during cold start (Oracle-X baseline).

    Returns:
        Configured :class:`PhiOracleCalibrator` ready to receive observations.
    """
    return PhiOracleCalibrator(
        window_size=window_size,
        min_samples=min_samples,
        target_accuracy=target_accuracy,
        length_scale=length_scale,
        default_threshold=default_threshold,
    )
