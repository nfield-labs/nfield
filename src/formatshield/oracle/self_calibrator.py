"""
Φ-Oracle Self-Calibration.

After each TTF generation call the engine records a data point:
``(phi_score, semantic_eval_score, quality_gate_passed)``.  Once the rolling
window reaches ``min_samples`` the calibrator fits a logistic curve on
``semantic_eval_score ≥ target_accuracy`` and extracts the Φ value at which
the probability of a "good outcome" crosses 0.5 — that becomes the new routing
threshold.

This allows FormatShield to self-correct its routing threshold from production
outcomes.  Domain shift (e.g. moving from synthetic benchmarks to real API
traffic) is handled automatically with no human tuning required.

Design
------
* **Rolling window** — fixed-size deque (default 200 observations).  Older
  observations are discarded as new ones arrive, so the threshold tracks
  recent distribution rather than all-time averages.

* **Pure-Python logistic fit** — uses only the standard library + ``math``.
  No ``scikit-learn`` required.  A simple binary search over [0, 1] finds the
  Φ value where the logistic probability P(good | Φ) = 0.5.

* **Minimum sample guard** — calibration is skipped until at least
  ``min_samples`` observations (default 20) are collected, preventing
  premature threshold collapse on cold-start data.

* **Persistence** — optionally write the calibrated threshold to a JSON file
  so it survives process restarts.  Load is automatic on construction.

Public API
----------
- :class:`CalibrationRecord` — one observation
- :class:`SelfCalibrator` — the calibrator
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import math
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_WINDOW_SIZE: int = 200
DEFAULT_MIN_SAMPLES: int = 20
DEFAULT_TARGET_ACCURACY: float = 0.80
_DEFAULT_PERSIST_PATH: Path = Path(__file__).parent / "oracle_data" / "calibrated_threshold.json"


# ---------------------------------------------------------------------------
# Observation record
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CalibrationRecord:
    """One TTF outcome observation.

    Parameters
    ----------
    phi:
        Routing score Φ for this request.
    semantic_eval_score:
        Semantic evaluation score ∈ [0, 1] for the TTF output.  When not
        available (e.g. no semantic evaluator in the pipeline), pass ``None``
        and the record will be skipped in calibration.
    quality_gate_passed:
        Whether Pass 1 quality gate passed.  Used as a secondary signal.
    used_ttf:
        Whether TTF was actually used for this request.
    """

    phi: float
    semantic_eval_score: float | None
    quality_gate_passed: bool | None = None
    used_ttf: bool = True


# ---------------------------------------------------------------------------
# Pure-Python logistic regression (single feature)
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _fit_logistic(
    xs: list[float],
    y: list[int],
    lr: float = 0.1,
    n_iter: int = 200,
) -> tuple[float, float]:
    """Fit logistic regression y ~ sigmoid(w*x + b) via gradient descent.

    Parameters
    ----------
    xs:
        Feature values (Φ scores).
    y:
        Binary labels (1 = good outcome, 0 = bad outcome).
    lr:
        Learning rate.
    n_iter:
        Number of gradient-descent iterations.

    Returns
    -------
    tuple[float, float]
        ``(weight, bias)`` of the fitted logistic function.
    """
    w, b = 0.0, 0.0
    n = len(xs)
    if n == 0:
        return w, b

    for _ in range(n_iter):
        dw, db = 0.0, 0.0
        for xi, yi in zip(xs, y, strict=True):
            pred = _sigmoid(w * xi + b)
            err = pred - yi
            dw += err * xi
            db += err
        w -= lr * dw / n
        b -= lr * db / n

    return w, b


def _threshold_from_logistic(w: float, b: float) -> float | None:
    """Find Φ where P(good | Φ) = 0.5, i.e. w*Φ + b = 0.

    Returns ``None`` if the weight is effectively zero (degenerate fit).
    """
    if abs(w) < 1e-6:
        return None
    threshold = -b / w
    # Clamp to a sane routing range
    return max(0.30, min(0.95, threshold))


# ---------------------------------------------------------------------------
# SelfCalibrator
# ---------------------------------------------------------------------------


class SelfCalibrator:
    """Rolling self-calibration of the Φ routing threshold.

    Parameters
    ----------
    window_size:
        Maximum number of recent observations to retain.
    min_samples:
        Minimum observations required before calibration is attempted.
    target_accuracy:
        Semantic eval score threshold that defines a "good" outcome.
        Observations with ``semantic_eval_score ≥ target_accuracy`` are
        labelled 1; others are labelled 0.
    persist_path:
        Optional file path for persisting the calibrated threshold.  Set to
        ``None`` to disable persistence.
    initial_threshold:
        Starting threshold before any calibration data is collected.

    Example
    -------
    .. code-block:: python

        calibrator = SelfCalibrator()

        # After each TTF call:
        calibrator.record(CalibrationRecord(
            phi=routing_score.phi,
            semantic_eval_score=eval_score,
            quality_gate_passed=gate_result.passed,
            used_ttf=True,
        ))

        # Get the current calibrated threshold:
        threshold = calibrator.current_threshold
    """

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        target_accuracy: float = DEFAULT_TARGET_ACCURACY,
        persist_path: Path | str | None = _DEFAULT_PERSIST_PATH,
        initial_threshold: float = 0.65,
    ) -> None:
        self._window_size = window_size
        self._min_samples = min_samples
        self._target_accuracy = target_accuracy
        self._persist_path: Path | None = Path(persist_path) if persist_path else None
        self._lock = threading.Lock()
        self._window: collections.deque[CalibrationRecord] = collections.deque(maxlen=window_size)
        self._current_threshold: float = initial_threshold
        self._calibration_count: int = 0

        # Try to load a previously persisted threshold
        self._load_persisted()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current_threshold(self) -> float:
        """The current calibrated (or initial) routing threshold."""
        return self._current_threshold

    @property
    def sample_count(self) -> int:
        """Number of observations in the rolling window."""
        with self._lock:
            return len(self._window)

    @property
    def calibration_count(self) -> int:
        """Number of times the threshold has been recalibrated."""
        return self._calibration_count

    def record(self, observation: CalibrationRecord) -> None:
        """Add an observation and attempt recalibration.

        Recalibration runs synchronously but is fast (pure Python, ≤200
        iterations on ≤200 points — typically sub-millisecond).

        Parameters
        ----------
        observation:
            The TTF outcome to record.
        """
        # Only TTF calls produce useful calibration signal
        if not observation.used_ttf:
            return
        # Skip records with no eval score
        if observation.semantic_eval_score is None:
            return

        with self._lock:
            self._window.append(observation)
            if len(self._window) >= self._min_samples:
                self._recalibrate()

    def stats(self) -> dict[str, Any]:
        """Return a stats snapshot for logging / observability."""
        with self._lock:
            return {
                "current_threshold": self._current_threshold,
                "sample_count": len(self._window),
                "window_size": self._window_size,
                "min_samples": self._min_samples,
                "target_accuracy": self._target_accuracy,
                "calibration_count": self._calibration_count,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recalibrate(self) -> None:
        """Refit the logistic model and update the threshold (caller holds lock)."""
        records = list(self._window)

        xs = [r.phi for r in records]
        y = [1 if (r.semantic_eval_score or 0) >= self._target_accuracy else 0 for r in records]

        # Need both classes to fit; if all outcomes are one class, skip
        if sum(y) == 0 or sum(y) == len(y):
            logger.debug(
                "SelfCalibrator: skipping recalibration — all labels are %d (need both 0 and 1)",
                y[0],
            )
            return

        w, b = _fit_logistic(xs, y)
        new_threshold = _threshold_from_logistic(w, b)

        if new_threshold is None:
            logger.debug("SelfCalibrator: degenerate fit (w≈0) — threshold unchanged")
            return

        old = self._current_threshold
        self._current_threshold = new_threshold
        self._calibration_count += 1

        logger.info(
            "SelfCalibrator: threshold recalibrated %.3f → %.3f (n=%d good=%d w=%.3f b=%.3f)",
            old,
            new_threshold,
            len(records),
            sum(y),
            w,
            b,
        )

        self._persist()

    def _persist(self) -> None:
        """Write current threshold to disk (caller holds lock)."""
        if self._persist_path is None:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "threshold": self._current_threshold,
                "calibration_count": self._calibration_count,
            }
            self._persist_path.write_text(json.dumps(data, indent=2))
            logger.debug("SelfCalibrator: persisted threshold=%.3f", self._current_threshold)
        except OSError as exc:
            logger.warning("SelfCalibrator: could not persist threshold: %s", exc)

    def _load_persisted(self) -> None:
        """Load a previously persisted threshold from disk."""
        if self._persist_path is None or not self._persist_path.exists():
            return
        try:
            data = json.loads(self._persist_path.read_text())
            loaded = float(data.get("threshold", self._current_threshold))
            loaded = max(0.30, min(0.95, loaded))
            self._current_threshold = loaded
            self._calibration_count = int(data.get("calibration_count", 0))
            logger.debug(
                "SelfCalibrator: loaded persisted threshold=%.3f (recalibrations=%d)",
                self._current_threshold,
                self._calibration_count,
            )
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("SelfCalibrator: could not load persisted threshold: %s", exc)
