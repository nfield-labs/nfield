"""
FormatShield threshold oracle.

:class:`ThresholdOracle` decides whether each inference request should use
the Think-Then-Format (TTF) strategy or direct structured generation.

Decision flow
-------------
1. If the target model is a *native thinker* (has built-in CoT / extended
   thinking), always use direct generation – TTF would double-think.
2. If a ``latency_budget_ms`` is supplied and the estimated TTF overhead
   would exceed it, force direct generation.
3. If a trained scikit-learn :class:`~sklearn.linear_model.LogisticRegression`
   model is available on disk (``oracle_data/threshold_oracle_v1.pkl``), use it
   for prediction and derive confidence from ``predict_proba``.
4. Otherwise fall back to a simple weighted-score threshold lookup keyed on
   the inference backend.

The oracle can be retrained from benchmark CSV data via the
:meth:`ThresholdOracle.from_benchmark_data` classmethod.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from formatshield.oracle.routing_decision import RoutingDecision
from formatshield.scorer.features import ComplexityFeatures

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Models that have native chain-of-thought / extended thinking built in.
#: For these models, TTF is counter-productive – we always use direct mode.
NATIVE_THINKERS: frozenset[str] = frozenset(
    {
        "o1",
        "o3",
        "o1-mini",
        "o3-mini",
        "deepseek-r1",
        "deepseek-r1-distill-llama-70b",
        "deepseek-r1-distill-qwen-32b",
    }
)

#: Per-backend complexity score thresholds (v0 heuristics, overridden after
#: a benchmark training run).  Values are in [0, 1].
BACKEND_THRESHOLDS: dict[str, float] = {
    "vllm": 0.60,  # KV cache reuse → lower threshold
    "ollama": 0.65,
    "groq": 0.65,
    "openrouter": 0.67,
    "outlines": 0.62,
    "guidance": 0.63,
    "default": 0.65,
}

#: Estimated percentage latency overhead introduced by TTF per backend.
BACKEND_TTF_OVERHEAD: dict[str, float] = {
    "vllm": 10.0,
    "ollama": 25.0,
    "groq": 30.0,
    "openrouter": 35.0,
    "outlines": 20.0,
    "guidance": 22.0,
    "default": 30.0,
}

#: Expected accuracy improvement when TTF is used (positive means TTF helps).
_TTF_ACCURACY_DELTA: float = 0.17  # midpoint of 0.15–0.20 range
_DIRECT_ACCURACY_DELTA: float = 0.0

#: Heuristic confidence when no sklearn model is available.
_HEURISTIC_CONFIDENCE: float = 0.70

#: Default path for the persisted oracle model (relative to package root).
_DEFAULT_MODEL_PATH: Path = Path(__file__).parent / "oracle_data" / "threshold_oracle_v1.pkl"

# ---------------------------------------------------------------------------
# Feature weights used for the heuristic weighted-score fallback
# ---------------------------------------------------------------------------

_FEATURE_WEIGHTS: list[float] = [0.20, 0.25, 0.20, 0.15, 0.10, 0.10]
_FEATURE_CAPS: list[float] = [1.0, 10.0, 20.0, 1.0, 3.0, 30.0]


class ThresholdOracle:
    """Route each inference request to either TTF or direct generation.

    Parameters
    ----------
    model_path:
        Path to a pre-trained :class:`~sklearn.linear_model.LogisticRegression`
        pickle.  If ``None``, the default location
        ``oracle_data/threshold_oracle_v1.pkl`` (relative to this module) is
        tried.  If the file is missing the oracle falls back to heuristics.

    Example::

        oracle = ThresholdOracle()
        features = scorer.score(prompt, schema=schema, model_id=model_id)
        decision = oracle.predict(features, backend="vllm", model_id=model_id)
        if decision.use_ttf:
            ...
    """

    def __init__(self, model_path: Path | str | None = None) -> None:
        self._clf: Any = None  # sklearn LogisticRegression or None
        self._model_path: Path = Path(model_path) if model_path is not None else _DEFAULT_MODEL_PATH
        self._try_load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        features: ComplexityFeatures,
        backend: str,
        model_id: str,
        latency_budget_ms: float | None = None,
        cost_aware: bool = False,
    ) -> RoutingDecision:
        """Return a :class:`RoutingDecision` for the given request.

        Parameters
        ----------
        features:
            Feature vector computed by :class:`~formatshield.scorer.ComplexityScorer`.
        backend:
            Inference backend identifier (e.g. ``"vllm"``, ``"groq"``).
        model_id:
            Model identifier string (e.g. ``"gpt-4o"``).
        latency_budget_ms:
            Optional hard latency cap in milliseconds.  When set, TTF is
            suppressed if the estimated overhead would exceed the budget.
        cost_aware:
            When ``True`` the oracle applies a small upward bias to the
            direct-routing threshold (not yet fully implemented – reserved for
            token-cost optimisation).

        Returns
        -------
        RoutingDecision
            The routing decision.  Falls back to
            ``RoutingDecision(strategy="direct", confidence=0.3)`` on any error.
        """
        try:
            return self._predict_impl(
                features=features,
                backend=backend,
                model_id=model_id,
                latency_budget_ms=latency_budget_ms,
                cost_aware=cost_aware,
            )
        except Exception:
            logger.warning(
                "ThresholdOracle.predict: unexpected error – defaulting to direct",
                exc_info=True,
            )
            return RoutingDecision(
                strategy="direct",
                expected_accuracy_delta=_DIRECT_ACCURACY_DELTA,
                expected_overhead_pct=0.0,
                confidence=0.30,
                explanation="Oracle error - defaulting to direct",
            )

    @classmethod
    def from_benchmark_data(
        cls,
        csv_path: str | Path,
        model_path: Path | str | None = None,
        *,
        save: bool = True,
    ) -> ThresholdOracle:
        """Train a :class:`~sklearn.linear_model.LogisticRegression` from benchmark CSV.

        The CSV must be producible from
        :meth:`~formatshield.scorer.BenchmarkResult.to_dict` rows.  The target
        label is ``1`` (use TTF) when ``accuracy_delta > 0``, and ``0``
        (use direct) otherwise.

        Parameters
        ----------
        csv_path:
            Path to the benchmark results CSV file.
        model_path:
            Where to save the trained model.  Defaults to the same location as
            the instance default.
        save:
            If ``True`` (default) the trained model is persisted to *model_path*.

        Returns
        -------
        ThresholdOracle
            A new oracle instance backed by the freshly trained sklearn model.

        Raises
        ------
        ImportError
            If ``scikit-learn`` or ``joblib`` are not installed.
        FileNotFoundError
            If *csv_path* does not exist.
        ValueError
            If the CSV contains insufficient data to train a model.
        """
        try:
            import joblib  # type: ignore[import]
            import numpy as np  # type: ignore[import]
            from sklearn.linear_model import LogisticRegression  # type: ignore[import]
            from sklearn.preprocessing import StandardScaler  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "scikit-learn, joblib, and numpy are required to train the oracle. "
                "Install them with: pip install scikit-learn joblib numpy"
            ) from exc

        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Benchmark CSV not found: {csv_path}")

        rows = _load_benchmark_csv(csv_path)
        if len(rows) < 10:
            raise ValueError(
                f"Need at least 10 benchmark rows to train the oracle; got {len(rows)}."
            )

        feature_rows: list[list[float]] = []
        y: list[int] = []
        for row in rows:
            try:
                features = _features_from_benchmark_row(row)
                label = 1 if float(row["accuracy_delta"]) > 0.0 else 0
                feature_rows.append(features.to_feature_vector())
                y.append(label)
            except (KeyError, ValueError):
                logger.debug("Skipping malformed benchmark row: %s", row)
                continue

        if len(feature_rows) < 10:
            raise ValueError(
                f"Too few valid rows after filtering ({len(feature_rows)}); need at least 10."
            )

        x_arr = np.array(feature_rows, dtype=np.float64)
        y_arr = np.array(y, dtype=np.int32)

        # Scale features before logistic regression
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_arr)

        clf = LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced")
        clf.fit(x_scaled, y_arr)

        # Bundle scaler + clf so we can apply the same transform at predict time
        bundle = {"clf": clf, "scaler": scaler}

        out_path = Path(model_path) if model_path is not None else _DEFAULT_MODEL_PATH
        if save:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(bundle, out_path)
            logger.info("ThresholdOracle: trained model saved to %s", out_path)

        oracle = cls(model_path=out_path)
        # Inject the trained bundle directly (avoids a round-trip through disk)
        oracle._clf = bundle
        return oracle

    def save(self, path: Path | str) -> None:
        """Persist the current sklearn model bundle to *path*.

        Parameters
        ----------
        path:
            Destination file path.  Parent directories are created if needed.

        Raises
        ------
        RuntimeError
            If no trained model is currently loaded.
        ImportError
            If ``joblib`` is not installed.
        """
        if self._clf is None:
            raise RuntimeError("No trained model to save.  Train first with from_benchmark_data().")
        try:
            import joblib  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("joblib is required to save the model.") from exc

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._clf, path)
        logger.info("ThresholdOracle: model saved to %s", path)

    def load(self, path: Path | str) -> None:
        """Load a persisted sklearn model bundle from *path*.

        Parameters
        ----------
        path:
            Source file path.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ImportError
            If ``joblib`` is not installed.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        try:
            import joblib  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("joblib is required to load the model.") from exc

        self._clf = joblib.load(path)
        logger.info("ThresholdOracle: model loaded from %s", path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_load_model(self) -> None:
        """Attempt to load a pre-trained model from :attr:`_model_path`."""
        if not self._model_path.exists():
            logger.debug(
                "ThresholdOracle: no pre-trained model at %s – using heuristics.",
                self._model_path,
            )
            return
        try:
            import joblib  # type: ignore[import]

            self._clf = joblib.load(self._model_path)
            logger.info(
                "ThresholdOracle: loaded pre-trained model from %s",
                self._model_path,
            )
        except Exception:
            logger.warning(
                "ThresholdOracle: failed to load model from %s – using heuristics.",
                self._model_path,
                exc_info=True,
            )
            self._clf = None

    def _predict_impl(
        self,
        features: ComplexityFeatures,
        backend: str,
        model_id: str,
        latency_budget_ms: float | None,
        cost_aware: bool,
    ) -> RoutingDecision:
        """Core prediction logic (called inside try/except in :meth:`predict`)."""

        backend_key = backend.lower() if backend else "default"
        threshold = BACKEND_THRESHOLDS.get(backend_key, BACKEND_THRESHOLDS["default"])
        estimated_overhead = BACKEND_TTF_OVERHEAD.get(backend_key, BACKEND_TTF_OVERHEAD["default"])

        # ------------------------------------------------------------------
        # Rule 1: Native thinker models – always direct
        # ------------------------------------------------------------------
        if _is_native_thinker(model_id):
            return RoutingDecision(
                strategy="direct",
                expected_accuracy_delta=_DIRECT_ACCURACY_DELTA,
                expected_overhead_pct=0.0,
                confidence=0.95,
                explanation="Native thinker model detected – TTF would double-think.",
            )

        # ------------------------------------------------------------------
        # Rule 2: Latency budget exceeded – force direct
        # ------------------------------------------------------------------
        if latency_budget_ms is not None and latency_budget_ms > 0:
            if estimated_overhead > latency_budget_ms:
                return RoutingDecision(
                    strategy="direct",
                    expected_accuracy_delta=_DIRECT_ACCURACY_DELTA,
                    expected_overhead_pct=0.0,
                    confidence=0.85,
                    explanation=(
                        f"Estimated TTF overhead ({estimated_overhead:.0f} ms) "
                        f"exceeds latency budget ({latency_budget_ms:.0f} ms)."
                    ),
                )

        # ------------------------------------------------------------------
        # Rule 3: sklearn model available
        # ------------------------------------------------------------------
        if self._clf is not None:
            return self._predict_sklearn(
                features=features,
                backend_key=backend_key,
                threshold=threshold,
                estimated_overhead=estimated_overhead,
                cost_aware=cost_aware,
            )

        # ------------------------------------------------------------------
        # Rule 4: Heuristic weighted-score fallback
        # ------------------------------------------------------------------
        return self._predict_heuristic(
            features=features,
            backend_key=backend_key,
            threshold=threshold,
            estimated_overhead=estimated_overhead,
            cost_aware=cost_aware,
        )

    def _predict_sklearn(
        self,
        features: ComplexityFeatures,
        backend_key: str,
        threshold: float,
        estimated_overhead: float,
        cost_aware: bool,
    ) -> RoutingDecision:
        """Predict using the loaded sklearn model bundle."""
        try:
            import numpy as np  # type: ignore[import]

            vec = features.to_feature_vector()
            clf_bundle = self._clf

            # Handle both raw LogisticRegression and bundled dict
            if isinstance(clf_bundle, dict):
                clf = clf_bundle["clf"]
                scaler = clf_bundle.get("scaler")
            else:
                clf = clf_bundle
                scaler = None

            x_vec = np.array(vec, dtype=np.float64).reshape(1, -1)
            if scaler is not None:
                x_vec = scaler.transform(x_vec)

            pred = int(clf.predict(x_vec)[0])
            proba = clf.predict_proba(x_vec)[0]
            # proba shape: [p_direct, p_ttf]; class 1 = TTF
            confidence = float(proba[pred])

            if pred == 1:
                explanation = (
                    f"Sklearn oracle predicts TTF (confidence {confidence:.2f}) "
                    f"for backend '{backend_key}' (threshold {threshold:.2f})."
                )
                return RoutingDecision(
                    strategy="ttf",
                    expected_accuracy_delta=_TTF_ACCURACY_DELTA,
                    expected_overhead_pct=estimated_overhead,
                    confidence=confidence,
                    explanation=explanation,
                )
            else:
                explanation = (
                    f"Sklearn oracle predicts direct (confidence {confidence:.2f}) "
                    f"for backend '{backend_key}'."
                )
                return RoutingDecision(
                    strategy="direct",
                    expected_accuracy_delta=_DIRECT_ACCURACY_DELTA,
                    expected_overhead_pct=0.0,
                    confidence=confidence,
                    explanation=explanation,
                )
        except Exception:
            logger.warning(
                "ThresholdOracle: sklearn prediction failed – falling back to heuristics.",
                exc_info=True,
            )
            return self._predict_heuristic(
                features=features,
                backend_key=backend_key,
                threshold=threshold,
                estimated_overhead=estimated_overhead,
                cost_aware=cost_aware,
            )

    def _predict_heuristic(
        self,
        features: ComplexityFeatures,
        backend_key: str,
        threshold: float,
        estimated_overhead: float,
        cost_aware: bool,
    ) -> RoutingDecision:
        """Heuristic fallback: weighted feature score vs. backend threshold."""
        vec = features.to_feature_vector()

        # Normalise each feature by its cap, then apply weights
        weighted_score = 0.0
        for raw_val, cap, weight in zip(vec, _FEATURE_CAPS, _FEATURE_WEIGHTS, strict=True):
            normalised = min(1.0, max(0.0, raw_val / cap if cap > 0 else 0.0))
            weighted_score += normalised * weight
        weighted_score = min(1.0, max(0.0, weighted_score))

        # cost_aware: apply a small upward bias to the threshold
        effective_threshold = threshold + (0.03 if cost_aware else 0.0)

        if weighted_score > effective_threshold:
            explanation = (
                f"Heuristic score {weighted_score:.3f} > threshold {effective_threshold:.2f} "
                f"for backend '{backend_key}' → TTF."
            )
            return RoutingDecision(
                strategy="ttf",
                expected_accuracy_delta=_TTF_ACCURACY_DELTA,
                expected_overhead_pct=estimated_overhead,
                confidence=_HEURISTIC_CONFIDENCE,
                explanation=explanation,
            )
        else:
            explanation = (
                f"Heuristic score {weighted_score:.3f} ≤ threshold {effective_threshold:.2f} "
                f"for backend '{backend_key}' → direct."
            )
            return RoutingDecision(
                strategy="direct",
                expected_accuracy_delta=_DIRECT_ACCURACY_DELTA,
                expected_overhead_pct=0.0,
                confidence=_HEURISTIC_CONFIDENCE,
                explanation=explanation,
            )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _is_native_thinker(model_id: str) -> bool:
    """Return ``True`` if *model_id* (case-insensitive prefix match) is a
    known native-thinker model or its direct distillation."""
    lower = model_id.lower().strip()
    # Exact membership
    if lower in NATIVE_THINKERS:
        return True
    # Prefix match (e.g. "deepseek-r1-distill-..." variants)
    for native in NATIVE_THINKERS:
        if lower.startswith(native):
            return True
    return False


def _load_benchmark_csv(path: Path) -> list[dict]:  # type: ignore[type-arg]
    """Load a BenchmarkResult CSV as a list of row dicts."""
    rows: list[dict] = []  # type: ignore[type-arg]
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(dict(row))
    return rows


def _features_from_benchmark_row(row: dict) -> ComplexityFeatures:  # type: ignore[type-arg]
    """Reconstruct a :class:`ComplexityFeatures` from a ``BenchmarkResult.to_dict()`` row.

    The benchmark CSV does not store individual features, only the composite
    ``complexity_score``.  We reconstruct a simplified feature vector that
    re-uses the scalar score as ``token_entropy`` and leaves the other
    features at neutral values, so the logistic regression can still learn
    a meaningful decision boundary.

    When full per-feature CSV columns are available (advanced export), they
    take priority.
    """

    def _float(key: str, default: float = 0.0) -> float:
        try:
            return float(row.get(key, default) or default)
        except (ValueError, TypeError):
            return default

    def _int(key: str, default: int = 0) -> int:
        try:
            return int(float(row.get(key, default) or default))
        except (ValueError, TypeError):
            return default

    complexity = _float("complexity_score", 0.5)

    return ComplexityFeatures(
        token_entropy=_float("token_entropy", complexity),
        schema_depth=_int("schema_depth", 1),
        required_reasoning_ops=_int("required_reasoning_ops", 0),
        instruction_tune_score=_float("instruction_tune_score", 0.5),
        prompt_length_bucket=_int("prompt_length_bucket", 1),
        schema_constraint_count=_int("schema_constraint_count", 1),
    )
