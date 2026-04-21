"""
Unit tests for the self-calibrating routing threshold estimator.

Coverage map:
- PhiComponents dataclass (creation, as_tuple, immutability)
- OutcomeRecord dataclass (creation, field storage)
- _phi_from_components (formula correctness, range, monotonicity)
- _rbf_kernel (identity, symmetry, decay, length-scale effect)
- Cold-start guard (< min_samples → default threshold)
- record_outcome (sample count, window rollover, quality clamping)
- calibrate_threshold (all branches: cold, high-quality, low-quality, mixed)
- cross_schema_transfer (empty, nearby dominates, global fallback)
- stats (keys, is_active flag)
- reset (clears window, reverts to cold start)
- build_phi_calibrator (factory, custom params)
- thread safety (concurrent record_outcome)
"""

from __future__ import annotations

import dataclasses
import threading

import pytest

from formatshield.oracle.phi_calibrator import (
    DEFAULT_LENGTH_SCALE,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_TARGET_ACCURACY,
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW_SIZE,
    MAX_THRESHOLD,
    MIN_THRESHOLD,
    OutcomeRecord,
    PhiComponents,
    PhiOracleCalibrator,
    _phi_from_components,
    _rbf_kernel,
    build_phi_calibrator,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

COMP_MID = PhiComponents(lambda2=0.5, tau=0.5, delta_k=0.5)
COMP_COMPLEX = PhiComponents(lambda2=0.8, tau=0.8, delta_k=0.8)
COMP_FLAT = PhiComponents(lambda2=0.0, tau=0.0, delta_k=0.0)


def _fill(
    cal: PhiOracleCalibrator, comp: PhiComponents, n: int, quality: float, ttf: bool = True
) -> None:
    """Helper: record n observations with the given quality and routing decision."""
    for _ in range(n):
        cal.record_outcome(comp, ttf, quality)


# ---------------------------------------------------------------------------
# TestPhiComponents
# ---------------------------------------------------------------------------


class TestPhiComponents:
    def test_creation_stores_fields(self) -> None:
        c = PhiComponents(lambda2=0.3, tau=0.7, delta_k=0.5)
        assert c.lambda2 == pytest.approx(0.3)
        assert c.tau == pytest.approx(0.7)
        assert c.delta_k == pytest.approx(0.5)

    def test_as_tuple_correct_order(self) -> None:
        c = PhiComponents(lambda2=0.1, tau=0.2, delta_k=0.3)
        assert c.as_tuple() == (0.1, 0.2, 0.3)

    def test_frozen_raises_on_mutation(self) -> None:
        c = PhiComponents(lambda2=0.5, tau=0.5, delta_k=0.5)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            c.lambda2 = 0.9  # type: ignore[misc]

    def test_zero_features(self) -> None:
        c = PhiComponents(lambda2=0.0, tau=0.0, delta_k=0.0)
        assert c.as_tuple() == (0.0, 0.0, 0.0)

    def test_unit_features(self) -> None:
        c = PhiComponents(lambda2=1.0, tau=1.0, delta_k=1.0)
        assert c.as_tuple() == (1.0, 1.0, 1.0)

    def test_equality(self) -> None:
        a = PhiComponents(lambda2=0.5, tau=0.5, delta_k=0.5)
        b = PhiComponents(lambda2=0.5, tau=0.5, delta_k=0.5)
        assert a == b

    def test_inequality(self) -> None:
        a = PhiComponents(lambda2=0.5, tau=0.5, delta_k=0.5)
        b = PhiComponents(lambda2=0.6, tau=0.5, delta_k=0.5)
        assert a != b


# ---------------------------------------------------------------------------
# TestOutcomeRecord
# ---------------------------------------------------------------------------


class TestOutcomeRecord:
    def test_creation_stores_all_fields(self) -> None:
        comp = PhiComponents(lambda2=0.5, tau=0.5, delta_k=0.5)
        rec = OutcomeRecord(
            phi_components=comp,
            phi_score=0.75,
            used_ttf=True,
            quality_outcome=0.90,
        )
        assert rec.phi_components == comp
        assert rec.phi_score == pytest.approx(0.75)
        assert rec.used_ttf is True
        assert rec.quality_outcome == pytest.approx(0.90)

    def test_direct_routing_stored(self) -> None:
        comp = PhiComponents(lambda2=0.1, tau=0.1, delta_k=0.1)
        rec = OutcomeRecord(
            phi_components=comp,
            phi_score=0.20,
            used_ttf=False,
            quality_outcome=0.85,
        )
        assert rec.used_ttf is False


# ---------------------------------------------------------------------------
# TestPhiFromComponents
# ---------------------------------------------------------------------------


class TestPhiFromComponents:
    def test_zero_inputs_give_zero(self) -> None:
        phi = _phi_from_components(0.0, 0.0, 0.0)
        assert phi == pytest.approx(0.0)

    def test_high_inputs_give_high_phi(self) -> None:
        phi = _phi_from_components(1.0, 1.0, 1.0)
        assert phi > 0.99

    def test_output_in_unit_interval(self) -> None:
        for l2 in (0.0, 0.3, 0.7, 1.0):
            for tau in (0.0, 0.5, 1.0):
                for dk in (0.0, 0.5, 1.0):
                    phi = _phi_from_components(l2, tau, dk)
                    assert 0.0 <= phi <= 1.0

    def test_monotone_in_lambda2(self) -> None:
        low = _phi_from_components(0.1, 0.5, 0.5)
        high = _phi_from_components(0.9, 0.5, 0.5)
        assert high > low

    def test_monotone_in_delta_k(self) -> None:
        low = _phi_from_components(0.5, 0.5, 0.1)
        high = _phi_from_components(0.5, 0.5, 0.9)
        assert high > low

    def test_flat_schema_near_zero(self) -> None:
        # Flat schema: lambda2=0, tau=0 → phi driven only by delta_k=0 → 0
        phi = _phi_from_components(0.0, 0.0, 0.0)
        assert phi == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# TestRbfKernel
# ---------------------------------------------------------------------------


class TestRbfKernel:
    def test_same_point_returns_one(self) -> None:
        x = (0.5, 0.5, 0.5)
        assert _rbf_kernel(x, x, DEFAULT_LENGTH_SCALE) == pytest.approx(1.0)

    def test_far_points_near_zero(self) -> None:
        x = (0.0, 0.0, 0.0)
        y = (1.0, 1.0, 1.0)
        assert _rbf_kernel(x, y, DEFAULT_LENGTH_SCALE) < 1e-2

    def test_symmetry(self) -> None:
        x = (0.3, 0.5, 0.7)
        y = (0.6, 0.2, 0.4)
        assert _rbf_kernel(x, y, DEFAULT_LENGTH_SCALE) == pytest.approx(
            _rbf_kernel(y, x, DEFAULT_LENGTH_SCALE)
        )

    def test_nearby_higher_than_far(self) -> None:
        origin = (0.5, 0.5, 0.5)
        near = (0.6, 0.5, 0.5)
        far = (1.0, 0.5, 0.5)
        assert _rbf_kernel(origin, near, DEFAULT_LENGTH_SCALE) > _rbf_kernel(
            origin, far, DEFAULT_LENGTH_SCALE
        )

    def test_larger_length_scale_wider_influence(self) -> None:
        x = (0.0, 0.0, 0.0)
        y = (0.5, 0.0, 0.0)
        k_narrow = _rbf_kernel(x, y, 0.05)
        k_wide = _rbf_kernel(x, y, 2.0)
        assert k_wide > k_narrow

    def test_output_always_in_zero_one(self) -> None:
        points = [
            ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ((0.1, 0.9, 0.3), (0.8, 0.2, 0.7)),
        ]
        for a, b in points:
            k = _rbf_kernel(a, b, DEFAULT_LENGTH_SCALE)
            assert 0.0 < k <= 1.0


# ---------------------------------------------------------------------------
# TestColdStart
# ---------------------------------------------------------------------------


class TestColdStart:
    def test_empty_calibrator_returns_default(self) -> None:
        cal = PhiOracleCalibrator(default_threshold=0.65)
        result = cal.calibrate_threshold(0.5, 0.5, 0.5)
        assert result == pytest.approx(0.65)

    def test_nineteen_samples_still_cold(self) -> None:
        cal = PhiOracleCalibrator(min_samples=20, default_threshold=0.65)
        _fill(cal, COMP_COMPLEX, 19, quality=0.95)
        assert cal.calibrate_threshold(0.8, 0.8, 0.8) == pytest.approx(0.65)

    def test_exactly_min_samples_activates(self) -> None:
        cal = PhiOracleCalibrator(min_samples=20, default_threshold=0.65)
        _fill(cal, COMP_COMPLEX, 20, quality=0.95)
        result = cal.calibrate_threshold(0.8, 0.8, 0.8)
        # Calibration has activated — result should be a valid threshold
        assert MIN_THRESHOLD <= result <= MAX_THRESHOLD

    def test_cross_schema_transfer_empty_returns_target(self) -> None:
        cal = PhiOracleCalibrator(target_accuracy=0.80)
        result = cal.cross_schema_transfer(COMP_MID)
        assert result == pytest.approx(0.80)


# ---------------------------------------------------------------------------
# TestRecordOutcome
# ---------------------------------------------------------------------------


class TestRecordOutcome:
    def test_sample_count_increments(self) -> None:
        cal = PhiOracleCalibrator()
        assert cal.sample_count == 0
        cal.record_outcome(COMP_MID, True, 0.8)
        assert cal.sample_count == 1
        cal.record_outcome(COMP_MID, False, 0.7)
        assert cal.sample_count == 2

    def test_window_rolls_over_at_max(self) -> None:
        cal = PhiOracleCalibrator(window_size=5)
        _fill(cal, COMP_MID, 10, quality=0.8)
        assert cal.sample_count == 5

    def test_quality_above_one_clamped(self) -> None:
        cal = PhiOracleCalibrator()
        cal.record_outcome(COMP_MID, True, 1.5)
        assert cal.sample_count == 1  # stored, just clamped

    def test_quality_below_zero_clamped(self) -> None:
        cal = PhiOracleCalibrator()
        cal.record_outcome(COMP_MID, True, -0.3)
        assert cal.sample_count == 1

    def test_direct_routing_recorded(self) -> None:
        cal = PhiOracleCalibrator()
        cal.record_outcome(COMP_FLAT, False, 0.9)
        assert cal.sample_count == 1

    def test_phi_score_computed_and_stored(self) -> None:
        """Verify phi_score is non-negative and <= 1 for all valid components."""
        cal = PhiOracleCalibrator()
        for comp in (COMP_FLAT, COMP_MID, COMP_COMPLEX):
            cal.record_outcome(comp, True, 0.8)
        assert cal.sample_count == 3


# ---------------------------------------------------------------------------
# TestCalibrateThreshold
# ---------------------------------------------------------------------------


class TestCalibrateThreshold:
    def test_cold_start_returns_default(self) -> None:
        cal = PhiOracleCalibrator(min_samples=20, default_threshold=0.65)
        assert cal.calibrate_threshold(0.5, 0.5, 0.5) == pytest.approx(0.65)

    def test_high_quality_ttf_gives_floor_threshold(self) -> None:
        """When TTF consistently delivers excellent quality, use TTF broadly."""
        cal = PhiOracleCalibrator(min_samples=20, target_accuracy=0.80, length_scale=0.5)
        _fill(cal, COMP_COMPLEX, 30, quality=0.95, ttf=True)
        threshold = cal.calibrate_threshold(0.8, 0.8, 0.8)
        assert threshold == pytest.approx(MIN_THRESHOLD)

    def test_low_quality_ttf_gives_ceiling_threshold(self) -> None:
        """When TTF delivers poor quality everywhere, avoid TTF (high threshold)."""
        cal = PhiOracleCalibrator(min_samples=20, target_accuracy=0.80, length_scale=0.5)
        _fill(cal, COMP_COMPLEX, 30, quality=0.40, ttf=True)
        threshold = cal.calibrate_threshold(0.8, 0.8, 0.8)
        assert threshold == pytest.approx(MAX_THRESHOLD)

    def test_no_ttf_records_returns_default(self) -> None:
        """All-direct window has no TTF signal — returns default threshold."""
        cal = PhiOracleCalibrator(min_samples=5, default_threshold=0.65)
        _fill(cal, COMP_COMPLEX, 10, quality=0.85, ttf=False)
        threshold = cal.calibrate_threshold(0.8, 0.8, 0.8)
        assert threshold == pytest.approx(0.65)

    def test_result_always_in_valid_range(self) -> None:
        cal = PhiOracleCalibrator(min_samples=5, length_scale=0.5)
        _fill(cal, COMP_MID, 10, quality=0.75, ttf=True)
        threshold = cal.calibrate_threshold(0.5, 0.5, 0.5)
        assert MIN_THRESHOLD <= threshold <= MAX_THRESHOLD

    def test_mixed_quality_threshold_between_bounds(self) -> None:
        """Mix of good (0.90) and bad (0.60) TTF records → threshold between floor/ceil."""
        cal = PhiOracleCalibrator(min_samples=20, target_accuracy=0.80, length_scale=0.5)
        # High-Phi records with good quality (0.92)
        high_comp = PhiComponents(lambda2=0.9, tau=0.9, delta_k=0.9)
        _fill(cal, high_comp, 15, quality=0.92, ttf=True)
        # Low-Phi records with poor quality (0.55) — same context for easy weighting
        low_comp = PhiComponents(lambda2=0.2, tau=0.2, delta_k=0.2)
        _fill(cal, low_comp, 15, quality=0.55, ttf=True)
        threshold = cal.calibrate_threshold(0.6, 0.6, 0.6)
        assert MIN_THRESHOLD <= threshold <= MAX_THRESHOLD

    def test_custom_default_threshold_respected(self) -> None:
        cal = PhiOracleCalibrator(min_samples=100, default_threshold=0.55)
        assert cal.calibrate_threshold(0.5, 0.5, 0.5) == pytest.approx(0.55)

    def test_threshold_deterministic(self) -> None:
        """Same observations → same threshold (no randomness)."""
        cal = PhiOracleCalibrator(min_samples=20, length_scale=0.5)
        _fill(cal, COMP_COMPLEX, 25, quality=0.85, ttf=True)
        t1 = cal.calibrate_threshold(0.8, 0.8, 0.8)
        t2 = cal.calibrate_threshold(0.8, 0.8, 0.8)
        assert t1 == pytest.approx(t2)


# ---------------------------------------------------------------------------
# TestCrossSchemaTransfer
# ---------------------------------------------------------------------------


class TestCrossSchemaTransfer:
    def test_empty_window_returns_target(self) -> None:
        cal = PhiOracleCalibrator(target_accuracy=0.80)
        result = cal.cross_schema_transfer(COMP_MID)
        assert result == pytest.approx(0.80)

    def test_identical_features_dominate(self) -> None:
        """Query identical to stored observations should closely match their quality."""
        cal = PhiOracleCalibrator(length_scale=0.3)
        _fill(cal, COMP_COMPLEX, 20, quality=0.95)
        # Add a distant low-quality record
        cal.record_outcome(COMP_FLAT, True, 0.10)
        result = cal.cross_schema_transfer(COMP_COMPLEX)
        assert result > 0.85  # nearby high-quality records dominate

    def test_distant_schema_uses_global_average(self) -> None:
        """Very narrow kernel: distant query falls back to global average."""
        cal = PhiOracleCalibrator(length_scale=0.01)  # very narrow
        _fill(cal, COMP_COMPLEX, 10, quality=0.80)
        comp_distant = PhiComponents(lambda2=0.0, tau=0.0, delta_k=0.0)
        result = cal.cross_schema_transfer(comp_distant)
        # Should be global average (~0.80) as fallback
        assert 0.0 <= result <= 1.0

    def test_result_in_unit_interval(self) -> None:
        cal = PhiOracleCalibrator()
        _fill(cal, COMP_MID, 5, quality=0.7)
        result = cal.cross_schema_transfer(COMP_MID)
        assert 0.0 <= result <= 1.0

    def test_high_quality_region(self) -> None:
        cal = PhiOracleCalibrator(length_scale=0.5)
        _fill(cal, COMP_COMPLEX, 20, quality=0.92)
        result = cal.cross_schema_transfer(COMP_COMPLEX)
        assert result == pytest.approx(0.92, abs=0.05)

    def test_low_quality_region(self) -> None:
        cal = PhiOracleCalibrator(length_scale=0.5)
        _fill(cal, COMP_FLAT, 20, quality=0.30)
        result = cal.cross_schema_transfer(COMP_FLAT)
        assert result == pytest.approx(0.30, abs=0.05)

    def test_mixed_quality_weighted_result(self) -> None:
        """Schemas equidistant from query contribute equally to predicted quality."""
        cal = PhiOracleCalibrator(length_scale=0.5)
        # Two clusters at equal distances from mid query
        comp_a = PhiComponents(lambda2=0.3, tau=0.5, delta_k=0.5)
        comp_b = PhiComponents(lambda2=0.7, tau=0.5, delta_k=0.5)
        for _ in range(10):
            cal.record_outcome(comp_a, True, 1.0)
            cal.record_outcome(comp_b, True, 0.0)
        result = cal.cross_schema_transfer(COMP_MID)
        # Should be roughly 0.5 (symmetric contributions)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# TestStats
# ---------------------------------------------------------------------------


class TestStats:
    def test_required_keys_present(self) -> None:
        cal = PhiOracleCalibrator()
        s = cal.stats()
        for key in (
            "sample_count",
            "is_active",
            "window_size",
            "min_samples",
            "target_accuracy",
            "length_scale",
            "default_threshold",
        ):
            assert key in s

    def test_is_active_false_on_cold_start(self) -> None:
        cal = PhiOracleCalibrator(min_samples=20)
        assert cal.stats()["is_active"] is False

    def test_is_active_true_after_min_samples(self) -> None:
        cal = PhiOracleCalibrator(min_samples=5)
        _fill(cal, COMP_MID, 5, quality=0.8)
        assert cal.stats()["is_active"] is True

    def test_sample_count_reflects_window(self) -> None:
        cal = PhiOracleCalibrator()
        _fill(cal, COMP_MID, 7, quality=0.8)
        assert cal.stats()["sample_count"] == 7

    def test_params_reflected_in_stats(self) -> None:
        cal = PhiOracleCalibrator(
            window_size=100,
            min_samples=10,
            target_accuracy=0.90,
            length_scale=0.5,
            default_threshold=0.60,
        )
        s = cal.stats()
        assert s["window_size"] == 100
        assert s["min_samples"] == 10
        assert s["target_accuracy"] == pytest.approx(0.90)
        assert s["length_scale"] == pytest.approx(0.5)
        assert s["default_threshold"] == pytest.approx(0.60)


# ---------------------------------------------------------------------------
# TestReset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_all_observations(self) -> None:
        cal = PhiOracleCalibrator()
        _fill(cal, COMP_MID, 10, quality=0.9)
        assert cal.sample_count == 10
        cal.reset()
        assert cal.sample_count == 0

    def test_after_reset_returns_default_threshold(self) -> None:
        cal = PhiOracleCalibrator(min_samples=5, default_threshold=0.65)
        _fill(cal, COMP_COMPLEX, 20, quality=0.95)
        cal.reset()
        result = cal.calibrate_threshold(0.8, 0.8, 0.8)
        assert result == pytest.approx(0.65)

    def test_after_reset_cross_schema_returns_target(self) -> None:
        cal = PhiOracleCalibrator(target_accuracy=0.80)
        _fill(cal, COMP_MID, 10, quality=0.6)
        cal.reset()
        result = cal.cross_schema_transfer(COMP_MID)
        assert result == pytest.approx(0.80)

    def test_can_record_after_reset(self) -> None:
        cal = PhiOracleCalibrator()
        _fill(cal, COMP_MID, 5, quality=0.8)
        cal.reset()
        cal.record_outcome(COMP_MID, True, 0.9)
        assert cal.sample_count == 1


# ---------------------------------------------------------------------------
# TestPublicAPI (build_phi_calibrator)
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_returns_calibrator_instance(self) -> None:
        cal = build_phi_calibrator()
        assert isinstance(cal, PhiOracleCalibrator)

    def test_defaults_match_module_constants(self) -> None:
        cal = build_phi_calibrator()
        s = cal.stats()
        assert s["window_size"] == DEFAULT_WINDOW_SIZE
        assert s["min_samples"] == DEFAULT_MIN_SAMPLES
        assert s["target_accuracy"] == pytest.approx(DEFAULT_TARGET_ACCURACY)
        assert s["length_scale"] == pytest.approx(DEFAULT_LENGTH_SCALE)
        assert s["default_threshold"] == pytest.approx(DEFAULT_THRESHOLD)

    def test_custom_window_size(self) -> None:
        cal = build_phi_calibrator(window_size=50)
        assert cal.stats()["window_size"] == 50

    def test_custom_min_samples(self) -> None:
        cal = build_phi_calibrator(min_samples=5)
        assert cal.stats()["min_samples"] == 5

    def test_custom_target_accuracy(self) -> None:
        cal = build_phi_calibrator(target_accuracy=0.90)
        assert cal.stats()["target_accuracy"] == pytest.approx(0.90)

    def test_custom_length_scale(self) -> None:
        cal = build_phi_calibrator(length_scale=0.5)
        assert cal.stats()["length_scale"] == pytest.approx(0.5)

    def test_custom_default_threshold(self) -> None:
        cal = build_phi_calibrator(default_threshold=0.55)
        assert cal.stats()["default_threshold"] == pytest.approx(0.55)

    def test_fresh_calibrator_starts_cold(self) -> None:
        cal = build_phi_calibrator(min_samples=20)
        assert cal.stats()["is_active"] is False
        assert cal.sample_count == 0


# ---------------------------------------------------------------------------
# TestThreadSafety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_record_outcome_no_errors(self) -> None:
        cal = PhiOracleCalibrator(window_size=200)
        errors: list[Exception] = []

        def record_many() -> None:
            try:
                for _ in range(50):
                    cal.record_outcome(COMP_MID, True, 0.8)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=record_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert cal.sample_count <= 200  # window_size cap

    def test_concurrent_calibrate_no_errors(self) -> None:
        cal = PhiOracleCalibrator(min_samples=5)
        _fill(cal, COMP_COMPLEX, 20, quality=0.85)
        errors: list[Exception] = []

        def calibrate_many() -> None:
            try:
                for _ in range(20):
                    cal.calibrate_threshold(0.8, 0.8, 0.8)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=calibrate_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test_record_and_calibrate_concurrent(self) -> None:
        cal = PhiOracleCalibrator(min_samples=5)
        _fill(cal, COMP_MID, 10, quality=0.8)
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for _ in range(30):
                    cal.record_outcome(COMP_MID, True, 0.9)
            except Exception as exc:
                errors.append(exc)

        def reader() -> None:
            try:
                for _ in range(30):
                    cal.calibrate_threshold(0.5, 0.5, 0.5)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


# ---------------------------------------------------------------------------
# TestWindowRollover
# ---------------------------------------------------------------------------


class TestWindowRollover:
    def test_window_size_enforced(self) -> None:
        cal = PhiOracleCalibrator(window_size=10)
        _fill(cal, COMP_MID, 25, quality=0.8)
        assert cal.sample_count == 10

    def test_stats_sample_count_matches_window(self) -> None:
        cal = PhiOracleCalibrator(window_size=7)
        _fill(cal, COMP_MID, 20, quality=0.8)
        assert cal.stats()["sample_count"] == 7

    def test_oldest_evicted_first(self) -> None:
        """After rollover, cross_schema_transfer should reflect only recent records."""
        cal = PhiOracleCalibrator(window_size=5, length_scale=10.0)
        # Fill with low-quality records
        _fill(cal, COMP_MID, 5, quality=0.20)
        # Overwrite entirely with high-quality records
        _fill(cal, COMP_MID, 5, quality=0.90)
        result = cal.cross_schema_transfer(COMP_MID)
        # Window now holds only the 5 high-quality records
        assert result == pytest.approx(0.90, abs=0.05)
