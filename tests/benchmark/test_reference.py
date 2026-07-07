"""Integrity checks for the published external reference curves."""

from __future__ import annotations

from benchmark.reference import EXTRACTBENCH_WALL, IFSCALE_CURVES


def test_ifscale_curves_are_present_and_cited() -> None:
    assert IFSCALE_CURVES
    for curve in IFSCALE_CURVES:
        assert curve.label
        assert curve.source.startswith("arXiv:")
        assert curve.points


def test_ifscale_accuracies_are_fractions_in_range() -> None:
    for curve in IFSCALE_CURVES:
        for count, accuracy in curve.points:
            assert count > 0
            assert 0.0 <= accuracy <= 1.0


def test_ifscale_curves_share_the_same_instruction_grid() -> None:
    # Every model is reported at the same densities, so the reference lines align.
    grids = {tuple(n for n, _ in curve.points) for curve in IFSCALE_CURVES}
    assert grids == {(10, 100, 250, 500)}


def test_ifscale_curves_are_non_increasing() -> None:
    # Accuracy is reported as a decay with density; no curve should rise.
    for curve in IFSCALE_CURVES:
        accuracies = [a for _, a in curve.points]
        assert accuracies == sorted(accuracies, reverse=True)


def test_ifscale_best_at_max_density_matches_headline() -> None:
    # The paper's headline: the best frontier model reaches only 68% at 500.
    best_at_500 = max(curve.points[-1][1] for curve in IFSCALE_CURVES)
    assert 0.68 <= best_at_500 <= 0.69


def test_extractbench_wall_is_the_zero_point() -> None:
    assert EXTRACTBENCH_WALL.accuracy == 0.0
    assert EXTRACTBENCH_WALL.n_fields == 369
    assert EXTRACTBENCH_WALL.source.startswith("arXiv:")
