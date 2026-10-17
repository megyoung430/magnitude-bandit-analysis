"""Tests for the rsync pulse-train aligner."""

import numpy as np
import pytest

from src.sync.rsync import Rsync_aligner, RsyncAligner, RsyncError, RsyncWarning
from tests.sync_fixtures import kept_pulse_indices, simulate_pulses

SEED = 7


def _train(**kwargs):
    return simulate_pulses(np.random.default_rng(SEED), **kwargs)


def _unrelated_train(n_pulse):
    return simulate_pulses(np.random.default_rng(SEED + 1), n_pulse=n_pulse)[1]


def test_alias_names_the_same_class():
    assert Rsync_aligner is RsyncAligner


@pytest.mark.parametrize("units", ["auto", "explicit"])
def test_recovers_every_pulse_under_drift_and_unit_scaling(units):
    # B stamps two ticks per millisecond and its clock drifts by a random walk.
    pulses_A, pulses_B = _train(n_pulse=300, units_B=2.0)
    kwargs = {} if units == "auto" else {"units_A": 1, "units_B": 0.5}
    aligner = RsyncAligner(pulses_A, pulses_B, **kwargs)

    # Pulse i on A pairs with pulse i on B; only the final partial chunk is left unpaired.
    paired = ~np.isnan(aligner.cor_times_B)
    assert paired.sum() >= 290
    np.testing.assert_array_equal(aligner.cor_times_B[paired], pulses_B[paired])
    np.testing.assert_array_equal(aligner.A_to_B(pulses_A[paired]), pulses_B[paired])
    np.testing.assert_array_equal(aligner.B_to_A(pulses_B[paired]), pulses_A[paired])


def test_auto_units_estimate_the_scale_between_systems():
    pulses_A, pulses_B = _train(n_pulse=200, units_B=2.0)
    aligner = RsyncAligner(pulses_A, pulses_B)
    assert aligner.units_A == 1
    assert aligner.units_B == pytest.approx(0.5, rel=1e-3)


def test_conversion_interpolates_between_pulses():
    pulses_A, pulses_B = _train(n_pulse=200)
    aligner = RsyncAligner(pulses_A, pulses_B)
    midpoints_A = (pulses_A[10:20] + pulses_A[11:21]) / 2
    midpoints_B = (pulses_B[10:20] + pulses_B[11:21]) / 2
    np.testing.assert_allclose(aligner.A_to_B(midpoints_A), midpoints_B)


@pytest.mark.parametrize("pulses_lost_by", ["both", "swapped"])
def test_pulses_missing_on_either_side_pair_only_true_partners(pulses_lost_by):
    n_pulse = 400
    pulses_A, pulses_B = _train(n_pulse=n_pulse, missing_pulses=True)
    keep_A, keep_B = kept_pulse_indices(n_pulse)
    if pulses_lost_by == "swapped":
        # The same trains with the roles of the two systems exchanged.
        pulses_A, pulses_B, keep_A, keep_B = pulses_B, pulses_A, keep_B, keep_A
    aligner = RsyncAligner(pulses_A, pulses_B)

    # Ground truth: a pulse's partner is the pulse with the same original number, if kept.
    shared = np.intersect1d(keep_A, keep_B)
    expected = np.full(len(pulses_A), np.nan)
    expected[np.isin(keep_A, shared)] = pulses_B[np.isin(keep_B, shared)]
    paired = ~np.isnan(aligner.cor_times_B)
    np.testing.assert_array_equal(aligner.cor_times_B[paired], expected[paired])
    assert paired.sum() >= 0.8 * len(shared)


def test_extrapolation_continues_the_clock_beyond_the_matched_pulses():
    pulses_A, pulses_B = _train(n_pulse=100)
    aligner = RsyncAligner(pulses_A, pulses_B)
    times_A = np.array([pulses_A[0] - 5000.0, pulses_A[-1] + 5000.0])
    assert np.isnan(aligner.A_to_B(times_A)).all()
    extrapolated = aligner.A_to_B(times_A, extrapolate=True)
    # B ticks twice per A millisecond, so 5000 A ms beyond either end is ~10000 B ticks.
    assert extrapolated[0] == pytest.approx(pulses_B[0] - 10_000, rel=1e-2)
    assert extrapolated[1] == pytest.approx(pulses_B[-1] + 10_000, rel=1e-2)
    np.testing.assert_allclose(aligner.B_to_A(extrapolated, extrapolate=True), times_A)


def test_non_matching_trains_raise_rsync_error():
    pulses_A, _ = _train(n_pulse=200)
    with pytest.raises(RsyncError):
        RsyncAligner(pulses_A, _unrelated_train(200))


def test_non_matching_trains_warn_when_exceptions_are_off():
    pulses_A, _ = _train(n_pulse=200)
    with pytest.warns(RsyncWarning, match="No match"):
        RsyncAligner(pulses_A, _unrelated_train(200), raise_exception=False)


def _swapped_blocks_train():
    """A pair of trains whose matched A times stay ordered but whose matched B times do not.

    A's third block of intervals comes first in B, separated from the other two by unrelated
    filler. Along B, the matched A times then jump across a NaN gap (invisible to a check
    that skips NaN differences), while along A the matched B times step backwards where the
    third block follows the second.
    """
    rng = np.random.default_rng(SEED)
    block_1, block_2, block_3, filler = (rng.integers(100, 1900, size=20) for _ in range(4))
    intervals_A = np.concatenate([block_1, block_2, block_3]).astype(float)
    intervals_B = np.concatenate([block_3, filler, block_1, block_2]).astype(float)
    intervals_B += rng.normal(scale=1.0, size=intervals_B.shape)  # nonzero mse for matches
    return np.r_[0.0, np.cumsum(intervals_A)], np.r_[0.0, np.cumsum(intervals_B)]


def test_order_check_catches_matches_out_of_order_in_b():
    pulses_A, pulses_B = _swapped_blocks_train()
    with pytest.warns(RsyncWarning):
        aligner = RsyncAligner(pulses_A, pulses_B, units_A=1, units_B=1, raise_exception=False)
    assert np.nanmin(np.diff(aligner.cor_times_A)) > 0  # the A-side check alone would pass
    assert np.nanmin(np.diff(aligner.cor_times_B)) < 0
    with pytest.raises(RsyncError):
        RsyncAligner(pulses_A, pulses_B, units_A=1, units_B=1)


def test_interval_discrepancy_is_small_for_a_true_match():
    pulses_A, pulses_B = _train(n_pulse=200)
    discrepancy = RsyncAligner(pulses_A, pulses_B, units_A=1, units_B=0.5).interval_discrepancy_ms()
    assert len(discrepancy) > 150
    assert np.abs(discrepancy).max() < 20  # clock noise is a few ms per pulse


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pulse_times_A": np.zeros((4, 4))}, "1-D"),
        ({"pulse_times_B": np.arange(60.0)[::-1]}, "increasing"),
        ({"pulse_times_A": np.arange(6.0)}, "at least"),
        ({"pulse_times_B": np.r_[0.0, 1.0, np.nan, np.arange(3.0, 60.0)]}, "finite"),
        ({"chunk_size": 0}, "chunk_size"),
    ],
)
def test_invalid_inputs_raise_value_error(kwargs, message):
    pulses_A, pulses_B = _train(n_pulse=50)
    args = {"pulse_times_A": pulses_A, "pulse_times_B": pulses_B, **kwargs}
    with pytest.raises(ValueError, match=message):
        RsyncAligner(**args)
