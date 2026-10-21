"""Circular-shift chance levels: the batched computation against a plain loop, and which trace
each branch shifts."""

import numpy as np
import pytest

from src.funcimg_analysis.shared_signal import (
    leave_one_out_mean,
    partial_row_correlation,
    row_correlation,
)
from src.funcimg_analysis.shared_signal_null import (
    null_partial_r,
    null_r,
    shift_offsets,
)
from tests.test_funcimg_shared_signal import N_FRAMES, make_cells, smooth_noise

OFFSETS = shift_offsets(N_FRAMES, 300, 7, np.random.default_rng(0))


def reference_null_r(cells, other, offsets):
    return np.stack([row_correlation(cells, np.roll(other, k, axis=-1)) for k in offsets])


def reference_null_partial(cells, average, controls, offsets, shifted):
    """One shift at a time through the plain (unbatched) partial correlation."""
    controls = controls if isinstance(controls, list) else [controls]
    out = []
    for k in offsets:
        if shifted == "average":
            out.append(partial_row_correlation(cells, np.roll(average, k, axis=-1), controls))
        else:
            shifted_controls = [np.roll(controls[0], k, axis=-1), *controls[1:]]
            out.append(partial_row_correlation(cells, average, shifted_controls))
    return np.stack(out)


@pytest.fixture
def scenario():
    rng = np.random.default_rng(1)
    cells, field = make_cells(rng, n_cells=9, common_weight=0.6)
    neuropil = np.array([field + 0.5 * smooth_noise(rng) for _ in range(9)])
    return cells, field, neuropil, leave_one_out_mean(cells), leave_one_out_mean(neuropil)


def test_null_r_with_a_shared_trace_equals_correlating_each_shifted_copy(scenario):
    cells, field, *_ = scenario
    np.testing.assert_allclose(
        null_r(cells, field, OFFSETS), reference_null_r(cells, field, OFFSETS), atol=1e-10
    )


def test_null_r_with_a_trace_per_cell_equals_correlating_each_shifted_copy(scenario):
    cells, _, _, loo, _ = scenario
    np.testing.assert_allclose(
        null_r(cells, loo, OFFSETS), reference_null_r(cells, loo, OFFSETS), atol=1e-10
    )


@pytest.mark.parametrize("shifted", ["average", "background"])
def test_null_partial_r_with_a_background_only_equals_the_plain_loop(scenario, shifted):
    cells, field, _, loo, _ = scenario
    np.testing.assert_allclose(
        null_partial_r(cells, loo, field, OFFSETS, shifted),
        reference_null_partial(cells, loo, field, OFFSETS, shifted),
        atol=1e-9,
    )


@pytest.mark.parametrize("shifted", ["average", "background"])
def test_null_partial_r_with_neuropil_controls_equals_the_plain_loop(scenario, shifted):
    cells, field, neuropil, loo, loo_neuropil = scenario
    controls = [field, neuropil, loo_neuropil]
    np.testing.assert_allclose(
        null_partial_r(cells, loo, controls, OFFSETS, shifted),
        reference_null_partial(cells, loo, controls, OFFSETS, shifted),
        atol=1e-9,
    )


def test_rows_with_missing_frames_are_computed_exactly_and_do_not_change_the_clean_rows(scenario):
    cells, field, neuropil, loo, loo_neuropil = scenario
    cells = cells.copy()
    cells[2, 400:460] = np.nan  # a row that needs the frame-by-frame mask
    loo = leave_one_out_mean(cells)
    controls = [field, neuropil, loo_neuropil]
    np.testing.assert_allclose(
        null_r(cells, field, OFFSETS), reference_null_r(cells, field, OFFSETS), atol=1e-9
    )
    np.testing.assert_allclose(
        null_partial_r(cells, loo, controls, OFFSETS, "average"),
        reference_null_partial(cells, loo, controls, OFFSETS, "average"),
        atol=1e-9,
    )


def test_a_background_with_missing_frames_still_gives_the_exact_answer(scenario):
    cells, field, _, loo, _ = scenario
    field = field.copy()
    field[1000:1040] = np.nan
    np.testing.assert_allclose(
        null_partial_r(cells, loo, field, OFFSETS, "background"),
        reference_null_partial(cells, loo, field, OFFSETS, "background"),
        atol=1e-9,
    )


def test_the_background_branch_shifts_the_background_and_nothing_else(scenario):
    cells, field, _, loo, _ = scenario
    null = null_partial_r(cells, loo, field, OFFSETS, "background")
    for row, k in zip(null, OFFSETS, strict=True):
        np.testing.assert_allclose(
            row, partial_row_correlation(cells, loo, np.roll(field, k)), atol=1e-9
        )
    assert not np.allclose(null, null_partial_r(cells, loo, field, OFFSETS, "average"))


def test_the_average_branch_shifts_the_other_cells_and_keeps_the_background(scenario):
    cells, field, _, loo, _ = scenario
    null = null_partial_r(cells, loo, field, OFFSETS, "average")
    for row, k in zip(null, OFFSETS, strict=True):
        np.testing.assert_allclose(
            row, partial_row_correlation(cells, np.roll(loo, k, axis=1), field), atol=1e-9
        )


def test_a_constant_row_gives_nan_in_the_null_not_an_error(scenario):
    cells, field, _, loo, _ = scenario
    cells = cells.copy()
    cells[0] = 5.0
    assert np.isnan(null_r(cells, field, OFFSETS)[:, 0]).all()


def test_null_partial_r_rejects_an_unknown_shifted_trace():
    with pytest.raises(ValueError, match="shifted"):
        null_partial_r(np.zeros((2, 50)), np.zeros((2, 50)), np.zeros(50), np.array([5]), "cells")


def test_the_batched_null_is_much_faster_than_the_loop_on_a_large_run():
    """Timing sanity check on 120 cells x 20000 frames: the batched path, not the loop, is used."""
    import time

    rng = np.random.default_rng(2)
    n, t = 120, 20000
    field = smooth_noise(rng, t)
    cells = np.array([field + smooth_noise(rng, t) for _ in range(n)])
    offsets = shift_offsets(t, 600, 60, np.random.default_rng(0))
    start = time.perf_counter()
    fast = null_r(cells, field, offsets)
    fast_s = time.perf_counter() - start
    start = time.perf_counter()
    slow = reference_null_r(cells, field, offsets)
    slow_s = time.perf_counter() - start
    np.testing.assert_allclose(fast, slow, atol=1e-9)
    assert fast_s < slow_s / 3
