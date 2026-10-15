import numpy as np
import pytest

from src.behavior_analysis.get_task_statistics import get_block_lengths
from tests.test_reversal_boundaries import session, snapshot


def mouse(good, bad=None):
    """One session whose cumulative good/bad reversal counters are given per trial."""
    bad = bad if bad is not None else (0,) * len(good)
    return {"ses-1": session(snapshot(), snapshot(), good=good, bad=bad)}


# good reversals start at trials 2 and 5, bad ones at trials 1 and 4
GOOD = (0, 0, 1, 1, 1, 2, 2, 2, 2)
BAD = (0, 1, 1, 1, 2, 2, 2, 2, 2)


def test_blocks_run_from_trial_zero_to_each_reversal():
    stats = get_block_lengths({"m": mouse(GOOD, BAD)}, boundary="good")
    assert stats["per_mouse_blocklens"] == {"m": {1: 2.0, 2: 3.0}}


def test_boundary_selects_good_bad_or_both():
    data = {"m": mouse(GOOD, BAD)}
    assert get_block_lengths(data, "bad")["per_mouse_blocklens"]["m"] == {1: 1.0, 2: 3.0}
    assert get_block_lengths(data, "all")["per_mouse_blocklens"]["m"] == {
        1: 1.0,
        2: 1.0,
        3: 2.0,
        4: 1.0,
    }


def test_default_boundary_is_all():
    stats = get_block_lengths({"m": mouse(GOOD, BAD)})
    assert stats["boundary"] == "all"
    assert len(stats["per_mouse_blocklens"]["m"]) == 4


def test_a_reversal_of_both_kinds_on_one_trial_is_one_boundary():
    data = {"m": mouse((0, 0, 1, 1), (0, 0, 1, 1))}
    assert get_block_lengths(data, "all")["per_mouse_blocklens"]["m"] == {1: 2.0}


def test_the_stretch_after_the_last_reversal_is_not_a_block():
    stats = get_block_lengths({"m": mouse((0, 0, 1) + (1,) * 20)}, "good")
    assert stats["per_mouse_blocklens"]["m"] == {1: 2.0}


def test_pools_blocks_across_mice_with_median_and_standard_error():
    data = {
        "a": mouse((0, 0, 1, 1, 2, 2)),  # blocks 2, 2
        "b": mouse((0, 0, 0, 0, 1, 1, 2, 2)),  # blocks 4, 2
        "c": mouse((0, 1, 1, 2, 2, 3)),  # blocks 1, 2, 2
    }
    stats = get_block_lengths(data, "good")
    assert stats["mice"] == ["a", "b", "c"]
    assert stats["blocks"] == [1, 2, 3]
    assert stats["block_to_vals"] == {1: [2.0, 4.0, 1.0], 2: [2.0, 2.0, 2.0], 3: [2.0]}
    assert stats["meds"].tolist() == [2.0, 2.0, 2.0]
    expected_se = np.std([2.0, 4.0, 1.0], ddof=1) / np.sqrt(3)
    assert stats["ses"].tolist() == pytest.approx(
        [expected_se, 0.0, np.nan], nan_ok=True
    )  # one value: NaN, never 0


def test_summary_arrays_are_float():
    stats = get_block_lengths({"m": mouse(GOOD, BAD)}, "good")
    assert stats["meds"].dtype == stats["ses"].dtype == np.float64


def test_mice_without_a_reversal_are_left_out():
    data = {"flat": mouse((0, 0, 0)), "m": mouse(GOOD, BAD)}
    stats = get_block_lengths(data, "good")
    assert stats["mice"] == ["m"]
    assert "flat" not in stats["per_mouse_blocklens"]


def test_no_mouse_with_a_boundary_raises():
    with pytest.raises(ValueError, match="No mice"):
        get_block_lengths({"flat": mouse((0, 0, 0))}, "good")
    with pytest.raises(ValueError, match="No mice"):
        get_block_lengths({"m": mouse(GOOD, (0,) * len(GOOD))}, "bad")


def test_unknown_boundary_raises():
    with pytest.raises(ValueError, match="boundary must be"):
        get_block_lengths({"m": mouse(GOOD, BAD)}, "sometimes")
