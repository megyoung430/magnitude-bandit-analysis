import numpy as np
import pytest

from src.behavior_analysis.rank_proportions import (
    mean_and_se_by_rank,
    mouse_block_rows,
    n_plotted_blocks,
    proportions_by_block,
    proportions_by_mouse,
    prop_units,
    row_props,
    rows_in_block,
    standard_error,
    subject_rank_proportions,
)


def make_row(best, second, third):
    total = best + second + third
    return {
        "best": best,
        "second": second,
        "third": third,
        "total": total,
        "best_prop": best / total if total else np.nan,
        "second_prop": second / total if total else np.nan,
        "third_prop": third / total if total else np.nan,
    }


@pytest.fixture
def counts():
    # the last row of each subject is the entry the by-block views drop
    return {
        "m1": [make_row(6, 3, 1), make_row(2, 2, 6), make_row(1, 1, 1)],
        "m2": [make_row(4, 4, 2), make_row(0, 0, 0), make_row(9, 0, 0), make_row(1, 1, 1)],
        "m3": [make_row(1, 0, 0)],
    }


def test_subject_rank_proportions_averages_block_proportions():
    rows = [make_row(8, 2, 0), make_row(0, 5, 5)]
    props = subject_rank_proportions(rows)
    assert props == pytest.approx({"best": 0.4, "second": 0.35, "third": 0.25})


def test_subject_rank_proportions_pooled_weights_by_trial_count():
    rows = [make_row(9, 1, 0), make_row(0, 0, 30)]
    pooled = subject_rank_proportions(rows, pool_trials_within_subject=True)
    assert pooled == pytest.approx({"best": 0.225, "second": 0.025, "third": 0.75})
    unpooled = subject_rank_proportions(rows)
    assert unpooled == pytest.approx({"best": 0.45, "second": 0.05, "third": 0.5})


def test_subject_rank_proportions_pooled_without_trials_is_nan():
    props = subject_rank_proportions([make_row(0, 0, 0)], pool_trials_within_subject=True)
    assert all(np.isnan(v) for v in props.values())


def test_standard_error_needs_two_values():
    assert np.isnan(standard_error([0.5]))
    assert np.isnan(standard_error([]))
    assert standard_error([1.0, 3.0]) == pytest.approx(np.sqrt(2) / np.sqrt(2))


def test_standard_error_ignores_nan_in_spread_but_not_in_n():
    # n counts every unit, including the NaN one
    assert standard_error([1.0, 3.0, np.nan]) == pytest.approx(np.sqrt(2) / np.sqrt(3))


def test_mean_and_se_by_rank():
    units = [
        {"best": 0.2, "second": 0.5, "third": 0.3},
        {"best": 0.6, "second": 0.3, "third": 0.1},
    ]
    means, ses = mean_and_se_by_rank(units)
    assert means == pytest.approx([0.4, 0.4, 0.2])
    assert ses == pytest.approx([0.2, 0.1, 0.1])


def test_mean_and_se_by_rank_without_units_is_nan():
    means, ses = mean_and_se_by_rank([])
    assert np.isnan(means).all() and np.isnan(ses).all()


def test_row_props_defaults_missing_to_nan_unless_strict():
    assert np.isnan(row_props({"best_prop": 0.5})[1])
    with pytest.raises(KeyError):
        row_props({"best_prop": 0.5}, strict=True)


def test_prop_units_follow_rows():
    units = prop_units([make_row(1, 1, 2)])
    assert units == [{"best": 0.25, "second": 0.25, "third": 0.5}]


def test_mouse_block_rows_drops_last_row_unless_pooled():
    rows = [make_row(1, 0, 0), make_row(0, 1, 0), make_row(0, 0, 1)]
    assert mouse_block_rows(rows, pool_trials_within_subject=False) == rows[:-1]
    assert mouse_block_rows(rows, pool_trials_within_subject=True) == rows


def test_proportions_by_mouse_skips_subjects_without_rows(counts):
    counts["m4"] = []
    per_mouse = proportions_by_mouse(counts)
    assert list(per_mouse) == ["m1", "m2", "m3"]
    assert per_mouse["m3"]["best"] == 1.0


def test_rows_in_block_excludes_each_subjects_last_row(counts):
    assert rows_in_block(counts, 0) == [counts["m1"][0], counts["m2"][0]]
    assert rows_in_block(counts, 2) == [counts["m2"][2]]
    assert rows_in_block(counts, 3) == []


def test_proportions_by_block_averages_subjects_per_block(counts):
    per_block = proportions_by_block(counts)
    assert len(per_block) == 3
    assert per_block[0]["best"] == pytest.approx((0.6 + 0.4) / 2)
    # m2's empty block is NaN and is skipped by the mean
    assert per_block[1]["best"] == pytest.approx(0.2)
    assert per_block[2]["best"] == pytest.approx(1.0)


def test_proportions_by_block_of_empty_counts_is_empty():
    assert proportions_by_block({}) == []


def test_proportions_by_block_fails_when_every_subject_is_empty():
    # the block count is the max over non-empty subjects; with none it is undefined
    with pytest.raises(ValueError):
        proportions_by_block({"a": [], "b": []})


def test_n_plotted_blocks(counts):
    assert n_plotted_blocks(counts) == 3
    assert n_plotted_blocks({"a": [], "b": [make_row(1, 0, 0)]}) == 0
    assert n_plotted_blocks({}) == 0
