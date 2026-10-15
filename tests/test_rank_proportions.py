import numpy as np
import pytest

from src.behavior_analysis.rank_proportions import (
    mean_by_rank,
    n_blocks,
    prop_units,
    proportions_by_block,
    proportions_by_mouse,
    row_props,
    rows_in_block,
    se_by_rank,
    standard_error,
    subject_rank_proportions,
    subject_rank_ses,
)
from tests.rank_rows import make_row


@pytest.fixture
def counts():
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


def test_standard_error_of_two_values():
    assert standard_error([1.0, 3.0]) == pytest.approx(np.sqrt(2) / np.sqrt(2))


def test_standard_error_of_fewer_than_two_values_is_nan_with_a_warning():
    with pytest.warns(RuntimeWarning):
        assert np.isnan(standard_error([0.5]))
    with pytest.warns(RuntimeWarning):
        assert np.isnan(standard_error([]))


def test_standard_error_counts_only_non_nan_units():
    assert standard_error([1.0, 3.0, np.nan]) == pytest.approx(np.sqrt(2) / np.sqrt(2))


def test_standard_error_of_one_non_nan_unit_among_nans_is_nan_with_a_warning():
    with pytest.warns(RuntimeWarning):
        assert np.isnan(standard_error([0.5, np.nan]))


UNITS = [
    {"best": 0.2, "second": 0.5, "third": 0.3},
    {"best": 0.6, "second": 0.3, "third": 0.1},
]


def test_mean_by_rank():
    assert mean_by_rank(UNITS, guard_empty=True) == pytest.approx([0.4, 0.4, 0.2])
    assert mean_by_rank(UNITS, guard_empty=False) == pytest.approx([0.4, 0.4, 0.2])


def test_mean_by_rank_of_no_units_is_nan_and_warns_only_unguarded(recwarn):
    assert np.isnan(mean_by_rank([], guard_empty=True)).all()
    assert len(recwarn) == 0
    with pytest.warns(RuntimeWarning):
        assert np.isnan(mean_by_rank([], guard_empty=False)).all()


def test_se_by_rank():
    assert se_by_rank(UNITS, guard_single=True) == pytest.approx([0.2, 0.1, 0.1])
    assert se_by_rank(UNITS, guard_single=False) == pytest.approx([0.2, 0.1, 0.1])


@pytest.mark.parametrize("units", [[], UNITS[:1]])
def test_se_by_rank_of_fewer_than_two_units_is_nan_and_warns_only_unguarded(units, recwarn):
    assert np.isnan(se_by_rank(units, guard_single=True)).all()
    assert len(recwarn) == 0
    with pytest.warns(RuntimeWarning):
        assert np.isnan(se_by_rank(units, guard_single=False)).all()


def test_se_by_rank_counts_only_non_nan_units():
    units = [*UNITS, {"best": np.nan, "second": 0.4, "third": 0.2}]
    ses = se_by_rank(units, guard_single=True)
    assert ses[0] == pytest.approx(0.2)  # two values: sd 0.283 / sqrt(2), not / sqrt(3)
    assert ses[1] == pytest.approx(np.std([0.5, 0.3, 0.4], ddof=1) / np.sqrt(3))


def test_se_by_rank_guard_counts_non_nan_units(recwarn):
    units = [UNITS[0], {"best": np.nan, "second": 0.3, "third": 0.1}]
    ses = se_by_rank(units, guard_single=True)
    assert np.isnan(ses[0]) and not np.isnan(ses[1])  # one non-NaN unit for "best"
    assert len(recwarn) == 0
    with pytest.warns(RuntimeWarning):
        se_by_rank(units, guard_single=False)


def test_row_props_defaults_missing_to_nan_unless_strict():
    assert np.isnan(row_props({"best_prop": 0.5})[1])
    with pytest.raises(KeyError):
        row_props({"best_prop": 0.5}, strict=True)


def test_prop_units_follow_rows():
    units = prop_units([make_row(1, 1, 2)])
    assert units == [{"best": 0.25, "second": 0.25, "third": 0.5}]


def test_proportions_by_mouse_skips_subjects_without_rows(counts):
    counts["m4"] = []
    per_mouse = proportions_by_mouse(counts)
    assert list(per_mouse) == ["m1", "m2", "m3"]
    assert per_mouse["m3"]["best"] == 1.0


def test_rows_in_block_takes_every_subject_that_has_the_block(counts):
    assert rows_in_block(counts, 0) == [counts["m1"][0], counts["m2"][0], counts["m3"][0]]
    assert rows_in_block(counts, 2) == [counts["m1"][2], counts["m2"][2]]
    assert rows_in_block(counts, 3) == [counts["m2"][3]]
    assert rows_in_block(counts, 4) == []


def test_proportions_by_block_averages_subjects_per_block_including_the_last_row(counts):
    per_block = proportions_by_block(counts)
    assert len(per_block) == 4
    assert per_block[0]["best"] == pytest.approx((0.6 + 0.4 + 1.0) / 3)
    # m2's empty block is NaN and is skipped by the mean
    assert per_block[1]["best"] == pytest.approx(0.2)
    assert per_block[2]["best"] == pytest.approx((1 / 3 + 1.0) / 2)
    assert per_block[3]["best"] == pytest.approx(1 / 3)


def test_proportions_by_block_of_empty_counts_is_empty():
    assert proportions_by_block({}) == []


def test_proportions_by_block_is_empty_when_every_subject_is_empty():
    assert proportions_by_block({"a": [], "b": []}) == []


def test_n_blocks_is_the_longest_subject_list(counts):
    assert n_blocks(counts) == 4
    assert n_blocks({"a": [], "b": [make_row(1, 0, 0)]}) == 1
    assert n_blocks({"a": []}) == 0
    assert n_blocks({}) == 0


def test_subject_rank_ses_unpooled_is_the_se_across_block_proportions():
    rows = [make_row(8, 2, 0), make_row(0, 5, 5), make_row(5, 5, 0)]
    columns = ([0.8, 0.0, 0.5], [0.2, 0.5, 0.5], [0.0, 0.5, 0.0])
    expected = [np.std(col, ddof=1) / np.sqrt(3) for col in columns]
    assert subject_rank_ses(rows) == pytest.approx(expected)


def test_subject_rank_ses_skips_nan_blocks_in_n():
    rows = [make_row(8, 2, 0), make_row(0, 5, 5), make_row(0, 0, 0)]
    expected = np.std([0.8, 0.0], ddof=1) / np.sqrt(2)
    assert subject_rank_ses(rows)[0] == pytest.approx(expected)


def test_subject_rank_ses_unpooled_of_a_single_block_is_nan():
    assert np.isnan(subject_rank_ses([make_row(1, 1, 1)])).all()


def test_subject_rank_ses_pooled_is_the_se_across_trials():
    # the pooled mean averages over trials, so its SE is across those same trials
    rows = [make_row(9, 1, 0), make_row(0, 0, 30)]
    ses = subject_rank_ses(rows, pool_trials_within_subject=True)
    n = 40
    for rank_count, se in zip((9, 1, 30), ses, strict=True):
        indicators = np.r_[np.ones(rank_count), np.zeros(n - rank_count)]
        assert se == pytest.approx(np.std(indicators, ddof=1) / np.sqrt(n))


def test_subject_rank_ses_pooled_with_fewer_than_two_trials_is_nan():
    assert np.isnan(subject_rank_ses([make_row(1, 0, 0)], pool_trials_within_subject=True)).all()
    assert np.isnan(subject_rank_ses([make_row(0, 0, 0)], pool_trials_within_subject=True)).all()
