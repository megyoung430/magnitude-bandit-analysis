import numpy as np
import pytest

from src.behavior_analysis.get_choice_probs_around_good_reversals import (
    across_subject_curves,
    apply_moving_average_to_choice_probs,
    assert_probabilities_sum_to_one,
    first_bad_between,
    get_choice_probs_around_good_reversals,
    keep_zero_then_skip,
    mean_and_se,
    nan_moving_average,
    next_good_after,
    pad_to_length,
    remove_trials_after_bad_rev,
    reversal_choice_rows,
    smooth_pre_post,
    subject_choice_matrices,
    summarize_subject,
    trial_offsets,
    truncate_post,
    truncate_reversal_at_bad,
)


def make_reversal(idx, pre, post, prev_best="A", next_best="B"):
    """Reversal whose prev-best tower is ``prev_best`` and next-best is ``next_best``.

    ``pre``/``post`` are the chosen tower per trial; choices are one-hot.
    """
    towers = ["A", "B", "C"]
    before = {t: 1.0 for t in towers}
    after = {t: 1.0 for t in towers}
    before[prev_best] = 3.0
    after[next_best] = 3.0
    return {
        "reversal_idx": idx,
        "reward_magnitudes_by_tower_before": before,
        "reward_magnitudes_by_tower_after": after,
        "trial_window_idx": {
            "pre": list(range(idx - len(pre), idx)),
            "post": list(range(idx, idx + len(post))),
        },
        "choices_by_tower": {
            t: {"pre": [float(p == t) for p in pre], "post": [float(p == t) for p in post]}
            for t in towers
        },
        "choices_by_rank": {"best": {"pre": [1.0] * len(pre), "post": [1.0] * len(post)}},
    }


# ---------------------------------------------------------------- choice probabilities


def test_keep_zero_then_skip_keeps_index_zero():
    assert keep_zero_then_skip([0, 1, 2, 3, 4], 2) == [0, 2, 3, 4]
    assert keep_zero_then_skip([0, 1, 2], 0) == [0, 1, 2]
    assert keep_zero_then_skip([0], 3) == [0]


def test_pad_to_length_pads_with_nan_and_truncates():
    np.testing.assert_array_equal(pad_to_length([1, 0], 4), [1, 0, np.nan, np.nan])
    np.testing.assert_array_equal(pad_to_length([1, 0, 1], 2), [1, 0])


def test_reversal_choice_rows_assigns_tower_roles():
    rev = make_reversal(10, pre=["A", "A"], post=["B", "C", "B"], prev_best="A", next_best="B")
    rows = reversal_choice_rows(rev, skip=0, length=5)
    np.testing.assert_array_equal(rows["prev_best"], [1, 1, 0, 0, 0])
    np.testing.assert_array_equal(rows["next_best"], [0, 0, 1, 0, 1])
    np.testing.assert_array_equal(rows["third"], [0, 0, 0, 1, 0])


def test_reversal_choice_rows_skips_after_reversal_trial():
    rev = make_reversal(10, pre=["A"], post=["B", "C", "B", "A"], prev_best="A", next_best="B")
    rows = reversal_choice_rows(rev, skip=2, length=4)
    # post becomes [trial 0, trial 2, trial 3] = [B, B, A]
    np.testing.assert_array_equal(rows["next_best"], [0, 1, 1, 0])
    np.testing.assert_array_equal(rows["prev_best"], [1, 0, 0, 1])


def test_subject_choice_matrices_stacks_one_row_per_reversal():
    revs = [make_reversal(10, ["A"], ["B", "B"]), make_reversal(30, ["A"], ["C", "B"])]
    mats = subject_choice_matrices(revs, skip=0, length=3)
    assert set(mats) == {"prev_best", "next_best", "third"}
    assert mats["next_best"].shape == (2, 3)
    np.testing.assert_array_equal(mats["third"], [[0, 0, 0], [0, 1, 0]])


def test_summarize_subject_reports_means_and_counts():
    mats = {
        "prev_best": np.array([[1.0, 0.0], [0.0, 0.0]]),
        "next_best": np.array([[0.0, 1.0], [1.0, 0.0]]),
        "third": np.array([[0.0, 0.0], [0.0, 1.0]]),
    }
    summary = summarize_subject(mats)
    assert summary["num_reversals"] == 2
    np.testing.assert_allclose(summary["prev_best_mean"], [0.5, 0.0])
    np.testing.assert_allclose(summary["third_mean"], [0.0, 0.5])
    np.testing.assert_array_equal(summary["next_best"], mats["next_best"])


def test_mean_and_se_ignores_nan_and_uses_sample_sd():
    stack = np.array([[1.0, np.nan], [3.0, 2.0], [5.0, 4.0]])
    mean, se = mean_and_se(stack)
    np.testing.assert_allclose(mean, [3.0, 3.0])
    # sample SD / sqrt(number of rows), rows being counted even where a bin is NaN
    np.testing.assert_allclose(se, [2.0 / np.sqrt(3), np.sqrt(2) / np.sqrt(3)])


def test_mean_and_se_single_row_has_zero_se():
    mean, se = mean_and_se(np.array([[0.2, 0.8]]))
    np.testing.assert_array_equal(mean, [0.2, 0.8])
    np.testing.assert_array_equal(se, [0.0, 0.0])


def test_across_subject_curves_reads_the_requested_keys():
    per_subject = {
        "m1": {"k": np.array([1.0, 0.0]), "num_reversals": 4},
        "m2": {"k": np.array([0.0, 0.0])},  # no count: contributes 0 reversals
    }
    across = across_subject_curves(per_subject, {"prev_best": "k"})
    np.testing.assert_allclose(across["mean"]["prev_best"], [0.5, 0.0])
    np.testing.assert_allclose(across["se"]["prev_best"], [0.5, 0.0])
    assert (across["num_subjects"], across["num_reversals"]) == (2, 4)


def test_across_subject_curves_rejects_no_subjects():
    with pytest.raises(ValueError):
        across_subject_curves({}, {"prev_best": "k"})


def test_trial_offsets_skip_shortens_the_post_axis():
    np.testing.assert_array_equal(trial_offsets(2, 3, 0), [-2, -1, 0, 1, 2])
    np.testing.assert_array_equal(trial_offsets(2, 5, 3), [-2, -1, 0, 1])


@pytest.mark.parametrize("skip", [-1, 3, 4])
def test_trial_offsets_rejects_bad_skip(skip):
    with pytest.raises(ValueError, match="skip_n_trials_after_reversal"):
        trial_offsets(1, 3, skip)


def test_assert_probabilities_sum_to_one():
    ok = {
        "prev_best": np.array([0.5, np.nan]),
        "next_best": np.array([0.5, np.nan]),
        "third": np.array([0.0, np.nan]),
    }
    assert_probabilities_sum_to_one(ok, "Probabilities", allow_no_finite_bins=False)
    bad = {**ok, "third": np.array([0.1, np.nan])}
    with pytest.raises(AssertionError, match="Probabilities do not sum to 1"):
        assert_probabilities_sum_to_one(bad, "Probabilities", allow_no_finite_bins=False)


def test_assert_probabilities_sum_to_one_all_nan():
    nan = np.full(2, np.nan)
    all_nan = {"prev_best": nan, "next_best": nan, "third": nan}
    assert_probabilities_sum_to_one(all_nan, "Probabilities", allow_no_finite_bins=True)
    with pytest.raises(AssertionError, match="No finite bins"):
        assert_probabilities_sum_to_one(all_nan, "Probabilities", allow_no_finite_bins=False)


def test_get_choice_probs_end_to_end():
    windows = {
        "m1": [
            make_reversal(10, ["A", "A"], ["B", "B", "C"]),
            make_reversal(40, ["A", "A"], ["B", "C", "C"]),
        ],
        "m2": [make_reversal(15, ["A", "A"], ["B", "B", "B"])],
        "none": [],
    }
    x, per_subject, across = get_choice_probs_around_good_reversals(windows, pre=2, post=3)
    np.testing.assert_array_equal(x, [-2, -1, 0, 1, 2])
    assert set(per_subject) == {"m1", "m2"}
    assert across["num_subjects"] == 2
    assert across["num_reversals"] == 3
    # m1 next-best post = [1, 0.5, 0], m2 = [1, 1, 1]
    np.testing.assert_allclose(across["mean"]["next_best"], [0, 0, 1, 0.75, 0.5])
    np.testing.assert_allclose(across["mean"]["prev_best"], [1, 1, 0, 0, 0])
    total = sum(across["mean"][k] for k in ("prev_best", "next_best", "third"))
    np.testing.assert_allclose(total, 1.0)


def test_get_choice_probs_skip_shortens_the_axis():
    windows = {"m1": [make_reversal(10, ["A"] * 3, ["B"] * 6)]}
    x, per_subject, _ = get_choice_probs_around_good_reversals(
        windows, pre=3, post=6, skip_n_trials_after_reversal=4
    )
    np.testing.assert_array_equal(x, [-3, -2, -1, 0, 1])
    assert per_subject["m1"]["prev_best"].shape == (1, 5)


@pytest.mark.parametrize("skip", [-1, 3, 4])
def test_get_choice_probs_rejects_bad_skip(skip):
    windows = {"m1": [make_reversal(10, ["A"], ["B", "B", "B"])]}
    with pytest.raises(ValueError, match="skip_n_trials_after_reversal"):
        get_choice_probs_around_good_reversals(
            windows, pre=1, post=3, skip_n_trials_after_reversal=skip
        )


def test_get_choice_probs_asserts_probabilities_sum_to_one():
    rev = make_reversal(10, ["A"], ["B", "B"])
    rev["choices_by_tower"]["A"]["post"] = [1.0, 1.0]  # tower A also chosen: sums to 2
    with pytest.raises(AssertionError, match="do not sum to 1"):
        get_choice_probs_around_good_reversals({"m1": [rev]}, pre=1, post=2)


# ---------------------------------------------------------------- moving average


def test_nan_moving_average_centered_skips_nan():
    y = np.array([1.0, np.nan, 3.0, 5.0])
    out = nan_moving_average(y, 3, mode="centered")
    np.testing.assert_allclose(out, [1.0, 2.0, 4.0, 4.0])


def test_nan_moving_average_trailing_uses_past_only():
    out = nan_moving_average(np.array([1.0, 2.0, 3.0, 4.0]), 2, mode="trailing")
    np.testing.assert_allclose(out, [1.0, 1.5, 2.5, 3.5])


def test_nan_moving_average_all_nan_window_stays_nan():
    out = nan_moving_average(np.array([np.nan, np.nan, 1.0]), 2, mode="trailing")
    assert np.isnan(out[:2]).all() and out[2] == 1.0


@pytest.mark.parametrize("window", [None, 0, 1])
def test_nan_moving_average_small_window_is_identity(window):
    y = [1, np.nan, 2]
    np.testing.assert_array_equal(nan_moving_average(y, window), np.asarray(y, dtype=float))


def test_smooth_pre_post_does_not_bleed_across_the_split():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    pre_mask = np.array([True, True, False, False])
    out = smooth_pre_post(y, pre_mask, window=2, mode="trailing")
    np.testing.assert_allclose(out, [0.0, 0.0, 1.0, 1.0])


def test_smooth_pre_post_with_empty_pre_segment():
    out = smooth_pre_post(np.array([1.0, 3.0]), np.array([False, False]), window=2, mode="trailing")
    np.testing.assert_allclose(out, [1.0, 2.0])


def per_subject_curves(prev, nxt, third, n=2):
    return {
        "prev_best_mean": np.array(prev, dtype=float),
        "next_best_mean": np.array(nxt, dtype=float),
        "third_mean": np.array(third, dtype=float),
        "num_reversals": n,
    }


def test_apply_moving_average_smooths_and_summarises():
    x = np.array([-2, -1, 0, 1])
    per_subject = {
        "m1": per_subject_curves([1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0], n=2),
        "m2": per_subject_curves([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], n=3),
    }
    x_out, smoothed, across = apply_moving_average_to_choice_probs(
        x, per_subject, moving_avg_window=2, mode="trailing"
    )
    np.testing.assert_array_equal(x_out, x)
    np.testing.assert_allclose(smoothed["m2"]["next_best_mean_sm"], [0, 0, 0, 0.5])
    assert "next_best_mean_sm" not in per_subject["m1"]  # input untouched
    assert across["num_subjects"] == 2
    assert across["num_reversals"] == 5
    assert (across["moving_avg_window"], across["mode"], across["split_at"]) == (2, "trailing", 0)
    total = sum(across["mean"][k] for k in ("prev_best", "next_best", "third"))
    np.testing.assert_allclose(total, 1.0)


def test_apply_moving_average_without_usable_subjects():
    x = np.array([-1, 0])
    _, smoothed, across = apply_moving_average_to_choice_probs(x, {"m1": {"num_reversals": 1}})
    assert smoothed == {}
    assert across["num_subjects"] == 0
    assert across["mean"] == {"prev_best": None, "next_best": None, "third": None}
    assert across["se"] == {"prev_best": None, "next_best": None, "third": None}


def test_apply_moving_average_rejects_probabilities_off_one():
    x = np.array([-1, 0])
    per_subject = {"m1": per_subject_curves([0.9, 0.9], [0, 0], [0, 0])}
    with pytest.raises(AssertionError, match="Smoothed probabilities"):
        apply_moving_average_to_choice_probs(x, per_subject, moving_avg_window=1)


# ------------------------------- removing trials after bad reversals


def test_next_good_after_and_first_bad_between():
    assert next_good_after([5, 20, 40], 20) == 40
    assert next_good_after([5, 20], 20) == float("inf")
    assert first_bad_between([3, 25, 30], 20, 40) == 25
    assert first_bad_between([20, 40], 20, 40) is None  # both bounds are exclusive
    assert first_bad_between([], 20, float("inf")) is None


def test_truncate_post_clips_every_post_list_and_skips_non_dicts():
    rev = make_reversal(10, ["A"], ["B"] * 5)
    rev["choices_by_tower"]["odd"] = "not a dict"
    truncate_post(rev, 2)
    assert rev["choices_by_tower"]["A"]["post"] == [0.0, 0.0]
    assert len(rev["choices_by_tower"]["A"]["pre"]) == 1
    assert rev["choices_by_rank"]["best"]["post"] == [1.0, 1.0]
    assert rev["trial_window_idx"]["post"] == [10, 11]


def test_truncate_reversal_at_bad_includes_or_excludes_the_bad_trial():
    rev = make_reversal(10, ["A"], ["B"] * 8)
    kept = truncate_reversal_at_bad(rev, [10, 30], [14], include_bad_trial=True)
    assert (kept["removed_bad_idx"], kept["post_len_after_removal"]) == (14, 5)
    assert len(kept["choices_by_tower"]["A"]["post"]) == 5
    cut = truncate_reversal_at_bad(rev, [10, 30], [14], include_bad_trial=False)
    assert cut["post_len_after_removal"] == 4
    assert len(rev["choices_by_tower"]["A"]["post"]) == 8  # original untouched


def test_truncate_reversal_at_bad_ignores_bads_after_next_good():
    rev = make_reversal(10, ["A"], ["B"] * 8)
    out = truncate_reversal_at_bad(rev, [10, 12], [13], include_bad_trial=True)
    assert out["removed_bad_idx"] is None and out["post_len_after_removal"] is None
    assert len(out["choices_by_tower"]["A"]["post"]) == 8


def test_truncate_reversal_at_bad_without_reversal_idx():
    rev = make_reversal(10, ["A"], ["B"] * 3)
    del rev["reversal_idx"]
    out = truncate_reversal_at_bad(rev, [10], [11], include_bad_trial=True)
    assert out["removed_bad_idx"] is None


def test_remove_trials_after_bad_rev_per_subject(capsys):
    windows = {
        "m1": [make_reversal(30, ["A"], ["B"] * 6), make_reversal(10, ["A"], ["B"] * 6)],
        "empty": [],
    }
    out = remove_trials_after_bad_rev(
        windows, {"m1": [10, 30], "empty": []}, {"m1": [13]}, include_bad_trial=True
    )
    assert [r["reversal_idx"] for r in out["m1"]] == [10, 30]  # sorted by reversal_idx
    assert out["m1"][0]["post_len_after_removal"] == 4
    assert out["m1"][1]["post_len_after_removal"] is None
    assert out["empty"] == []
    assert "[REMOVE] m1 good@10: cut post at bad@13 (kept 4 post trials)" in capsys.readouterr().out
