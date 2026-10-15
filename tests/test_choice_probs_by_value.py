"""accumulate_choice_probs on synthetic sessions: true and believed binning, pooling, averaging."""

import copy

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis.get_choice_probs_by_value import (
    BEST_VALS,
    SECOND_VALS,
    accumulate_choice_probs,
    init_matrix,
)


def session(magnitudes, ranks, choices=None):
    """Session whose trial_variables hold parallel per-trial lists."""
    tv = {"reward_magnitudes": magnitudes, "chosen_rank": ranks}
    tv["choice"] = choices if choices is not None else ["A"] * len(ranks)
    return {"trial_variables": [tv]}


def one_session(sess, subject="m1", problem=1):
    return {problem: {subject: {"ses-1": sess}}}


def assert_only_cells(df, expected):
    """Every cell not named in *expected* is NaN; named cells equal their value."""
    full = init_matrix(fill=np.nan)
    for (b, s), value in expected.items():
        full.loc[b, s] = value
    pd.testing.assert_frame_equal(df, full)


def test_init_matrix_has_labelled_value_grid():
    m = init_matrix(fill=0.0)
    assert list(m.index) == BEST_VALS
    assert list(m.columns) == SECOND_VALS
    assert m.index.name == "best_value" and m.columns.name == "second_value"
    assert (m.to_numpy() == 0.0).all()


def test_true_binning_counts_best_second_and_third_choices_per_cell():
    mags = {"A": 4, "B": 3, "C": 0}
    sess = session([mags] * 4, ["best", "second", "third", "best"])
    best, second, per_best, per_second = accumulate_choice_probs(one_session(sess))
    assert_only_cells(per_best["m1"], {(4, 3): 0.5})
    assert_only_cells(per_second["m1"], {(4, 3): 0.25})
    assert_only_cells(best, {(4, 3): 0.5})
    assert_only_cells(second, {(4, 3): 0.25})


def test_trials_are_binned_into_cells_by_their_best_and_second_magnitudes():
    sess = session(
        [{"A": 2, "B": 1, "C": 0}, {"A": 3, "B": 2, "C": 0}, {"A": 3, "B": 2, "C": 0}],
        ["best", "best", "second"],
    )
    _, _, per_best, per_second = accumulate_choice_probs(one_session(sess))
    assert_only_cells(per_best["m1"], {(2, 1): 1.0, (3, 2): 0.5})
    assert_only_cells(per_second["m1"], {(2, 1): 0.0, (3, 2): 0.5})


@pytest.mark.parametrize(
    "mags, rank",
    [
        ({"A": 4, "B": 3, "C": 0}, "weird"),  # unrecognised rank
        ({"A": 4}, "best"),  # fewer than two arms
        ({"A": 1, "B": 1, "C": 0}, "best"),  # best value 1 is outside BEST_VALS
        ({"A": 4, "B": 4, "C": 0}, "best"),  # second value 4 is outside SECOND_VALS
        ({"A": 4, "B": 0, "C": 0}, "best"),  # second value 0 is outside SECOND_VALS
    ],
)
def test_true_binning_skips_trials_outside_the_grid(mags, rank):
    _, _, per_best, per_second = accumulate_choice_probs(one_session(session([mags], [rank])))
    assert_only_cells(per_best["m1"], {})
    assert_only_cells(per_second["m1"], {})


def test_choice_labels_are_ignored_when_not_using_believed_values():
    mags = {"A": 4, "B": 3, "C": 0}
    sess = session([mags] * 2, ["best", "second"], choices=[None, "zzz"])
    _, _, per_best, _ = accumulate_choice_probs(one_session(sess))
    assert_only_cells(per_best["m1"], {(4, 3): 0.5})


@pytest.mark.parametrize("use_believed", [False, True])
@pytest.mark.parametrize(
    "n_mags, n_ranks, n_choices",
    [(5, 3, 4), (3, 3, 4), (3, 0, 3), (3, 3, 0)],
)
def test_mismatched_trial_list_lengths_raise(n_mags, n_ranks, n_choices, use_believed):
    mags = {"A": 4, "B": 3, "C": 0}
    sess = session([mags] * n_mags, ["best"] * n_ranks, choices=["A"] * n_choices)
    with pytest.raises(ValueError, match="reward_magnitudes.*chosen_rank.*choice"):
        accumulate_choice_probs(one_session(sess), use_believed)


@pytest.mark.parametrize(
    "sess",
    [
        {},
        {"trial_variables": []},
        {"trial_variables": [{}]},
        session([], []),
    ],
)
def test_sessions_without_usable_trials_still_register_the_mouse(sess):
    best, second, per_best, per_second = accumulate_choice_probs(one_session(sess))
    assert list(per_best) == list(per_second) == ["m1"]
    assert_only_cells(per_best["m1"], {})
    assert_only_cells(best, {})
    assert_only_cells(second, {})


def test_empty_input_gives_nan_matrices_and_no_mice():
    best, second, per_best, per_second = accumulate_choice_probs({})
    assert_only_cells(best, {})
    assert_only_cells(second, {})
    assert per_best == {} and per_second == {}


def test_counts_pool_across_problems_and_sessions_for_the_same_mouse():
    mags = {"A": 4, "B": 3, "C": 0}
    data = {
        1: {"m1": {"ses-1": session([mags], ["best"]), "ses-2": session([mags], ["third"])}},
        2: {"m1": {"ses-1": session([mags], ["second"])}, "m2": {}},
    }
    _, _, per_best, per_second = accumulate_choice_probs(data)
    assert list(per_best) == ["m1", "m2"]
    assert_only_cells(per_best["m1"], {(4, 3): 1 / 3})
    assert_only_cells(per_second["m1"], {(4, 3): 1 / 3})
    assert_only_cells(per_best["m2"], {})


def test_overall_is_the_nanmean_over_mice_so_each_mouse_weighs_equally():
    mags = {"A": 4, "B": 3, "C": 0}
    other = {"A": 2, "B": 1, "C": 0}
    data = {
        1: {
            "m1": {"ses-1": session([mags], ["best"])},  # 1 trial: P(best)=1
            "m2": {"ses-1": session([mags] * 3 + [other], ["third", "third", "third", "best"])},
        }
    }
    best, _, per_best, _ = accumulate_choice_probs(data)
    assert_only_cells(per_best["m1"], {(4, 3): 1.0})
    assert_only_cells(per_best["m2"], {(4, 3): 0.0, (2, 1): 1.0})
    # (4,3): mean(1.0, 0.0) not 1/4; (2,1): only m2 has data
    assert_only_cells(best, {(4, 3): 0.5, (2, 1): 1.0})


def test_all_recordings_of_a_multi_file_session_are_concatenated_in_order():
    mags = {"A": 4, "B": 3, "C": 0}
    first = session([mags], ["best"])["trial_variables"][0]
    later = session([mags] * 3, ["third"] * 3)["trial_variables"][0]
    both = accumulate_choice_probs(one_session({"trial_variables": [first, later]}))
    # 1 best + 3 third trials: P(best) = 1/4 (the first recording alone would give 1.0)
    assert_only_cells(both[2]["m1"], {(4, 3): 0.25})
    # the pooled result equals one recording holding all four trials
    merged = session([mags] * 4, ["best", "third", "third", "third"])
    expected = accumulate_choice_probs(one_session(merged))
    pd.testing.assert_frame_equal(both[2]["m1"], expected[2]["m1"])


def test_believed_binning_runs_across_recording_boundaries_in_recording_order():
    mags = {"A": 4, "B": 3, "C": 1}
    first = session([mags] * 2, ["best"] * 2, ["A", "B"])["trial_variables"][0]
    later = session([mags] * 2, ["best"] * 2, ["A", "B"])["trial_variables"][0]
    both = accumulate_choice_probs(
        one_session({"trial_variables": [first, later]}), use_believed=True
    )
    merged = session([mags] * 4, ["best"] * 4, ["A", "B", "A", "B"])
    expected = accumulate_choice_probs(one_session(merged), use_believed=True)
    pd.testing.assert_frame_equal(both[2]["m1"], expected[2]["m1"])
    pd.testing.assert_frame_equal(both[3]["m1"], expected[3]["m1"])


def test_input_is_not_mutated():
    mags = {"A": 4, "B": 3, "C": 1}
    data = one_session(session([mags] * 4, ["best", "second", "third", "best"], list("ABCA")))
    before = copy.deepcopy(data)
    accumulate_choice_probs(data)
    accumulate_choice_probs(data, use_believed=True)
    assert data == before


# ---- believed binning ------------------------------------------------------


def test_believed_binning_waits_for_two_seen_arms_and_a_seen_chosen_arm():
    mags = {"A": 4, "B": 3, "C": 1}
    #            n_seen 0  1 (A)   2 but C unseen   3: A best  3: B second  3: C third
    choices = ["A", "B", "C", "A", "B", "C"]
    sess = session([mags] * 6, ["best"] * 6, choices)
    _, _, per_best, per_second = accumulate_choice_probs(one_session(sess), use_believed=True)
    assert_only_cells(per_best["m1"], {(4, 3): 1 / 3})
    assert_only_cells(per_second["m1"], {(4, 3): 1 / 3})


def test_believed_binning_uses_last_observed_magnitudes_not_current_ones():
    first = {"A": 4, "B": 3, "C": 1}
    changed = {"A": 0, "B": 0, "C": 0}  # true values change; beliefs keep the last observed ones
    sess = session([first] * 4 + [changed], ["best"] * 5, ["A", "B", "C", "A", "B"])
    _, _, per_best, per_second = accumulate_choice_probs(one_session(sess), use_believed=True)
    # Trial 3: believes A=4, B=3, C=1 and chooses A (best). Trial 4: still believes the same
    # although the true values are now all 0, and chooses B (second).
    assert_only_cells(per_best["m1"], {(4, 3): 0.5})
    assert_only_cells(per_second["m1"], {(4, 3): 0.5})


def test_tied_believed_values_split_best_and_second_evenly_and_third_counts_in_denominator():
    mags = {"A": 3, "B": 3, "C": 1}
    # Trials 0-2 reveal A, B, C; trial 3 and 4 choose a tied arm; trial 5 the lower arm.
    choices = ["A", "B", "C", "A", "B", "C"]
    sess = session([mags] * 6, ["best"] * 6, choices)
    _, _, per_best, per_second = accumulate_choice_probs(one_session(sess), use_believed=True)
    # Trial 2 has the chosen arm C unseen (skipped); trials 3-5 are in cell (3, 3).
    assert_only_cells(per_best["m1"], {(3, 3): 1 / 3})
    assert_only_cells(per_second["m1"], {(3, 3): 1 / 3})


def test_a_choice_of_one_tied_arm_gives_half_to_each_numerator():
    mags = {"A": 3, "B": 3, "C": 1}
    # Without the split, the fixed A-before-B ranking would call this choice a plain "best".
    sess = session([mags] * 4, ["best"] * 4, ["A", "B", "C", "A"])
    _, _, per_best, per_second = accumulate_choice_probs(one_session(sess), use_believed=True)
    assert_only_cells(per_best["m1"], {(3, 3): 0.5})
    assert_only_cells(per_second["m1"], {(3, 3): 0.5})


def test_believed_binning_ignores_the_chosen_rank_values():
    mags = {"A": 4, "B": 3, "C": 1}
    sess = session([mags] * 5, ["third"] * 5, ["A", "B", "C", "A", "B"])
    _, _, best, second = accumulate_choice_probs(one_session(sess), use_believed=True)
    assert_only_cells(best["m1"], {(4, 3): 0.5})
    assert_only_cells(second["m1"], {(4, 3): 0.5})


def test_believed_binning_skips_cells_outside_the_value_grid():
    mags = {"A": 1, "B": 0, "C": 0}  # best believed value 1 is outside BEST_VALS
    sess = session([mags] * 5, ["best"] * 5, ["A", "B", "C", "A", "B"])
    _, _, per_best, _ = accumulate_choice_probs(one_session(sess), use_believed=True)
    assert_only_cells(per_best["m1"], {})


def test_use_believed_defaults_to_false_and_is_accepted_positionally():
    mags = {"A": 4, "B": 3, "C": 1}
    data = one_session(session([mags] * 4, ["best", "second", "third", "best"], list("ABCA")))
    default = accumulate_choice_probs(data)
    explicit = accumulate_choice_probs(data, False)
    believed = accumulate_choice_probs(data, True)
    pd.testing.assert_frame_equal(default[0], explicit[0])
    assert not default[0].equals(believed[0])


def test_believed_binning_depends_on_the_recording_order():
    high = {"A": 4, "B": 3, "C": 1}
    low = {"A": 2, "B": 1, "C": 0}
    rec_high = session([high] * 3, ["best"] * 3, ["A", "B", "C"])["trial_variables"][0]
    rec_low = session([low] * 2, ["best"] * 2, ["A", "B"])["trial_variables"][0]
    rec_low3 = session([low] * 3, ["best"] * 3, ["A", "B", "C"])["trial_variables"][0]
    rec_high2 = session([high] * 2, ["best"] * 2, ["A", "B"])["trial_variables"][0]
    results = []
    for first, second, mags in [
        (rec_high, rec_low, [high] * 3 + [low] * 2),
        (rec_low3, rec_high2, [low] * 3 + [high] * 2),
    ]:
        merged = accumulate_choice_probs(
            one_session({"trial_variables": [first, second]}), use_believed=True
        )
        single = accumulate_choice_probs(
            one_session(session(mags, ["best"] * 5, ["A", "B", "C", "A", "B"])),
            use_believed=True,
        )
        pd.testing.assert_frame_equal(merged[2]["m1"], single[2]["m1"])
        pd.testing.assert_frame_equal(merged[3]["m1"], single[3]["m1"])
        results.append(merged)
    # beliefs carry across the boundary, so the two orders fill different cells
    assert not results[0][2]["m1"].equals(results[1][2]["m1"])


def test_lengths_are_checked_per_recording_before_merging():
    mags = {"A": 4, "B": 3, "C": 0}
    # 3 magnitudes / 2 ranks plus 2 / 3 would merge to equal totals of 5 and 5
    first = {"reward_magnitudes": [mags] * 3, "chosen_rank": ["best"] * 2, "choice": ["A"] * 3}
    second = {"reward_magnitudes": [mags] * 2, "chosen_rank": ["best"] * 3, "choice": ["A"] * 2}
    with pytest.raises(ValueError, match="reward_magnitudes=3, chosen_rank=2"):
        accumulate_choice_probs(one_session({"trial_variables": [first, second]}))
