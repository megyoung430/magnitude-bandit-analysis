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


def test_trials_are_truncated_to_the_shortest_trial_list():
    mags = {"A": 4, "B": 3, "C": 0}
    sess = session([mags] * 5, ["best", "second", "second"], choices=["A"] * 4)
    _, _, per_best, per_second = accumulate_choice_probs(one_session(sess))
    assert_only_cells(per_best["m1"], {(4, 3): 1 / 3})
    assert_only_cells(per_second["m1"], {(4, 3): 2 / 3})


@pytest.mark.parametrize(
    "sess",
    [
        {},
        {"trial_variables": []},
        {"trial_variables": [{}]},
        session([], []),
        session([{"A": 4, "B": 3}], []),
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


def test_only_the_first_trial_variables_element_is_read():
    # Known issue pinned: trials from later recording files of a merged session are ignored.
    mags = {"A": 4, "B": 3, "C": 0}
    first = session([mags], ["best"])["trial_variables"][0]
    later = session([mags] * 3, ["third"] * 3)["trial_variables"][0]
    for use_believed in (False, True):
        both = accumulate_choice_probs(
            one_session({"trial_variables": [first, later]}), use_believed
        )
        only_first = accumulate_choice_probs(
            one_session({"trial_variables": [first]}), use_believed
        )
        pd.testing.assert_frame_equal(both[2]["m1"], only_first[2]["m1"])
        pd.testing.assert_frame_equal(both[0], only_first[0])
    # True binning: the three later "third" choices would dilute P(best) to 1/4.
    plain = accumulate_choice_probs(one_session({"trial_variables": [first, later]}))
    assert_only_cells(plain[2]["m1"], {(4, 3): 1.0})


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
    sess = session([first, first, first, changed], ["best"] * 4, ["A", "B", "A", "B"])
    _, _, per_best, per_second = accumulate_choice_probs(one_session(sess), use_believed=True)
    # Trial 2: believes A=4, B=3 and chooses A (best). Trial 3: still believes A=4, B=3
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


def test_believed_binning_ignores_the_chosen_rank_values_but_not_their_count():
    mags = {"A": 4, "B": 3, "C": 1}
    choices = ["A", "B", "A", "B"]
    full = session([mags] * 4, ["third"] * 4, choices)
    short = session([mags] * 4, ["third"] * 3, choices)
    _, _, best_full, second_full = accumulate_choice_probs(one_session(full), use_believed=True)
    _, _, best_short, second_short = accumulate_choice_probs(one_session(short), use_believed=True)
    assert_only_cells(best_full["m1"], {(4, 3): 0.5})
    assert_only_cells(second_full["m1"], {(4, 3): 0.5})
    # The fourth trial is dropped with the missing rank: only the third trial remains.
    assert_only_cells(best_short["m1"], {(4, 3): 1.0})
    assert_only_cells(second_short["m1"], {(4, 3): 0.0})


def test_believed_binning_skips_cells_outside_the_value_grid():
    mags = {"A": 1, "B": 0, "C": 0}  # best believed value 1 is outside BEST_VALS
    sess = session([mags] * 4, ["best"] * 4, ["A", "B", "A", "B"])
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
