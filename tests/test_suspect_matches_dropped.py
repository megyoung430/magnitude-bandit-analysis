"""The suspect counts and the drop warning of get_good_reversal_info use one definition.

- suspect: checked, and the best tower is unchanged or tied;
- uncheckable: a magnitude is missing, or there is no trial to check against (the reversal
  was logged after a session's last trial and the next session's start does not show it).
  A reversal logged between recordings is checked like any other.

Every reversal that is suspect or uncheckable is dropped by get_good_reversal_info, counted
per reversal (a jump of two on one trial is two), except those the zero-reward gate or the
missing-tower check skips first.
"""

import re
import warnings

import numpy as np
import pytest

from src.behavior_analysis.best_change import is_suspect, is_uncheckable
from src.behavior_analysis.cumulative_reversal_removal import total_removal_curves
from src.behavior_analysis.get_good_reversal_info import ReversalWarning, get_good_reversal_info
from src.behavior_analysis.get_total_reversals import (
    get_reversal_indices_by_kind,
    get_total_reversals,
)
from src.behavior_analysis.suspect_reversals import count_suspect_good_reversals

TOWERS = "ABC"
A_BEST, B_BEST, TIE_AB = (4, 1, 2), (1, 4, 2), (4, 4, 2)


def session(good, magnitudes, **extra):
    """A counter session; *magnitudes* is one ``(A, B, C)`` triple per trial."""
    n = len(magnitudes)
    return {
        "trial": list(range(1, n + 1)),
        "blocks": [1] * n,
        "has_good": True,
        "good_reversals": good,
        "reward_magnitudes_by_tower": {t: [m[i] for m in magnitudes] for i, t in enumerate(TOWERS)},
        "choices_by_tower": {t: [0] * n for t in TOWERS},
        "choices_by_rank": {r: [0] * n for r in ("best", "second", "third")},
        **extra,
    }


def mixed_subject():
    """Five within-run reversals: three valid, one tie, one best-unchanged, one missing.

    ses-1 changes the best tower at trials 2 and 4. ses-2 has a tie at trial 2 and a missing
    magnitude at trial 4. ses-3 has an unchanged best at trial 2.
    """
    return {
        "ses-1": session([0, 0, 1, 1, 2, 2], [A_BEST, A_BEST, B_BEST, B_BEST, A_BEST, A_BEST]),
        "ses-2": session(
            [0, 0, 1, 1, 2, 2], [A_BEST, A_BEST, TIE_AB, TIE_AB, (4, 1, None), (4, 1, 2)]
        ),
        "ses-3": session([0, 0, 1, 1], [A_BEST, A_BEST, A_BEST, A_BEST]),
    }


def dropped_count(data):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        get_good_reversal_info(data)
    texts = [str(w.message) for w in caught if issubclass(w.category, ReversalWarning)]
    counts = [int(m[1]) for t in texts if (m := re.match(r"dropped (\d+) good", t))]
    return sum(counts)


def test_dropped_reversals_equal_suspect_plus_missing_for_within_run_events():
    sessions = mixed_subject()
    totals = get_total_reversals(sessions)
    assert (totals["suspect_good_reversals"], totals["uncheckable_good_reversals"]) == (2, 1)
    assert dropped_count({"m": sessions}) == 2 + 1


def test_a_missing_magnitude_is_uncheckable_not_suspect():
    sess = session([0, 0, 1, 1], [A_BEST, A_BEST, (4, None, 2), (4, 1, 2)])
    assert count_suspect_good_reversals(sess) == {"suspect": 0, "uncheckable": 1}


def test_events_while_nothing_recorded_are_checked_and_dropped_when_they_fail():
    sessions = mixed_subject()
    # A reversal after ses-1's last trial: the counter of ses-2 starts one higher, and ses-1's
    # best (A) is also ses-2's first best, so the best tower is unchanged across the gap. The
    # import did not mark whether the next session shows it, so it is uncheckable.
    sessions["ses-1"]["reversal_boundary_events"] = [
        dict(kind="good", index=6, count=1, source="run_end")
    ]
    sessions["ses-1"]["reversal_totals"] = {"good": 3}
    # ses-3's trial-2 reversal entered while the rig was not recording; it is checked like any
    # reversal, and its best tower is unchanged.
    sessions["ses-3"]["reversal_boundary_events"] = [
        dict(kind="good", index=2, count=1, source="between_runs")
    ]
    totals = get_total_reversals(sessions)
    # suspect: the tie and ses-3's event; uncheckable: the missing magnitude and ses-1's event
    assert (totals["suspect_good_reversals"], totals["uncheckable_good_reversals"]) == (2, 2)
    # every one of them is dropped: the tie, the missing magnitude and both events
    assert dropped_count({"m": sessions}) == 2 + 2


def test_a_terminal_event_and_a_between_run_event_on_one_boundary_are_two_dropped_reversals():
    # ses-1 ends with a reversal after its last trial; ses-2 starts one higher because another
    # one happened between runs. Both enter the merged counter at ses-2's first trial.
    first = session(
        [0] * 4,
        [A_BEST] * 4,
        reversal_boundary_events=[dict(kind="good", index=4, count=1, source="run_end")],
        reversal_totals={"good": 1},
    )
    second = session(
        [1] * 4,
        [A_BEST] * 4,
        reversal_boundary_events=[dict(kind="good", index=0, count=1, source="between_runs")],
        reversal_totals={"good": 1},
    )
    sessions = {"ses-1": first, "ses-2": second}
    totals = get_total_reversals(sessions)
    assert (totals["suspect_good_reversals"], totals["uncheckable_good_reversals"]) == (0, 2)
    assert dropped_count({"m": sessions}) == 2


def test_a_jump_of_two_on_one_trial_is_two_dropped_reversals():
    sess = session([0, 0, 2, 2], [A_BEST] * 4)
    assert count_suspect_good_reversals(sess) == {"suspect": 2, "uncheckable": 0}
    assert dropped_count({"m": {"ses-1": sess}}) == 2


def test_a_reversal_skipped_by_the_zero_reward_gate_is_counted_but_not_dropped():
    # The gate (more than max_zero_count towers at zero before the reversal) skips the
    # boundary before the best-tower check, so it is in neither the drop warning nor the
    # windows, while the counts still classify it. This is the one documented exclusion of
    # "dropped = suspect + uncheckable".
    zeros = (4, 0, 0)
    sess = session([0, 0, 1, 1], [zeros] * 4)
    assert count_suspect_good_reversals(sess)["suspect"] == 1
    assert dropped_count({"m": {"ses-1": sess}}) == 0


@pytest.mark.parametrize(
    "problem, suspect, uncheckable",
    [
        ("tie for best", True, False),
        ("best tower unchanged", True, False),
        ("missing magnitude", False, True),
        (None, False, False),
    ],
)
def test_best_change_classes_are_disjoint(problem, suspect, uncheckable):
    assert (is_suspect(problem), is_uncheckable(problem)) == (suspect, uncheckable)


def test_a_dropped_suspect_reversal_still_ends_the_window_and_counts_as_a_removal():
    # Reversals at trials 3 (A to B), 6 (B to B: dropped) and 9 (B to A).
    magnitudes = [A_BEST] * 3 + [B_BEST] * 3 + [B_BEST] * 3 + [A_BEST] * 3
    data = {"m": {"ses-1": session([0] * 3 + [1] * 3 + [2] * 3 + [3] * 3, magnitudes)}}
    with pytest.warns(ReversalWarning, match="dropped 1 good"):
        windows = get_good_reversal_info(data, pre=1)
    assert [w["reversal_idx"] for w in windows["m"]] == [3, 9]
    assert windows["m"][0]["trial_window_idx"]["post"] == [3, 4, 5]  # ends at the dropped one
    kinds = get_reversal_indices_by_kind(data)
    assert kinds["good"] == {"m": [3, 6, 9]}
    steps = np.arange(0, 6)
    cumulative, _ = total_removal_curves(
        windows,
        steps,
        good_idx=kinds["good"],
        bad_idx=kinds["bad"],
        unclassified_idx=kinds["unclassified"],
    )
    # the anchor at 3 is removed 3 trials later, by the dropped reversal at 6
    assert cumulative.tolist() == [0, 0, 0, 1, 1, 1]
