"""A block that is open at the end of a session continues into the next session.

The sessions are merged per subject before blocks are cut, so a session boundary
must neither end a block nor start one; only a real reversal does.
"""

import pytest

from src.behavior_analysis.get_good_reversal_info import get_good_reversal_info
from src.behavior_analysis.get_rank_counts_by_good_reversal import get_rank_counts_by_good_reversal

RANK_NAMES = ("best", "second", "third")


# The best tower changes at merged trials 2, 3, 5 and 7, so every reversal used below
# has a different best tower after it than before it (otherwise it is dropped as a
# suspected detection bug).
BEST_CHANGES = (2, 3, 5, 7)


def magnitudes(first, n):
    """Per-tower magnitudes for merged trials first..first+n-1 (best tower 5, others 3 and 1)."""
    segments = [sum(i >= c for c in BEST_CHANGES) for i in range(first, first + n)]
    return {
        t: [5 if j == s % 3 else 3 if j == (s + 1) % 3 else 1 for s in segments]
        for j, t in enumerate("ABC")
    }


def make_session(n, *, first=0, blocks=None, good=None, **extra):
    """A synthetic session of *n* trials starting at merged trial *first*.

    Every trial picks one rank (cycling).
    """
    session = {
        "trial": list(range(1, n + 1)),
        "reward_magnitudes_by_tower": magnitudes(first, n),
        "choices_by_tower": {t: [int(i % 3 == k) for i in range(n)] for k, t in enumerate("ABC")},
        "choices_by_rank": {
            r: [int(i % 3 == k) for i in range(n)] for k, r in enumerate(RANK_NAMES)
        },
        "has_good": good is not None,
        **extra,
    }
    if blocks is not None:
        session["blocks"] = blocks
    if good is not None:
        session["good_reversals"] = good
    return session


def block_rows(data):
    windows = get_good_reversal_info(data, include_first_block=True)
    rows = get_rank_counts_by_good_reversal(windows)["m"]
    return windows["m"], rows


# Session 1 has 5 trials, session 2 has 4; the block open at the end of session 1 is
# trials 3-6 (0-based; the first 2 in session 1 and the last 2 in session 2).
LEGACY = {
    "ses-1": make_session(5, blocks=[1, 1, 1, 2, 2]),
    "ses-2": make_session(4, first=5, blocks=[1, 1, 2, 2]),
}
COUNTER = {
    "ses-1": make_session(5, good=[0, 0, 1, 1, 1]),
    "ses-2": make_session(4, first=5, good=[0, 0, 1, 1]),
}


@pytest.mark.parametrize(
    "sessions, spanning_trials",
    [(LEGACY, [3, 4, 5, 6]), (COUNTER, [2, 3, 4, 5, 6])],
    ids=["legacy blocks", "counter"],
)
def test_a_block_spanning_two_sessions_is_one_block_with_all_its_trials(sessions, spanning_trials):
    windows, rows = block_rows({"m": sessions})
    # trial 4 is the first trial of session 2: the window holds trials from both sessions
    assert [w for w in windows if w["trial_window_idx"]["post"] == spanning_trials]
    assert len(windows) == 3 and len(rows) == 3
    assert sum(r["n_trials"] for r in rows) == 9  # every trial in exactly one block
    assert [r["n_trials"] for r in rows] == [len(w["trial_window_idx"]["post"]) for w in windows]


def test_legacy_block_boundaries_are_only_the_real_block_changes():
    windows, _ = block_rows({"m": LEGACY})
    assert [w["reversal_idx"] for w in windows] == [0, 3, 7]


def test_counter_boundaries_are_only_the_real_reversals():
    windows, _ = block_rows({"m": COUNTER})
    assert [w["reversal_idx"] for w in windows] == [0, 2, 7]


def test_legacy_reversal_at_a_sessions_first_trial_is_a_boundary():
    # the reversal happened after session 1's last trial, so it is carried into session 2
    events = [{"kind": "good", "index": 5, "count": 1}]
    sessions = {
        "ses-1": make_session(5, blocks=[1, 1, 1, 2, 2], reversal_boundary_events=events),
        "ses-2": LEGACY["ses-2"],
    }
    windows, rows = block_rows({"m": sessions})
    assert [w["reversal_idx"] for w in windows] == [0, 3, 5, 7]
    assert [r["n_trials"] for r in rows] == [3, 2, 2, 2]


def test_counter_reversal_at_a_sessions_first_trial_is_a_boundary():
    sessions = {
        "ses-1": make_session(5, good=[0, 0, 1, 1, 1], reversal_totals={"good": 2}),
        "ses-2": COUNTER["ses-2"],
    }
    windows, _ = block_rows({"m": sessions})
    assert [w["reversal_idx"] for w in windows] == [0, 2, 5, 7]


def test_n_trials_shows_how_short_the_final_block_is():
    _, rows = block_rows({"m": LEGACY})
    assert rows[-1]["n_trials"] == 2 and rows[-1]["total"] == 2
    assert [r["n_trials"] for r in rows] == [3, 4, 2]


def test_session_order_follows_the_session_number_not_the_dict_order():
    reordered = {"ses-2": LEGACY["ses-2"], "ses-1": LEGACY["ses-1"]}
    windows, _ = block_rows({"m": reordered})
    assert [w["reversal_idx"] for w in windows] == [0, 3, 7]
