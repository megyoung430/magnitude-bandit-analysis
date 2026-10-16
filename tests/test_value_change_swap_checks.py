"""Checks that the rank-swap value-change curves mean what their labels say.

Rank-swap events for a best-value decrease: the best arm's value drops below the second's
within a block. Labels name the arms by their rank at t = -1: ``prev_best`` is the arm that was
best just before the drop. These tests pin that meaning three ways:

- a mouse that always picks the true best gives ``prev_best`` = 1 wherever that arm was best,
  and ``prev_second`` = 1 from the swap on;
- an independent re-implementation of the event rules and the curves on random drifting
  sessions matches the module exactly;
- a mouse whose preference lags the values by a few trials picks ``prev_second`` before a
  swap that comes soon after the previous rank change, which is how the real curves can show
  ``prev_second`` above ``prev_best`` before t = 0 even though mice prefer the best overall;
- the pre window stops where the best arm last changed, so trials from before the previous
  rank change never enter the pre-event curves.
"""

import numpy as np
import pytest

from src.behavior_analysis.value_change_windows import get_value_change_windows, summarize_windows

ARMS = ("A", "B", "C")


def _session(rms, choices):
    n = len(rms)
    return {
        "trial_variables": [
            {"reward_magnitudes": list(rms), "choice": list(choices), "blocks": [1] * n}
        ]
    }


def _true_best(m):
    ordered = sorted(m, key=m.get, reverse=True)
    return ordered[0]


def _swap_curves(sessions, pre=5, post=15):
    w = get_value_change_windows(
        {"m": sessions}, "best", "decrease", "rank_swap", pre=pre, post=post
    )
    return w, summarize_windows(w)


def _drifting_magnitudes(rng, n):
    """A and B drift without ever tying (A even, B odd); C is the zero arm all block long."""
    a, b, out = 4, 3, []
    for _ in range(n):
        if rng.random() < 0.25:
            a = int(rng.choice([2, 4]))
        if rng.random() < 0.25:
            b = int(rng.choice([1, 3]))
        out.append({"A": a, "B": b, "C": 0})
    return out


def test_true_best_chooser_scores_prev_best_before_and_prev_second_after_one_swap():
    rms = [{"A": 3, "B": 2, "C": 0}] * 10 + [{"A": 1, "B": 2, "C": 0}] * 10
    choices = [_true_best(m) for m in rms]
    w, curves = _swap_curves({"ses-1_date-20260101": _session(rms, choices)})
    assert w.counts["n_events"] == 1
    t = curves.x
    assert np.all(curves.mean["prev_best"][t < 0] == 1.0)
    assert np.all(curves.mean["prev_second"][(t >= 0) & (t <= 9)] == 1.0)
    assert np.all(curves.mean["third"][~np.isnan(curves.mean["third"])] == 0.0)


@pytest.mark.parametrize("seed", range(5))
def test_true_best_chooser_picks_prev_best_exactly_where_that_arm_was_best(seed):
    rng = np.random.default_rng(seed)
    rms = _drifting_magnitudes(rng, 400)
    choices = [_true_best(m) for m in rms]
    w, _ = _swap_curves({"ses-1_date-20260101": _session(rms, choices)}, pre=5, post=10)
    assert w.counts["n_events"] > 0
    for event in w.events["m"]:
        i = event.trial_idx
        prev_best = _true_best(rms[i - 1])
        for k, label in zip(range(5, 0, -1), event.pre, strict=True):
            if label is not None:
                assert (label == "prev_best") == (_true_best(rms[i - k]) == prev_best)


def _oracle_swap_events(rms, choices, pre, post):
    """Independent rule: within one block with C fixed at 0, A/B never tie, so an event is a
    trial where the best arm's value fell and the other arm is now above it."""
    events = []
    for i in range(1, len(rms)):
        before, after = rms[i - 1], rms[i]
        best_before = max(before["A"], before["B"])
        best_after = max(after["A"], after["B"])
        old_best = "A" if before["A"] > before["B"] else "B"
        old_second = "B" if old_best == "A" else "A"
        if best_after < best_before and after[old_second] > after[old_best]:

            def label(j, old_best=old_best, old_second=old_second):
                c = choices[j]
                return (
                    "prev_best" if c == old_best else "prev_second" if c == old_second else "third"
                )

            # the pre window stops where the best arm last changed
            stable = 0
            while (
                stable < pre and i - stable - 1 >= 0 and _true_best(rms[i - stable - 1]) == old_best
            ):
                stable += 1
            pre_labels = [label(i - k) if k <= stable else None for k in range(pre, 0, -1)]
            post_labels = [label(i + k) if i + k < len(rms) else None for k in range(post + 1)]
            events.append(pre_labels + post_labels)
    return events


@pytest.mark.parametrize("seed", range(8))
def test_swap_curves_match_an_independent_computation(seed):
    rng = np.random.default_rng(100 + seed)
    rms = _drifting_magnitudes(rng, 500)
    choices = [str(rng.choice(ARMS, p=[0.5, 0.3, 0.2])) for _ in rms]
    pre, post = 5, 10
    _, curves = _swap_curves({"ses-1_date-20260101": _session(rms, choices)}, pre=pre, post=post)
    events = _oracle_swap_events(rms, choices, pre, post)
    assert curves.num_events == len(events) > 0
    for label in ("prev_best", "prev_second", "third"):
        expected = []
        for t in range(pre + post + 1):
            column = [e[t] for e in events if e[t] is not None]
            expected.append(np.mean([c == label for c in column]) if column else np.nan)
        np.testing.assert_allclose(curves.mean[label], expected, equal_nan=True)


def test_a_lagging_chooser_prefers_prev_second_before_a_swap_soon_after_a_rank_change():
    # B is best for trials 0-9; A becomes best at 10; at 13 A drops below B again (the event).
    rms = (
        [{"A": 1, "B": 3, "C": 0}] * 10
        + [{"A": 4, "B": 3, "C": 0}] * 3
        + [{"A": 2, "B": 3, "C": 0}] * 10
    )
    lag = 5  # the mouse picks the arm that was best five trials ago
    choices = [_true_best(rms[max(0, t - lag)]) for t in range(len(rms))]
    w, curves = _swap_curves({"ses-1_date-20260101": _session(rms, choices)}, pre=5, post=5)
    assert w.counts["n_events"] == 1
    t = curves.x
    # t = -3..-1 (trials 10-12): A was best but the mouse still picks B, the prev second.
    assert np.all(curves.mean["prev_second"][(t >= -3) & (t <= -1)] == 1.0)
    assert np.all(curves.mean["prev_best"][(t >= -3) & (t <= -1)] == 0.0)
    # t = -5, -4 (trials 8-9) had B as best, so they are cut from the pre window.
    assert curves.n_events_per_bin[(t == -5) | (t == -4)].tolist() == [0, 0]


def test_pre_window_stops_where_the_best_arm_last_changed_for_every_config():
    # B best at trials 0-6, A best from 7; A's value rises at 10 (a best/increase event).
    rms = (
        [{"A": 1, "B": 3, "C": 0}] * 7
        + [{"A": 2, "B": 1, "C": 0}] * 3
        + [{"A": 4, "B": 1, "C": 0}] * 5
    )
    choices = ["A"] * len(rms)
    w = get_value_change_windows(
        {"m": {"ses-1_date-20260101": _session(rms, choices)}}, "best", "increase", "all",
        pre=5, post=3,
    )  # fmt: skip
    (event,) = w.events["m"]
    assert event.trial_idx == 10
    # t = -3..-1 are trials 7-9 (A best); t = -5, -4 are trials 5-6 (B best) and are cut.
    assert event.pre == (None, None, "best", "best", "best")


def test_a_tied_best_at_t_minus_one_keeps_only_tied_trials():
    rms = (
        [{"A": 3, "B": 1, "C": 0}] * 4
        + [{"A": 2, "B": 2, "C": 0}] * 3
        + [{"A": 2, "B": 4, "C": 0}] * 3
    )
    w = get_value_change_windows(
        {"m": {"ses-1_date-20260101": _session(rms, ["A"] * len(rms))}}, "best", "increase",
        "all", pre=5, post=2,
    )  # fmt: skip
    (event,) = w.events["m"]
    assert event.trial_idx == 7
    assert [label is not None for label in event.pre] == [False, False, True, True, True]
