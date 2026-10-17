"""Tests for value-change-aligned windows (src.behavior_analysis.value_change_windows).

Every fixture is built here from three arms A, B, C whose magnitudes follow a
piecewise-constant script, so the expected events, windows and counts are known.
"""

import numpy as np
import pytest

from src.behavior_analysis.value_change_windows import (
    ValueChangeEvent,
    aggregate_windows,
    contamination_summary,
    get_value_change_windows,
    merge_value_change_windows,
    value_change_configs,
)
from src.behavior_import.extract_trials import merge_trial_variables


def script(n, start, changes):
    """Per-trial ``{arm: magnitude}`` list: *start* until a trial in *changes* sets a new state."""
    out, state = [], dict(start)
    for t in range(n):
        state = dict(changes.get(t, state))
        out.append(dict(state))
    return out


def recording(rms, choices=None, blocks=None):
    """One recording's trial variables; the animal cycles A, B, C unless *choices* is given."""
    n = len(rms)
    choices = choices or ["ABC"[t % 3] for t in range(n)]
    return {
        "reward_magnitudes": list(rms),
        "choice": list(choices),
        "blocks": list(blocks) if blocks is not None else [1] * n,
    }


def session(*recordings):
    return {"trial_variables": list(recordings)}


def mag(a, b, c):
    return {"A": a, "B": b, "C": c}


def key(number, day=None):
    return f"ses-{number}_date-2026010{day or number}"


def windows(sessions, **kwargs):
    """Windows for mouse "m" from *sessions* ``{key: session}``; defaults best/increase/all."""
    kwargs.setdefault("rank", "best")
    kwargs.setdefault("direction", "increase")
    kwargs.setdefault("rank_filter", "all")
    kwargs.setdefault("pre", 5)
    kwargs.setdefault("post", 15)
    return get_value_change_windows({"m": sessions}, **kwargs)


BASE = mag(4, 2, 1)


# ------------------------------------------------------------------ events and windows


def test_best_increase_event_has_known_window_and_labels():
    rms = script(30, BASE, {10: mag(5, 2, 1)})
    w = windows({key(1): session(recording(rms))})
    (event,) = w.events["m"]
    assert (event.session, event.trial_idx) == (key(1), 10)
    assert len(event.pre) == 5 and len(event.post) == 16
    cycle = ["best", "second", "third"]
    assert event.pre == tuple(cycle[j % 3] for j in range(5, 10))
    assert event.post[0] == cycle[10 % 3]  # post[0] is the first trial with the new value
    assert event.post == tuple(cycle[j % 3] for j in range(10, 26))


def test_direction_and_rank_select_the_tracked_value():
    rms = script(40, BASE, {10: mag(4, 3, 1), 25: mag(4, 3, 0)})
    kinds = {
        ("best", "increase"): 0,
        ("best", "decrease"): 0,
        ("second", "increase"): 1,
        ("second", "decrease"): 0,
    }
    for (rank, direction), expected in kinds.items():
        w = windows({key(1): session(recording(rms))}, rank=rank, direction=direction)
        assert sum(map(len, w.events.values())) == expected, (rank, direction)
    # third-arm change (25) is never an event for best or second
    w = windows({key(1): session(recording(rms))}, rank="second", direction="increase")
    assert [e.trial_idx for e in w.events["m"]] == [10]


def test_events_near_the_session_edges_are_kept_with_missing_trials_as_none():
    pre, post, n = 5, 15, 30
    # events at 4, 5, 14 and 15; the first has one trial too few before it, the last has
    # none after trial 29 for its final offset
    rms = script(
        n, mag(3, 2, 0), {4: mag(4, 2, 0), 5: mag(5, 2, 0), 14: mag(6, 2, 0), 15: mag(7, 2, 0)}
    )
    w = windows({key(1): session(recording(rms))}, pre=pre, post=post)
    by_idx = {e.trial_idx: e for e in w.events["m"]}
    assert sorted(by_idx) == [4, 5, 14, 15]
    assert by_idx[4].pre[0] is None and None not in by_idx[4].pre[1:]
    assert None not in by_idx[5].pre and None not in by_idx[14].post
    assert by_idx[15].post[-1] is None and None not in by_idx[15].post[:-1]
    assert w.counts["n_events"] == w.counts["n_changes"] == 4


def test_every_detected_change_is_accounted_for():
    rms = script(40, BASE, {3: mag(5, 2, 1), 12: mag(4, 3, 1), 14: mag(4, 3, 2), 20: mag(2, 3, 2)})
    for config in value_change_configs():
        w = windows(
            {key(1): session(recording(rms))},
            rank=config.rank,
            direction=config.direction,
            rank_filter=config.rank_filter,
        )
        c = w.counts
        assert c["n_changes"] == (
            c["n_third_arm_changed"] + c["n_tie_skipped"] + c["n_filtered_out"] + c["n_events"]
        ), config.label


def test_changes_at_a_block_boundary_are_not_events_and_cut_the_window():
    rms = script(40, BASE, {10: mag(5, 2, 1), 20: mag(1, 6, 2)})
    blocks = [1] * 20 + [2] * 20
    w = windows({key(1): session(recording(rms, blocks=blocks))})
    (event,) = w.events["m"]  # the change at 20 is a reversal (new block), not an event
    assert event.trial_idx == 10
    assert event.post[:10] != (None,) * 10
    assert event.post[10:] == (None,) * 6  # trials 20.. belong to the next block


def test_the_window_is_cut_where_the_third_arm_changes_within_a_block():
    # C is the third arm until trial 14, when B drops below it; the event at 10 has post
    # trials 10..13 only, and the event at 24 (A rises) has pre trials back to 14 only
    rms = script(40, BASE, {10: mag(5, 2, 1), 14: mag(5, 0, 1), 24: mag(6, 0, 1)})
    (first, second) = windows({key(1): session(recording(rms))}).events["m"]
    assert first.trial_idx == 10 and second.trial_idx == 24
    assert None not in first.post[:4] and first.post[4:] == (None,) * 12
    assert not first.contaminated  # the change at 14 is outside the window
    assert None not in second.pre  # trials 19..23 all have B as the third arm


def test_the_third_arm_is_fixed_over_pre_trials_the_window_stops_at_the_first_difference():
    rms = script(40, mag(4, 0, 1), {8: mag(4, 3, 1), 12: mag(5, 3, 1)})
    (event,) = windows({key(1): session(recording(rms))}).events["m"]
    assert event.trial_idx == 12
    # The pre window t=-5..-1 covers trials 7..11. B is the third arm before trial 8 and C from
    # trial 8 on, so only trial 7 is outside the window.
    assert event.pre[0] is None and None not in event.pre[1:]


def test_an_event_that_changes_the_third_arm_is_excluded_and_counted():
    # A falls to the bottom: the old best becomes the third arm, a reversal-type change
    rms = script(40, BASE, {10: mag(0, 2, 1)})
    s = {key(1): session(recording(rms))}
    for rank_filter in ("all", "rank_preserving"):
        w = windows(s, rank="best", direction="decrease", rank_filter=rank_filter)
        assert w.events == {} and w.counts["n_third_arm_changed"] == 1, rank_filter
    swap = windows(s, rank="best", direction="decrease", rank_filter="rank_swap")
    assert swap.events == {} and swap.counts["n_third_arm_changed"] == 1


def test_a_block_change_at_the_event_with_the_same_third_arm_is_not_an_event():
    rms = script(40, BASE, {10: mag(5, 2, 1)})  # C stays the third arm
    blocks = [1] * 10 + [2] * 30
    w = windows({key(1): session(recording(rms, blocks=blocks))})
    assert w.events == {} and w.counts["n_changes"] == 0


def test_an_event_at_trial_one_has_only_the_first_trial_before_it():
    rms = script(30, BASE, {1: mag(5, 2, 1)})
    (event,) = windows({key(1): session(recording(rms))}).events["m"]
    assert event.trial_idx == 1
    assert event.pre[:-1] == (None,) * 4 and event.pre[-1] == "best"  # trial 0: A is best


def test_a_nan_magnitude_makes_the_trial_missing():
    rms = script(40, BASE, {10: mag(5, 2, 1)})
    rms[12] = mag(5, float("nan"), 1)
    w = windows({key(1): session(recording(rms))})
    (event,) = w.events["m"]
    assert w.counts["n_missing_magnitude_trials"] == 1
    assert None not in event.post[:2] and event.post[2:] == (None,) * 14


def test_a_tied_lowest_magnitude_on_either_side_counts_as_tie_skipped():
    for before, after in ((BASE, mag(5, 1, 1)), (mag(4, 1, 1), mag(5, 2, 1))):
        rms = script(40, before, {10: after})
        w = windows({key(1): session(recording(rms))})
        c = w.counts
        assert w.events == {}
        assert (c["n_changes"], c["n_tie_skipped"]) == (1, 1), (before, after)
        assert c["n_third_arm_changed"] == c["n_filtered_out"] == 0


def test_a_recording_restart_inside_a_block_keeps_its_value_change_as_an_event():
    # The task continues the same drifting values across a restart. The importer's merge keeps
    # the block number (the second recording starts at the running block), so the step from
    # the last trial of recording 1 to the first of recording 2 is an ordinary event.
    first = recording(script(20, BASE, {}), blocks=[1] * 20)
    second = recording(script(30, mag(5, 2, 1), {}), blocks=[1] * 30)
    merged = merge_trial_variables([first, second])
    assert merged["blocks"] == [1] * 50
    (event,) = windows({key(1): session(first, second)}).events["m"]
    assert event.trial_idx == 20 and None not in event.pre and None not in event.post


def test_a_tied_lowest_magnitude_leaves_the_third_arm_undetermined():
    # B and C are both 1: even "all" cannot tell which arm is third
    rms = script(40, mag(4, 1, 1), {10: mag(5, 1, 1)})
    w = windows({key(1): session(recording(rms))})
    assert w.events == {} and w.counts["n_tie_skipped"] == 1


def test_a_trial_without_magnitudes_ends_the_window():
    rms = script(40, BASE, {10: mag(5, 2, 1), 15: mag(1, 2, 0)})
    rms[12] = None
    (event,) = windows({key(1): session(recording(rms))}).events["m"]
    assert None not in event.post[:2] and event.post[2:] == (None,) * 14
    assert not event.contaminated  # the change at 15 lies beyond the missing trial


def test_arms_that_change_within_a_window_raise():
    rms = script(40, BASE, {10: mag(5, 2, 1)})
    rms[13] = {"X": 5, "B": 2, "C": 1}  # same third arm, but the best arm has another name
    with pytest.raises(ValueError, match="arms changed"):
        windows({key(1): session(recording(rms))})


def test_a_trial_with_other_than_three_arms_raises():
    rms = script(40, BASE, {10: mag(5, 2, 1)})
    rms[20] = {"A": 4, "B": 2}
    with pytest.raises(ValueError, match="3 arms"):
        windows({key(1): session(recording(rms))})


def test_missing_or_misaligned_blocks_raise():
    rms = script(30, BASE, {10: mag(5, 2, 1)})
    rec = recording(rms)
    del rec["blocks"]
    with pytest.raises(ValueError, match="blocks"):
        windows({key(1): session(rec)})
    rec = recording(rms)
    rec["blocks"] = rec["blocks"][:-1]
    with pytest.raises(ValueError, match="differ in length"):
        windows({key(1): session(rec)})


def test_sessions_without_trials_are_counted_not_dropped_silently():
    rms = script(30, BASE, {10: mag(5, 2, 1)})
    w = windows({key(1): {"trial_variables": {}}, key(2): session(recording(rms))})
    assert w.counts["n_sessions_without_trials"] == 1
    assert w.counts["n_events"] == 1


def test_invalid_arguments_raise():
    s = {key(1): session(recording(script(30, BASE, {})))}
    with pytest.raises(ValueError, match="rank"):
        windows(s, rank="third")
    with pytest.raises(ValueError, match="direction"):
        windows(s, direction="up")
    with pytest.raises(ValueError, match="rank_filter"):
        windows(s, rank_filter="swap")
    with pytest.raises(ValueError, match="impossible"):
        windows(s, rank="best", direction="increase", rank_filter="rank_swap")
    with pytest.raises(ValueError, match="impossible"):
        windows(s, rank="second", direction="decrease", rank_filter="rank_swap")
    with pytest.raises(ValueError, match="pre"):
        windows(s, pre=0)


def test_configs_skip_exactly_the_impossible_swaps():
    labels = [c.label for c in value_change_configs()]
    assert len(labels) == 10
    assert "best/increase/rank_swap" not in labels
    assert "second/decrease/rank_swap" not in labels
    assert "best/decrease/rank_swap" in labels and "second/increase/rank_swap" in labels


# ------------------------------------------------------------- multi-recording sessions


def test_all_recordings_of_a_session_are_used():
    first = recording(script(30, BASE, {}))
    second = recording(script(30, BASE, {10: mag(5, 2, 1)}))
    w = windows({key(1): session(first, second)})
    # the event is at trial 30 + 10 of the merged session; reading only recording 0 finds none
    assert [e.trial_idx for e in w.events["m"]] == [40]


# --------------------------------------------------------------------- session order


def test_sessions_are_visited_in_session_order_not_dict_order():
    rms = script(30, BASE, {10: mag(5, 2, 1)})
    sessions = {key(3): session(recording(rms)), key(1): session(recording(rms))}
    sessions[key(2)] = session(recording(rms))
    w = windows(sessions)
    assert [e.session for e in w.events["m"]] == [key(1), key(2), key(3)]


def test_windows_do_not_cross_sessions():
    rms = script(30, BASE, {3: mag(5, 2, 1)})  # change 3 trials into the session
    sessions = {key(1): session(recording(script(30, BASE, {}))), key(2): session(recording(rms))}
    (event,) = windows(sessions).events["m"]
    assert event.session == key(2)
    assert event.pre[:2] == (None, None) and None not in event.pre[2:]


# -------------------------------------------------------------------- classification


def swap_script():
    # B (second) rises above A at trial 10: second/increase swap; best value also rises.
    return script(40, BASE, {10: mag(4, 6, 1)})


def test_rank_swap_requires_the_tracked_arm_to_move_and_the_top_two_to_exchange():
    s = {key(1): session(recording(swap_script()))}
    swap = windows(s, rank="second", direction="increase", rank_filter="rank_swap")
    assert [e.trial_idx for e in swap.events["m"]] == [10]
    # the best-position value also rose, but A (the best arm) did not move: not a best swap
    best_up_all = windows(s, rank="best", direction="increase", rank_filter="all")
    assert [e.trial_idx for e in best_up_all.events["m"]] == [10]
    best_up_pres = windows(s, rank="best", direction="increase", rank_filter="rank_preserving")
    assert best_up_pres.events == {}
    assert best_up_pres.counts["n_filtered_out"] == 1


def test_rank_preserving_keeps_the_rank_holder():
    rms = script(40, BASE, {10: mag(4, 3, 1)})  # B moves 2 -> 3, still second
    s = {key(1): session(recording(rms))}
    pres = windows(s, rank="second", direction="increase", rank_filter="rank_preserving")
    swap = windows(s, rank="second", direction="increase", rank_filter="rank_swap")
    assert [e.trial_idx for e in pres.events["m"]] == [10]
    assert swap.events == {}


# ----------------------------------------------------------------------------- ties


def test_ties_are_counted_and_skipped_when_the_filter_depends_on_them():
    # best falls 4 -> 3 and ties with B: whether A stays best is undefined
    rms = script(40, BASE, {10: mag(3, 3, 1)})
    s = {key(1): session(recording(rms))}
    kept = windows(s, rank="best", direction="decrease", rank_filter="all")
    assert len(kept.events["m"]) == 1  # "all" needs no ordering, so ties do not matter
    for rank_filter in ("rank_preserving", "rank_swap"):
        w = windows(s, rank="best", direction="decrease", rank_filter=rank_filter)
        assert w.events == {}
        assert w.counts["n_tie_skipped"] == 1, rank_filter


def test_tied_best_and_second_trials_split_the_choice_evenly():
    # from trial 12 on A and B tie at 4; the animal chooses A at 12 and C at 13
    rms = script(40, BASE, {10: mag(5, 2, 1), 12: mag(4, 4, 1)})
    choices = ["A"] * 40
    choices[13] = "C"
    (event,) = windows({key(1): session(recording(rms, choices=choices))}).events["m"]
    assert event.post[2] == ("best", "second") and event.post[3] == "third"
    x, _, across = aggregate_windows({"m": [event]}, 5, 15, ("best", "second", "third"))
    assert across["mean"]["best"][5 + 2] == 0.5 and across["mean"]["second"][5 + 2] == 0.5


# --------------------------------------------------------------------- label sets


def test_rank_swap_windows_use_previous_best_and_second_labels():
    s = {key(1): session(recording(swap_script()))}
    w = windows(s, rank="second", direction="increase", rank_filter="rank_swap")
    (event,) = w.events["m"]
    assert set(event.pre + event.post) <= {"prev_best", "prev_second", "third"}
    assert event.pre[0] == "third"  # trial 5 is a choice of C, the arm that was third


def test_swap_labels_name_the_arms_by_their_rank_before_the_event():
    # B (second) passes A (best) at trial 10. The animal cycles A, B, C by trial number:
    # trial 10 -> B was second, trial 12 -> A was best, trial 11 -> C was third.
    s = {key(1): session(recording(swap_script()))}
    (event,) = windows(s, rank="second", direction="increase", rank_filter="rank_swap").events["m"]
    assert event.post[0] == "prev_second"
    assert event.post[1] == "third"
    assert event.post[2] == "prev_best"


def test_aggregation_uses_the_filter_label_set_even_when_the_first_label_is_third():
    s = {key(1): session(recording(swap_script()))}
    w = windows(s, rank="second", direction="increase", rank_filter="rank_swap")
    x, _, across = aggregate_windows(w.events, 5, 15, w.labels)
    assert set(across["mean"]) == {"prev_best", "prev_second", "third"}
    expected = {lab: np.zeros(21) for lab in across["mean"]}
    (event,) = w.events["m"]
    for t, lab in enumerate(event.pre + event.post):
        expected[lab][t] = 1.0
    for lab, curve in expected.items():
        np.testing.assert_array_equal(across["mean"][lab], curve)
    assert across["mean"]["prev_best"].sum() > 0
    assert list(x) == list(range(-5, 16))


def test_aggregation_rejects_labels_outside_the_label_set():
    event = ValueChangeEvent("m", key(1), 5, ("best",) * 2, ("best",) * 3, False, None, False)
    with pytest.raises(ValueError, match="best"):
        aggregate_windows({"m": [event]}, 2, 2, ("prev_best", "prev_second", "third"))


def test_aggregation_rejects_windows_of_the_wrong_length():
    event = ValueChangeEvent("m", key(1), 5, ("best",) * 2, ("best",) * 3, False, None, False)
    with pytest.raises(ValueError, match="length"):
        aggregate_windows({"m": [event]}, 2, 5, ("best", "second", "third"))


# -------------------------------------------------------------------- standard error


def ev(pre, post, subject="m"):
    return ValueChangeEvent(subject, key(1), 5, tuple(pre), tuple(post), False, None, False)


def test_se_counts_only_subjects_with_data_and_is_nan_for_one():
    std = ("best", "second", "third")
    a = [ev(["best", "best"], ["best", "best"], "a")]
    b = [ev(["second", "best"], ["best", None], "b")]
    c = [ev(["best", "best"], ["second", None], "c")]
    _, _, across = aggregate_windows({"a": a, "b": b, "c": c}, 2, 1, std)
    best = across["mean"]["best"]
    np.testing.assert_allclose(best, [2 / 3, 1.0, 2 / 3, 1.0])
    # last point: subjects b and c are missing, so one value -> NaN SE (not 0)
    assert np.isnan(across["se"]["best"][3])
    assert across["se"]["best"][0] == pytest.approx(np.std([1, 0, 1], ddof=1) / np.sqrt(3))
    # t=0 point: all three present
    assert across["se"]["best"][2] == pytest.approx(np.std([1, 1, 0], ddof=1) / np.sqrt(3))
    _, _, one = aggregate_windows({"a": a}, 2, 1, std)
    assert np.isnan(one["se"]["best"]).all()


def test_event_weighting_counts_every_event_once():
    std = ("best", "second", "third")
    # mouse a: three events, all best; mouse b: one event, second
    a = [ev(["best"], ["best", "best"], "a") for _ in range(3)]
    b = [ev(["second"], ["second", None], "b")]
    _, _, by_mouse = aggregate_windows({"a": a, "b": b}, 1, 1, std)
    _, per_subject, by_event = aggregate_windows({"a": a, "b": b}, 1, 1, std, weighting="event")
    np.testing.assert_allclose(by_mouse["mean"]["best"], [0.5, 0.5, 1.0])
    np.testing.assert_allclose(by_event["mean"]["best"], [0.75, 0.75, 1.0])
    np.testing.assert_allclose(by_event["mean"]["second"], [0.25, 0.25, 0.0])
    se = np.std([1, 1, 1, 0], ddof=1) / 2
    assert by_event["se"]["best"][0] == pytest.approx(se)
    assert by_event["se"]["best"][2] == 0.0  # three events, all best
    assert by_event["weighting"] == "event" and by_mouse["weighting"] == "subject"
    # per-mouse curves and counts do not depend on the weighting
    np.testing.assert_allclose(per_subject["b"]["second"], [1.0, 1.0, np.nan])
    np.testing.assert_array_equal(by_event["n_events_per_bin"], by_mouse["n_events_per_bin"])


def test_event_weighting_splits_ties_and_skips_missing_trials_across_mice():
    labels = ("prev_best", "prev_second", "third")
    tie = ("prev_best", "prev_second")
    # mouse a: two events (a tie, and a missing trial at t=-1); mouse b: one prev_best event
    a = [ev([tie], ["third", "third"], "a"), ev([None], ["third", "third"], "a")]
    b = [ev(["prev_best"], ["prev_best", None], "b")]
    _, _, by_event = aggregate_windows({"a": a, "b": b}, 1, 1, labels, weighting="event")
    _, _, by_mouse = aggregate_windows({"a": a, "b": b}, 1, 1, labels)
    # t=-1: two valid events (tie, prev_best) -> prev_best 0.75; mice: (0.5 + 1) / 2 also 0.75
    assert by_event["mean"]["prev_best"][0] == pytest.approx(0.75)
    # t=0: three events (third, third, prev_best) vs mice (1 and 0)
    assert by_event["mean"]["third"][1] == pytest.approx(2 / 3)
    assert by_mouse["mean"]["third"][1] == pytest.approx(0.5)
    # t=1: only mouse a's two events
    assert by_event["mean"]["third"][2] == 1.0


def test_event_weighting_with_no_events_is_nan():
    _, _, across = aggregate_windows(
        {"m": []}, 1, 1, ("best", "second", "third"), weighting="event"
    )
    assert np.isnan(across["mean"]["best"]).all() and np.isnan(across["se"]["best"]).all()


def test_unknown_weighting_raises():
    with pytest.raises(ValueError, match="weighting"):
        aggregate_windows(
            {"m": [ev(["best"], ["best", "best"])]}, 1, 1, ("best",), weighting="mice"
        )


def test_time_points_without_data_are_nan_and_do_not_warn(recwarn):
    std = ("best", "second", "third")
    _, per_subject, across = aggregate_windows(
        {"a": [ev(["best", None], ["best", None])]}, 2, 1, std
    )
    assert np.isnan(per_subject["a"]["best"][1]) and np.isnan(across["mean"]["best"][3])
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


# ---------------------------------------------------------------------- contamination


def contaminated_session():
    # event at 10 (best 4 -> 5); at 14 A falls below B, a best/second exchange that is not
    # itself a best-increase event
    return session(recording(script(40, BASE, {10: mag(5, 2, 1), 14: mag(1, 2, 0)})))


def test_contamination_flags_later_changes_and_rank_swaps_within_the_block():
    w = windows({key(1): contaminated_session()})
    assert w.events["m"][0].contaminated
    assert w.events["m"][0].rank_swap_offset == 4
    summary = contamination_summary(w.events)
    assert summary == {
        "n_events": 1,
        "n_contaminated": 1,
        "frac_contaminated": 1.0,
        "n_rank_swap": 1,
        "frac_rank_swap": 1.0,
        "n_reference_tied": 0,
    }


def test_contamination_ignores_changes_beyond_the_post_window_and_other_blocks():
    rms = script(60, BASE, {10: mag(5, 2, 1), 40: mag(1, 2, 0)})  # change far beyond post
    quiet = windows({key(1): session(recording(rms))})
    assert not quiet.events["m"][0].contaminated
    rms = script(40, BASE, {10: mag(5, 2, 1), 14: mag(1, 3, 2)})
    blocks = [1] * 14 + [2] * 26  # the change at 14 is a reversal
    cut = windows({key(1): session(recording(rms, blocks=blocks))})
    assert not cut.events["m"][0].contaminated


def legacy_diagnose(sessions_by_subject, events_by_subject, post):
    """The notebook's diagnostic: one concatenated sequence per mouse, session-local indices."""
    n_contaminated = 0
    for subject, events in events_by_subject.items():
        rm_full = []
        for sess in sessions_by_subject[subject].values():
            rm_full.extend(sess["trial_variables"][0]["reward_magnitudes"])
        for event in events:
            for offset in range(1, post + 1):
                j = event.trial_idx + offset
                if j >= len(rm_full):
                    break
                if rm_full[j] != rm_full[j - 1]:
                    n_contaminated += 1
                    break
    return n_contaminated


def test_contamination_is_checked_at_session_local_trials_in_every_session():
    quiet = session(recording(script(40, BASE, {10: mag(5, 2, 1)})))
    contaminated = contaminated_session()
    sessions = {key(1): quiet, key(2): contaminated}
    w = windows(sessions)
    assert [e.contaminated for e in w.events["m"]] == [False, True]
    assert contamination_summary(w.events)["n_contaminated"] == 1
    # concatenating sessions and indexing with session-local positions looks at the wrong
    # trials of every session after the first, and here finds no contamination at all
    assert legacy_diagnose({"m": sessions}, w.events, 15) == 0


# ------------------------------------------------------------------------ truncation


def truncated(rms, **kwargs):
    kwargs.setdefault("rank", "best")
    kwargs.setdefault("direction", "increase")
    return windows({key(1): session(recording(rms))}, truncate_at_rank_swap=True, **kwargs)


def test_truncation_cuts_at_the_first_later_best_second_exchange():
    # best rises at 10; at 14 A falls below B (an exchange that does not undo the event ordering)
    rms = script(40, BASE, {10: mag(5, 2, 1), 14: mag(1, 2, 0)})
    (event,) = truncated(rms).events["m"]
    assert event.truncated and event.rank_swap_offset == 4
    assert None not in event.post[:4]
    assert event.post[4:] == (None,) * 12


def test_truncation_leaves_other_later_changes_alone():
    rms = script(40, BASE, {10: mag(5, 2, 1), 14: mag(5, 3, 1)})  # B rises but stays second
    (event,) = truncated(rms).events["m"]
    assert not event.truncated and None not in event.post
    assert event.contaminated


def test_a_swap_event_is_not_truncated_by_its_own_exchange():
    # B passes A at 10 (the event); the order at t = 0 is the reference, so only a later
    # return of A above B truncates
    rms = script(60, BASE, {10: mag(4, 6, 1)})
    s = {key(1): session(recording(rms))}
    kwargs = dict(rank="second", direction="increase", rank_filter="rank_swap")
    (alone,) = windows(s, truncate_at_rank_swap=True, **kwargs).events["m"]
    assert not alone.truncated and None not in alone.post
    back = script(60, BASE, {10: mag(4, 6, 1), 20: mag(7, 6, 1)})
    s = {key(1): session(recording(back))}
    event = windows(s, truncate_at_rank_swap=True, **kwargs).events["m"][0]
    assert event.truncated and event.rank_swap_offset == 10
    assert None not in event.post[:10] and event.post[10:] == (None,) * 6


def test_the_swap_filter_and_the_truncation_use_the_same_exchange():
    # A returns above B at trial 20: the swap filter sees that as a second/increase swap event
    # (A, the second arm, moved up and passed B), and truncation of the first event fires at
    # exactly that trial
    rms = script(60, BASE, {10: mag(4, 6, 1), 20: mag(7, 6, 1)})
    s = {key(1): session(recording(rms))}
    kwargs = dict(rank="second", direction="increase", rank_filter="rank_swap")
    assert [e.trial_idx for e in windows(s, **kwargs).events["m"]] == [10, 20]
    first = windows(s, truncate_at_rank_swap=True, **kwargs).events["m"][0]
    assert first.trial_idx + first.rank_swap_offset == 20
    # A falling further without passing B is neither a swap nor a truncation
    rms2 = script(60, BASE, {10: mag(4, 6, 1), 20: mag(3, 6, 1)})
    (event,) = windows(
        {key(1): session(recording(rms2))}, truncate_at_rank_swap=True, **kwargs
    ).events["m"]
    assert not event.truncated


def test_truncation_needs_a_strict_exchange_not_a_tie():
    rms = script(40, BASE, {10: mag(5, 2, 1), 14: mag(5, 5, 1)})  # B ties A, does not pass
    (event,) = truncated(rms).events["m"]
    assert not event.truncated


def test_truncation_with_a_tied_reference_is_flagged_not_applied():
    rms = script(40, mag(4, 4, 1), {10: mag(5, 5, 1), 14: mag(3, 5, 1)})
    (event,) = truncated(rms).events["m"]
    assert not event.truncated and event.truncation_reference_tied


def test_truncation_never_changes_which_events_are_found():
    rng = np.random.default_rng(0)
    for _ in range(5):
        rms = [mag(*rng.integers(0, 5, size=3)) for _ in range(120)]
        # hold each state for a few trials so that changes are events, not noise
        rms = [rms[t // 4] for t in range(120)]
        for config in value_change_configs():
            kwargs = dict(
                rank=config.rank, direction=config.direction, rank_filter=config.rank_filter
            )
            s = {key(1): session(recording(rms))}
            full = windows(s, **kwargs)
            cut = windows(s, truncate_at_rank_swap=True, **kwargs)
            assert [(e.session, e.trial_idx) for e in full.events.get("m", [])] == [
                (e.session, e.trial_idx) for e in cut.events.get("m", [])
            ], config.label
            assert full.counts == cut.counts


# ---------------------------------------------------------------------------- pooling


def test_problems_are_pooled_per_mouse_without_session_key_collisions():
    rms = script(30, BASE, {10: mag(5, 2, 1)})
    problem_1 = {key(1): session(recording(rms))}
    problem_2 = {key(1): session(recording(rms))}  # the same key in another problem
    w1 = get_value_change_windows({"m": problem_1}, "best", "increase", "all", pre=5, post=15)
    w2 = get_value_change_windows({"m": problem_2}, "best", "increase", "all", pre=5, post=15)
    pooled = merge_value_change_windows([w1, w2])
    assert len(pooled.events["m"]) == 2
    assert pooled.counts["n_events"] == 2
    assert pooled.counts["n_changes"] == w1.counts["n_changes"] + w2.counts["n_changes"]


def test_pooling_windows_with_different_settings_raises():
    s = {"m": {key(1): session(recording(script(30, BASE, {10: mag(5, 2, 1)})))}}
    a = get_value_change_windows(s, "best", "increase", "all", pre=5, post=15)
    b = get_value_change_windows(s, "best", "increase", "all", pre=5, post=10)
    with pytest.raises(ValueError, match="settings"):
        merge_value_change_windows([a, b])
