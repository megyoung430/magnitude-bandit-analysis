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

RANK_NAMES = ("best", "second", "third")


def rank_of(magnitudes, arm):
    """Rank label of *arm*; tied arms share the better label."""
    return RANK_NAMES[sum(v > magnitudes[arm] for v in magnitudes.values())]


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
        "chosen_rank": [rank_of(rm, c) for rm, c in zip(rms, choices, strict=True)],
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


def test_window_bounds_keep_full_windows_and_count_the_dropped_ones():
    pre, post, n = 5, 15, 30
    # events at 4 (no pre room), 5 (exactly enough), 14 (last with room), 15 (no post room)
    rms = script(
        n, mag(1, 0, 0), {4: mag(2, 0, 0), 5: mag(3, 0, 0), 14: mag(4, 0, 0), 15: mag(5, 0, 0)}
    )
    w = windows({key(1): session(recording(rms))}, pre=pre, post=post)
    assert [e.trial_idx for e in w.events["m"]] == [5, 14]
    assert w.counts["n_dropped_no_pre_room"] == 1
    assert w.counts["n_dropped_no_post_room"] == 1
    assert w.counts["n_events"] == 2


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
            c["n_filtered_out"]
            + c["n_tie_skipped"]
            + c["n_dropped_no_pre_room"]
            + c["n_dropped_no_post_room"]
            + c["n_events"]
        ), config.label


def test_changes_at_a_block_boundary_are_not_events_and_cut_the_window():
    rms = script(40, BASE, {10: mag(5, 2, 1), 20: mag(1, 6, 2)})
    blocks = [1] * 20 + [2] * 20
    w = windows({key(1): session(recording(rms, blocks=blocks))})
    (event,) = w.events["m"]  # the change at 20 is a reversal (new block), not an event
    assert event.trial_idx == 10
    assert event.post[:10] != (None,) * 10
    assert event.post[10:] == (None,) * 6  # trials 20.. belong to the next block


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
    w = windows(sessions)
    assert w.events == {}
    assert w.counts["n_dropped_no_pre_room"] == 1


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


def test_a_tie_between_the_other_arms_does_not_block_the_classification():
    rms = script(40, mag(4, 1, 1), {10: mag(5, 1, 1)})
    s = {key(1): session(recording(rms))}
    w = windows(s, rank="best", direction="increase", rank_filter="rank_preserving")
    assert len(w.events["m"]) == 1  # A is clearly the best before and after; B = C is irrelevant


# --------------------------------------------------------------------- label sets


def test_rank_swap_windows_use_previous_best_and_second_labels():
    s = {key(1): session(recording(swap_script()))}
    w = windows(s, rank="second", direction="increase", rank_filter="rank_swap")
    (event,) = w.events["m"]
    assert set(event.pre + event.post) <= {"prev_best", "prev_second", "third"}
    assert event.pre[0] == "third"  # trial 5 is a choice of C, the arm that was third


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
    assert across["mean"]["prev_best"].sum() > 0  # these were silently 0 when inferred from "third"
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
    rms = script(40, BASE, {10: mag(5, 2, 1), 14: mag(5, 2, 3)})  # only the third arm moves
    (event,) = truncated(rms).events["m"]
    assert not event.truncated and None not in event.post


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
