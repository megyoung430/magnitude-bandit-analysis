"""Tests for the symmetric, complete-window selection of value-change events.

Sessions are built from three arms A, B, C on a piecewise-constant magnitude script (see
tests/test_value_change_windows.py), so which events exist and why a window is incomplete are known.
"""

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis.value_change_symmetric import (
    ANY_CAUSE_KEYS,
    EXCLUSION_KEYS,
    select_symmetric_windows,
)
from src.behavior_analysis.value_change_windows import (
    get_value_change_windows,
    merge_value_change_windows,
    summarize_windows,
    truncate_windows_at_rank_swap,
)
from src.behavior_visualization.plot_helpers import figure_caption
from src.pipelines.value_change_analysis import run_value_change_analysis
from tests.test_value_change_windows import BASE, key, mag, recording, script, session

N = 6


def build(sessions, n=N, **kwargs):
    """Untruncated best/increase/all windows with pre = post = *n*, selected for symmetry."""
    kwargs.setdefault("rank", "best")
    kwargs.setdefault("direction", "increase")
    kwargs.setdefault("rank_filter", "all")
    full = get_value_change_windows({"m": sessions}, pre=n, post=n, **kwargs)
    return full, *select_symmetric_windows(full)


def one_event(rms, choices=None, blocks=None):
    """The single session of *rms* with its best-increase event, selected for symmetry."""
    return build({key(1): session(recording(rms, choices=choices, blocks=blocks))})


def kept_trials(windows):
    return [e.trial_idx for e in windows.events.get("m", [])]


# Best arm A rises at trial 20 of a 60-trial session: window 14..26 lies inside the session.
CLEAN = script(60, BASE, {20: mag(5, 2, 1)})


# ------------------------------------------------------------------ selection


def test_an_event_with_a_full_window_is_kept():
    full, kept, exclusions = one_event(CLEAN)
    assert kept_trials(kept) == [20]
    assert exclusions["n_events_before"] == exclusions["n_events_kept"] == 1
    assert all(exclusions[k] == 0 for k in EXCLUSION_KEYS)
    (event,) = kept.events["m"]
    assert None not in event.pre + event.post
    assert len(event.pre) == N and len(event.post) == N + 1


@pytest.mark.parametrize("block_start", [23, 19, 21])  # t = +3, t = -1, t = +1
def test_a_block_change_inside_the_window_excludes_the_event(block_start):
    n = len(CLEAN)
    blocks = [1] * block_start + [2] * (n - block_start)
    _, kept, exclusions = one_event(CLEAN, blocks=blocks)
    assert kept_trials(kept) == []
    assert exclusions["n_events_before"] == 1 and exclusions["n_events_kept"] == 0
    assert exclusions["n_excluded_block_or_third_arm_change"] == 1


def test_a_block_change_at_minus_two_excludes_the_event():
    blocks = [1] * 18 + [2] * 42  # trial 18 is t = -2; trials 19 and 20 share the new block
    _, kept, exclusions = one_event(CLEAN, blocks=blocks)
    assert kept_trials(kept) == []
    assert exclusions["n_excluded_block_or_third_arm_change"] == 1


def test_a_block_change_just_outside_the_window_keeps_the_event():
    blocks = [1] * 14 + [2] * 46  # trial 14 = t = -6 is the first trial of the new block
    _, kept, _ = one_event(CLEAN, blocks=blocks)
    assert kept_trials(kept) == [20]
    blocks = [1] * 27 + [2] * 33  # trial 27 = t = +7
    _, kept, _ = one_event(CLEAN, blocks=blocks)
    assert kept_trials(kept) == [20]


def test_a_third_arm_change_inside_the_window_excludes_the_event():
    rms = script(60, BASE, {20: mag(5, 2, 1), 23: mag(5, 1, 2)})  # lowest arm C -> B at t = +3
    _, kept, exclusions = one_event(rms)
    assert kept_trials(kept) == []
    assert exclusions["n_excluded_block_or_third_arm_change"] == 1


def test_a_session_edge_inside_the_window_excludes_the_event():
    early = script(60, BASE, {4: mag(5, 2, 1)})  # only 4 trials before t = 0
    late = script(60, BASE, {55: mag(5, 2, 1)})  # only 4 trials after t = 0
    for rms in (early, late):
        _, kept, exclusions = one_event(rms)
        assert kept_trials(kept) == []
        assert exclusions["n_excluded_session_edge"] == 1
        assert exclusions["n_excluded_block_or_third_arm_change"] == 0


def test_an_event_exactly_at_the_edge_of_the_session_is_kept():
    rms = script(60, BASE, {N: mag(5, 2, 1)})  # trials 0..5 before t = 0, 6..12 after
    _, kept, _ = one_event(rms)
    assert kept_trials(kept) == [N]
    rms = script(60, BASE, {60 - N - 1: mag(5, 2, 1)})  # last post trial is the last trial
    _, kept, _ = one_event(rms)
    assert kept_trials(kept) == [60 - N - 1]


def test_a_best_arm_change_in_the_pre_window_excludes_rather_than_truncates():
    # B overtakes A at 17 (t = -3), then rises further at 20: its pre window would be cut at 17
    rms = script(60, BASE, {17: mag(2, 4, 1), 20: mag(2, 5, 1)})
    full, kept, exclusions = one_event(rms)
    (event,) = full.events["m"]
    assert event.pre[0] is None and event.trial_idx == 20  # the regular window truncates it
    assert kept_trials(kept) == []
    assert exclusions["n_excluded_best_arm_change"] == 1


def test_a_best_arm_change_in_the_post_window_does_not_exclude_the_event():
    rms = script(60, BASE, {20: mag(5, 2, 1), 23: mag(2, 4, 1)})
    _, kept, _ = one_event(rms)
    assert kept_trials(kept) == [20]


def test_a_trial_without_a_choice_excludes_the_event_as_other():
    for missing in (16, 20, 25):  # t = -4, the event trial itself, t = +5
        choices = ["ABC"[t % 3] for t in range(60)]
        choices[missing] = ""
        _, kept, exclusions = one_event(CLEAN, choices=choices)
        assert kept_trials(kept) == [], missing
        assert exclusions["n_excluded_other"] == 1


def test_each_excluded_event_is_counted_under_one_reason_and_the_counts_add_up():
    # events at 20 (kept), 30 (best-arm change at 28), 40 (block change at +3), 55 (session edge)
    rms = script(
        60,
        BASE,
        {20: mag(5, 2, 1), 28: mag(2, 4, 1), 30: mag(2, 5, 1), 40: mag(2, 6, 1), 55: mag(2, 7, 1)},
    )
    blocks = [1] * 43 + [2] * 17
    _, kept, ex = one_event(rms, blocks=blocks)
    assert kept_trials(kept) == [20]
    assert ex["n_events_before"] == ex["n_events_kept"] + sum(ex[k] for k in EXCLUSION_KEYS)
    assert ex["n_excluded_best_arm_change"] >= 1
    assert ex["n_excluded_block_or_third_arm_change"] >= 1
    assert ex["n_excluded_session_edge"] >= 1


def test_an_event_with_several_causes_is_counted_once_by_the_most_structural_cause():
    # near the start (session edge) and with a block change at t = +2
    rms = script(60, BASE, {4: mag(5, 2, 1)})
    blocks = [1] * 6 + [2] * 54
    _, kept, ex = one_event(rms, blocks=blocks)
    assert kept_trials(kept) == []
    assert sum(ex[k] for k in EXCLUSION_KEYS) == 1
    assert ex["n_excluded_block_or_third_arm_change"] == 1


def test_the_selection_keeps_the_symmetric_flag_and_the_window():
    full, kept, _ = one_event(CLEAN)
    assert kept.symmetric and not full.symmetric
    assert kept.pre == kept.post == N
    assert kept.config == full.config


def test_selection_requires_equal_untruncated_windows():
    unequal = get_value_change_windows(
        {"m": {key(1): session(recording(CLEAN))}}, "best", "increase", "all", pre=5, post=6
    )
    with pytest.raises(ValueError, match="pre == post"):
        select_symmetric_windows(unequal)
    equal = get_value_change_windows(
        {"m": {key(1): session(recording(CLEAN))}}, "best", "increase", "all", pre=N, post=N
    )
    with pytest.raises(ValueError, match="untruncated"):
        select_symmetric_windows(truncate_windows_at_rank_swap(equal))


def test_selection_rejects_events_that_do_not_record_why_their_window_is_incomplete():
    full = get_value_change_windows(
        {"m": {key(1): session(recording(CLEAN))}}, "best", "increase", "all", pre=N, post=N
    )
    bare = full._replace(
        events={"m": [e._replace(incomplete_causes=None) for e in full.events["m"]]}
    )
    with pytest.raises(ValueError, match="incomplete_causes"):
        select_symmetric_windows(bare)


def test_a_block_that_starts_exactly_at_minus_n_keeps_the_event():
    blocks = [1] * 14 + [2] * 46  # trial 14 is t = -6, the first trial of the event's block
    _, kept, _ = one_event(CLEAN, blocks=blocks)
    assert kept_trials(kept) == [20]
    blocks = [1] * 15 + [2] * 45  # the block starts at t = -5: t = -6 is in another block
    _, kept, ex = one_event(CLEAN, blocks=blocks)
    assert kept_trials(kept) == [] and ex["n_excluded_block_or_third_arm_change"] == 1


def test_a_best_arm_that_starts_exactly_at_minus_n_keeps_the_event():
    rms = script(60, BASE, {14: mag(2, 4, 1), 20: mag(2, 5, 1)})  # B best from t = -6 on
    _, kept, _ = one_event(rms)
    assert kept_trials(kept) == [20]
    rms = script(60, BASE, {15: mag(2, 4, 1), 20: mag(2, 5, 1)})  # B best from t = -5 on
    _, kept, ex = one_event(rms)
    assert kept_trials(kept) == [] and ex["n_excluded_best_arm_change"] == 1


def test_a_best_second_tie_in_the_pre_window_counts_as_a_best_arm_change():
    rms = script(60, BASE, {17: mag(4, 4, 1), 18: mag(4, 2, 1), 20: mag(5, 2, 1)})
    full, kept, ex = one_event(rms)
    assert [e.trial_idx for e in full.events["m"]] == [20]
    assert kept_trials(kept) == []
    assert ex["n_excluded_best_arm_change"] == 1
    assert ex["n_excluded_block_or_third_arm_change"] == 0


def test_every_cause_on_the_same_side_is_counted_not_only_the_nearest():
    # t = -2 has another best arm than t = -1 (B overtakes A at trial 19); t = -4 is in block 1
    rms = script(60, BASE, {19: mag(2, 4, 1), 20: mag(2, 5, 1)})
    blocks = [1] * 17 + [2] * 43  # the event is in block 2
    _, kept, ex = one_event(rms, blocks=blocks)
    assert kept_trials(kept) == []
    assert ex["n_excluded_block_or_third_arm_change"] == 1  # priority: counted once, here
    assert ex["n_excluded_best_arm_change"] == 0
    assert ex["n_with_block_or_third_arm_change"] == 1
    assert ex["n_with_best_arm_change"] == 1  # the hidden cause is still visible
    assert ex["n_with_session_edge"] == ex["n_with_other"] == 0


def test_causes_on_both_sides_are_all_found():
    rms = script(60, BASE, {4: mag(5, 2, 1)})  # session edge at t = -6, -5
    blocks = [1] * 6 + [2] * 54  # block change at t = +2
    choices = ["ABC"[t % 3] for t in range(60)]
    choices[8] = ""  # trial 8 is t = +4, in the other block: not a no-choice cause
    _, _, ex = one_event(rms, choices=choices, blocks=blocks)
    assert ex["n_excluded_block_or_third_arm_change"] == 1
    assert sum(ex[k] for k in EXCLUSION_KEYS) == 1
    assert ex["n_with_session_edge"] == 1 and ex["n_with_block_or_third_arm_change"] == 1
    assert ex["n_with_other"] == 0
    assert set(ANY_CAUSE_KEYS) <= set(ex)


def test_symmetric_counts_say_how_many_events_were_found_and_kept():
    rms = script(60, BASE, {20: mag(5, 2, 1), 55: mag(6, 2, 1)})
    full, kept, ex = one_event(rms)
    assert full.counts["n_events"] == 2
    assert kept.counts["n_events"] == 1 and kept.counts["n_events_before_exclusion"] == 2
    assert ex["n_events_before"] == 2 and ex["n_events_kept"] == 1


def test_windows_of_different_modes_cannot_be_merged():
    full, kept, _ = one_event(CLEAN)
    with pytest.raises(ValueError, match="different settings.*True"):
        merge_value_change_windows([full, kept])


def test_symmetric_curves_equal_hand_computed_choice_probabilities():
    # Arm A is best, B second, C third throughout, so a choice of A / B / C is a "best" /
    # "second" / "third" label. m1 has events at 20 and 40 (complete) and 55 (session edge,
    # excluded: it would pull the t = 0 labels toward "third"); m2 has one event at 30.
    m1_rms = script(60, BASE, {20: mag(5, 2, 1), 40: mag(6, 2, 1), 55: mag(7, 2, 1)})
    m1_choices = ["A"] * 60
    m1_choices[20], m1_choices[22], m1_choices[55] = "B", "C", "C"
    m2_rms = script(60, BASE, {30: mag(5, 2, 1)})
    m2_choices = ["B"] * 60
    m2_choices[30] = "A"
    subjects = {
        "m1": {key(1): session(recording(m1_rms, choices=m1_choices))},
        "m2": {key(1): session(recording(m2_rms, choices=m2_choices))},
    }
    full = get_value_change_windows(subjects, "best", "increase", "all", pre=N, post=N)
    kept, ex = select_symmetric_windows(full)
    assert (ex["n_events_before"], ex["n_events_kept"]) == (4, 3)

    def expected(default, at_t0, at_t2):
        out = np.full(2 * N + 1, default)
        out[N], out[N + 2] = at_t0, at_t2
        return out

    # per mouse: m1 (events 20, 40) has best 1 except 0.5 at t0 and t2, second 0.5 at t0,
    # third 0.5 at t2; m2 (event 30) has best 0 except 1 at t0, second 1 except 0 at t0
    by_mouse = summarize_windows(kept)
    for label, want in {
        "best": expected(0.5, 0.75, 0.25),
        "second": expected(0.5, 0.25, 0.5),
        "third": expected(0.0, 0.0, 0.25),
    }.items():
        np.testing.assert_allclose(by_mouse.mean[label], want, err_msg=label)
    # per event: m1@20 (t0 B, t2 C, else A), m1@40 (all A), m2@30 (t0 A, else B)
    by_event = summarize_windows(kept, weighting="event")
    for label, want in {
        "best": expected(2 / 3, 2 / 3, 1 / 3),
        "second": expected(1 / 3, 1 / 3, 1 / 3),
        "third": expected(0.0, 0.0, 1 / 3),
    }.items():
        np.testing.assert_allclose(by_event.mean[label], want, err_msg=label)
    assert set(by_mouse.n_events_per_bin) == {3}


# ------------------------------------------------------------------ pipeline


def two_mice():
    """Each mouse: a kept best increase at 20 and an event at 55 cut by the session edge."""
    rms = script(60, BASE, {20: mag(5, 2, 1), 55: mag(6, 2, 1)})
    return {1: {"m1": {key(1): session(recording(rms))}, "m2": {key(1): session(recording(rms))}}}


@pytest.fixture
def no_figures(monkeypatch):
    monkeypatch.setattr(
        "src.pipelines.value_change_analysis.plot_value_change_choice_probs",
        lambda curves, base, **kwargs: None,
    )


def test_the_pipeline_adds_a_symmetric_mode_with_complete_windows(no_figures, tmp_path):
    result = run_value_change_analysis(two_mice(), tmp_path)
    s = result["summaries"]["best/increase/all"]
    assert set(s) == {"untruncated", "truncated_at_rank_swap", "symmetric_6"}
    sym = s["symmetric_6"]
    assert sym["n_events"] == 2 and sym["n_subjects"] == 2
    np.testing.assert_array_equal(sym["x"], np.arange(-6, 7))
    assert set(sym["n_events_per_bin"]) == {2}
    assert sym["counts"]["n_events"] == 2 and sym["counts"]["n_events_before_exclusion"] == 4
    assert s["untruncated"]["n_events"] == 4  # the edge events stay in the regular figures


def test_bin_counts_include_the_symmetric_mode(no_figures, tmp_path):
    table = run_value_change_analysis(two_mice(), tmp_path)["bin_counts_table"]
    sym = table[(table["mode"] == "symmetric_6") & (table["config"] == "best/increase/all")]
    assert sym["t"].tolist() == list(range(-6, 7))
    assert set(sym["n_events"]) == {2}


def test_exclusion_table_has_one_row_per_configuration(no_figures, tmp_path):
    table = run_value_change_analysis(two_mice(), tmp_path)["symmetric_exclusions_table"]
    assert isinstance(table, pd.DataFrame) and len(table) == 10
    row = table[table["config"] == "best/increase/all"].iloc[0]
    assert row["symmetric_window"] == 6
    assert (row["n_events_before"], row["n_events_kept"], row["n_excluded_session_edge"]) == (
        4,
        2,
        2,
    )
    assert row["n_events_before"] == row["n_events_kept"] + row["n_excluded"]
    assert set(EXCLUSION_KEYS + ANY_CAUSE_KEYS) <= set(table.columns)


def test_titles_and_file_names_say_symmetric_for_both_weightings(tmp_path):
    result = run_value_change_analysis(two_mice(), tmp_path, truncate_modes=(False,))
    stem = "Choice Probabilities Around Best-Value Increases (All Events, Symmetric ±6 Trials)"
    for folder in ("mouse-weighted", "event-weighted"):
        for ext in ("png", "pdf"):
            assert (tmp_path / folder / f"{stem}.{ext}").exists()
    assert all("Symmetric ±6 Trials" in f.name for f in result["files"] if "Symmetric" in f.name)
    assert not any("Truncated" in f.name for f in result["files"] if "Symmetric" in f.name)


def test_symmetric_title_says_complete_windows_and_counts_events_after_exclusion():
    from src.behavior_visualization.plot_value_change_choice_probs import (
        plot_value_change_panel,
    )

    full = two_mice()[1]
    windows = get_value_change_windows(full, "best", "increase", "all", pre=N, post=N)
    kept, _ = select_symmetric_windows(windows)
    for weighting in ("subject", "event"):
        ax = plot_value_change_panel(summarize_windows(kept, weighting=weighting))
        first, second = ax.get_title().split("\n")
        caption = figure_caption(ax.figure)
        assert first.endswith("(All Events)")
        assert "Window: symmetric ±6 trials, complete windows only." in caption
        assert "n=2 mice" in second and "n=2 events" in second
        if weighting == "event":
            assert "independent" in caption
    import matplotlib.pyplot as plt

    plt.close("all")


def test_symmetric_window_none_makes_no_symmetric_outputs(tmp_path):
    result = run_value_change_analysis(two_mice(), tmp_path, symmetric_window=None)
    assert not any("Symmetric" in f.name for f in result["files"])
    assert not any("Symmetric" in p.name for p in tmp_path.rglob("*"))
    assert all(
        set(s) == {"untruncated", "truncated_at_rank_swap"} for s in result["summaries"].values()
    )
    assert "symmetric_6" not in set(result["bin_counts_table"]["mode"])
    assert result["symmetric_exclusions_table"].empty


def test_a_numpy_integer_is_a_valid_symmetric_window(no_figures, tmp_path):
    result = run_value_change_analysis(two_mice(), tmp_path, symmetric_window=np.int64(6))
    assert "symmetric_6" in result["summaries"]["best/increase/all"]


@pytest.mark.parametrize("bad", [0, -3, 2.5, True, "6"])
def test_invalid_symmetric_window_raises(no_figures, tmp_path, bad):
    with pytest.raises(ValueError, match="symmetric_window"):
        run_value_change_analysis(two_mice(), tmp_path, symmetric_window=bad)


def test_another_symmetric_window_is_named_in_the_mode_and_files(no_figures, tmp_path):
    result = run_value_change_analysis(two_mice(), tmp_path, symmetric_window=3)
    assert "symmetric_3" in result["summaries"]["best/increase/all"]
    assert (result["symmetric_exclusions_table"]["symmetric_window"] == 3).all()
