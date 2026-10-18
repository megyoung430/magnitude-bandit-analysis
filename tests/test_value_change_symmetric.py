"""Tests for the symmetric, complete-window selection of value-change events.

Sessions are built from three arms A, B, C on a piecewise-constant magnitude script (see
tests/test_value_change_windows.py), so which events exist and why a window is incomplete are known.
"""

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis.value_change_symmetric import (
    EXCLUSION_KEYS,
    select_symmetric_windows,
)
from src.behavior_analysis.value_change_windows import (
    get_value_change_windows,
    summarize_windows,
    truncate_windows_at_rank_swap,
)
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
    assert kept.config == full.config and kept.counts == full.counts


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


def test_selection_rejects_events_that_do_not_record_why_their_window_was_cut():
    full = get_value_change_windows(
        {"m": {key(1): session(recording(CLEAN))}}, "best", "increase", "all", pre=N, post=N
    )
    bare = full._replace(
        events={"m": [e._replace(pre_cut=None, post_cut=None) for e in full.events["m"]]}
    )
    with pytest.raises(ValueError, match="cut"):
        select_symmetric_windows(bare)


def test_curves_of_kept_events_equal_the_regular_full_window_curves_of_the_same_events():
    # two mice, several events each; some excluded
    rms = script(
        120,
        BASE,
        {20: mag(5, 2, 1), 40: mag(2, 4, 1), 43: mag(2, 5, 1), 60: mag(2, 6, 1), 80: mag(2, 7, 1)},
    )
    subjects = {
        "m1": {key(1): session(recording(rms))},
        "m2": {key(1): session(recording(rms, choices=["A"] * 120))},
    }
    full = get_value_change_windows(subjects, "best", "increase", "all", pre=N, post=N)
    kept, _ = select_symmetric_windows(full)
    keep = {(s, e.trial_idx) for s, evs in kept.events.items() for e in evs}
    assert 0 < len(keep) < sum(map(len, full.events.values()))
    regular = full._replace(
        events={s: [e for e in evs if (s, e.trial_idx) in keep] for s, evs in full.events.items()}
    )
    for weighting in ("subject", "event"):
        got, want = (
            summarize_windows(kept, weighting=weighting),
            summarize_windows(regular, weighting=weighting),
        )
        assert got.num_events == want.num_events == len(keep)
        np.testing.assert_array_equal(got.x, np.arange(-N, N + 1))
        for label in want.labels:
            np.testing.assert_array_equal(got.mean[label], want.mean[label])
            np.testing.assert_array_equal(got.se[label], want.se[label])
    # complete windows: every time point has every event
    assert set(summarize_windows(kept).n_events_per_bin) == {len(keep)}


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
    assert set(EXCLUSION_KEYS) <= set(table.columns)


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
        title = plot_value_change_panel(summarize_windows(kept, weighting=weighting)).get_title()
        first, second, *rest = title.split("\n")
        assert first.endswith("(All Events, Symmetric ±6 Trials)")
        assert "complete windows only" in second
        assert "n=2 mice" in second and "n=2 events" in second
        if weighting == "event":
            assert any("independent" in line for line in rest)
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


@pytest.mark.parametrize("bad", [0, -3, 2.5, True, "6"])
def test_invalid_symmetric_window_raises(no_figures, tmp_path, bad):
    with pytest.raises(ValueError, match="symmetric_window"):
        run_value_change_analysis(two_mice(), tmp_path, symmetric_window=bad)


def test_another_symmetric_window_is_named_in_the_mode_and_files(no_figures, tmp_path):
    result = run_value_change_analysis(two_mice(), tmp_path, symmetric_window=3)
    assert "symmetric_3" in result["summaries"]["best/increase/all"]
    assert (result["symmetric_exclusions_table"]["symmetric_window"] == 3).all()
