"""Reversals of unknown kind: block changes in sessions that log no good/bad counters."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.behavior_analysis.get_choice_probs_around_good_reversals import (
    get_choice_probs_around_good_reversals,
)
from src.behavior_analysis.get_good_reversal_info import get_good_reversal_info
from src.behavior_analysis.get_total_reversals import (
    get_all_reversal_indices,
    get_reversal_indices_by_kind,
    get_unclassified_reversal_indices,
)
from src.behavior_visualization.plot_choice_probs_around_good_reversals import (
    plot_choice_probs_around_good_reversals,
)


def blocks_only(blocks):
    return {
        "trial": list(range(1, len(blocks) + 1)),
        "blocks": blocks,
        "reward_magnitudes_by_tower": {},
        "choices_by_tower": {},
        "choices_by_rank": {},
    }


def with_counters(blocks, good):
    return {**blocks_only(blocks), "has_good": True, "good_reversals": good}


def test_blocks_only_session_reports_every_block_change():
    data = {"m": {"ses-1": blocks_only([1, 1, 2, 2, 2, 3])}}
    assert get_unclassified_reversal_indices(data) == {"m": [2, 5]}
    assert get_all_reversal_indices(data)[:2] == ({"m": []}, {"m": []})  # nothing is good or bad


def test_sessions_with_counters_contribute_no_unclassified_reversals():
    data = {"m": {"ses-1": with_counters([1, 1, 2, 2], [0, 0, 1, 1])}}
    assert get_unclassified_reversal_indices(data) == {"m": []}


def test_subjects_are_classified_per_session_format_on_the_merged_trial_axis():
    data = {
        "m": {
            "ses-2": with_counters([1, 1, 2, 2], [0, 0, 1, 1]),
            "ses-1": with_counters([1, 1, 2], [0, 0, 1]),
        },
        "n": {"ses-1": blocks_only([1, 2])},
    }
    kinds = get_reversal_indices_by_kind(data)
    assert kinds["good"] == {"m": [2, 5], "n": []}
    assert kinds["bad"] == {"m": [], "n": []}
    assert kinds["unclassified"] == {"m": [], "n": [1]}


def test_a_subject_mixing_counter_and_blocks_only_sessions_raises():
    data = {"m": {"ses-1": with_counters([1, 1, 2, 2], [0, 0, 1, 1]), "ses-2": blocks_only([1, 2])}}
    with pytest.raises(ValueError, match="m: sessions mix the good-counter format"):
        get_reversal_indices_by_kind(data)


def test_sessions_are_ordered_by_number_not_by_name():
    data = {"m": {"ses-10": blocks_only([1, 2]), "ses-2": blocks_only([1, 1])}}
    # ses-2 comes first (2 trials), so ses-10's change sits at merged index 3
    assert get_unclassified_reversal_indices(data) == {"m": [3]}


# --- per-session length check -----------------------------------------------------------


def test_blocks_longer_than_trial_is_an_error_naming_the_session():
    session = {**blocks_only([1, 1, 2, 2]), "trial": [1, 2]}
    with pytest.raises(ValueError, match="m ses-1: blocks has 4 entries but trial has 2"):
        get_unclassified_reversal_indices({"m": {"ses-1": session}})


def test_blocks_shorter_than_trial_is_an_error():
    session = {**blocks_only([1, 1]), "trial": [1, 2, 3, 4]}
    with pytest.raises(ValueError, match="blocks has 2 entries but trial has 4"):
        get_unclassified_reversal_indices({"m": {"ses-1": session}})


def test_missing_blocks_with_trials_is_an_error():
    session = {"trial": [1, 2, 3]}
    with pytest.raises(ValueError, match="blocks has 0 entries but trial has 3"):
        get_unclassified_reversal_indices({"m": {"ses-1": session}})


def test_an_earlier_session_with_blocks_but_no_trial_is_an_error():
    # its blocks would shift every later index on the merged axis
    data = {"m": {"ses-1": {"blocks": [1, 1, 2]}, "ses-2": blocks_only([1, 2])}}
    with pytest.raises(ValueError, match="m ses-1: blocks has 3 entries but trial has 0"):
        get_unclassified_reversal_indices(data)


def test_a_session_with_neither_blocks_nor_trial_is_skipped():
    data = {"m": {"ses-1": {}, "ses-2": blocks_only([1, 2])}}
    assert get_unclassified_reversal_indices(data) == {"m": [1]}


# --- the script-style call path on blocks-only data -------------------------------------


def blocks_only_subject(block_lengths):
    """A blocks-only session (no good/bad counters) whose best tower rotates every block."""
    blocks = [b for b, n in enumerate(block_lengths) for _ in range(n)]
    names = ["A", "B", "C"]
    rewards = {t: [[4, 2, 1][(i - b) % 3] for b in blocks] for i, t in enumerate(names)}
    choices = {t: [int(i == b % 3) for b in blocks] for i, t in enumerate(names)}
    session = {
        **blocks_only([b + 1 for b in blocks]),
        "reward_magnitudes_by_tower": rewards,
        "choices_by_tower": choices,
        "choices_by_rank": {"best": [1] * len(blocks)},
    }
    return {"ses-1": session}


def plotted_total_line(monkeypatch, data, **indices):
    """Draw the total-mode cumulative axis the way the analysis scripts do; return its line."""
    windows = get_good_reversal_info(data, pre=3, post=30)
    x, _, across = get_choice_probs_around_good_reversals(windows, pre=3, post=30)
    figures = []
    monkeypatch.setattr(plt, "show", lambda: figures.append(plt.gcf()))
    plot_choice_probs_around_good_reversals(
        x,
        across,
        use_total=True,
        windows_for_cumulative_axis=windows,
        all_good_idx=indices["good"],
        all_bad_idx=indices["bad"],
        all_unclassified_idx=indices["unclassified"],
    )
    line = figures[-1].axes[1].lines[0].get_ydata()
    plt.close("all")
    return line[~np.isnan(line)]


def test_script_style_total_plot_of_blocks_only_data_shows_removal(monkeypatch):
    data = {"m": blocks_only_subject([10, 15, 15, 20]), "n": blocks_only_subject([12, 12, 12, 12])}
    line = plotted_total_line(monkeypatch, data, **get_reversal_indices_by_kind(data))
    assert line.max() > 0.1  # blocks-only data does not read flat 0


def test_without_the_unclassified_indices_blocks_only_total_reads_flat_zero(monkeypatch):
    # what the scripts did before they passed the unclassified indices
    data = {"m": blocks_only_subject([10, 15, 15, 20]), "n": blocks_only_subject([12, 12, 12, 12])}
    kinds = get_reversal_indices_by_kind(data)
    kinds["unclassified"] = {subject: [] for subject in data}
    assert plotted_total_line(monkeypatch, data, **kinds).max() == 0


def test_cumulative_plot_requires_the_unclassified_indices():
    with pytest.raises(ValueError, match="all_unclassified_idx"):
        plot_choice_probs_around_good_reversals(
            np.arange(-3, 5),
            {},
            windows_for_cumulative_axis={"m": []},
            all_good_idx={"m": []},
            all_bad_idx={"m": []},
            all_unclassified_idx=None,
        )
