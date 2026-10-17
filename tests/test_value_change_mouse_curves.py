"""Tests for the per-mouse value-change curve table (the numbers behind the mouse-weighted plot).

Two synthetic mice with unequal event counts and windows of pre=2, post=2 (t-2 .. t+2):

====== ============================== ===================================
mouse  event windows (t-2 .. t+2)      per-mouse probability by hand
====== ============================== ===================================
a      [pb, ps, pb, pb, -]            t-2: pb 1; t-1: pb/ps 1/2 each;
       [-, pb, ps, third, -]          t0: pb/ps 1/2 each; t+1: pb/third
b      [third, third, pb, -, -]       1/2 each; t+2: no event has a trial
====== ============================== ===================================
"""

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis.value_change_mouse_curves import (
    MEAN_ROW,
    N_EVENTS_LABEL,
    mouse_curves_table,
    t_column_name,
)
from src.behavior_analysis.value_change_windows import (
    ValueChangeConfig,
    ValueChangeEvent,
    ValueChangeWindows,
    _subject_means,
    summarize_windows,
)

PRE, POST = 2, 2
LABELS = ("prev_best", "prev_second", "third")
CONFIG = ValueChangeConfig("best", "decrease", "rank_swap")
T_COLUMNS = ["t-2", "t-1", "t+0", "t+1", "t+2"]


def event(subject, window, trial_idx=10):
    """An event from a window written as t-2 .. t+2 with ``None`` for a missing trial."""
    short = {"pb": "prev_best", "ps": "prev_second", "third": "third", None: None}
    labels = [short[w] for w in window]
    return ValueChangeEvent(
        subject, "ses-1_date-20260101", trial_idx, tuple(labels[:PRE]), tuple(labels[PRE:]),
        contaminated=False, rank_swap_offset=None, truncation_reference_tied=False,
    )  # fmt: skip


@pytest.fixture
def windows():
    events = {
        "a": [
            event("a", ["pb", "ps", "pb", "pb", None]),
            event("a", [None, "pb", "ps", "third", None], trial_idx=30),
        ],
        "b": [event("b", ["third", "third", "pb", None, None])],
    }
    return ValueChangeWindows(CONFIG, PRE, POST, True, events, {"n_events": 3})


@pytest.fixture
def curves(windows):
    return summarize_windows(windows)


@pytest.fixture
def table(curves):
    return mouse_curves_table(curves, scope="problem-01")


def row(table, mouse, label):
    found = table[(table["mouse"] == mouse) & (table["label"] == label)]
    assert len(found) == 1, f"expected one row for {mouse}/{label}, got {len(found)}"
    return found.iloc[0]


def values(table, mouse, label):
    return row(table, mouse, label)[T_COLUMNS].to_numpy(dtype=float)


def test_columns_are_the_metadata_then_one_column_per_t(table):
    assert list(table.columns) == [
        "scope", "config", "mode", "mouse", "n_events", "label", *T_COLUMNS
    ]  # fmt: skip


def test_t_columns_are_signed_and_follow_the_window():
    assert [t_column_name(t) for t in (-5, -1, 0, 1, 15)] == ["t-5", "t-1", "t+0", "t+1", "t+15"]


def test_metadata_describes_the_scope_configuration_and_mode(table):
    assert set(table["scope"]) == {"problem-01"}
    assert set(table["config"]) == {"best/decrease/rank_swap"}
    assert set(table["mode"]) == {"truncated_at_rank_swap"}


def test_untruncated_curves_are_labelled_untruncated(windows):
    curves = summarize_windows(windows._replace(truncate_at_rank_swap=False))
    assert set(mouse_curves_table(curves, scope="s")["mode"]) == {"untruncated"}


def test_each_mouse_has_its_total_events_on_every_row(table):
    assert set(table.loc[table["mouse"] == "a", "n_events"]) == {2}
    assert set(table.loc[table["mouse"] == "b", "n_events"]) == {1}


def test_per_mouse_probabilities_match_the_hand_computed_values(table):
    nan = np.nan
    np.testing.assert_allclose(values(table, "a", "prev_best"), [1, 0.5, 0.5, 0.5, nan])
    np.testing.assert_allclose(values(table, "a", "prev_second"), [0, 0.5, 0.5, 0, nan])
    np.testing.assert_allclose(values(table, "a", "third"), [0, 0, 0, 0.5, nan])
    np.testing.assert_allclose(values(table, "b", "third"), [1, 1, 0, nan, nan])


def test_per_mouse_rows_equal_the_aggregation_the_figure_uses(windows, table):
    for mouse, events in windows.events.items():
        expected, _ = _subject_means(events, PRE, POST, LABELS)
        for label in LABELS:
            np.testing.assert_array_equal(values(table, mouse, label), expected[label])


def test_n_events_at_t_counts_the_events_with_a_trial_at_each_t(table):
    np.testing.assert_array_equal(values(table, "a", N_EVENTS_LABEL), [1, 2, 2, 2, 0])
    np.testing.assert_array_equal(values(table, "b", N_EVENTS_LABEL), [1, 1, 1, 0, 0])


def test_mean_across_mice_equals_the_plotted_mouse_weighted_mean(curves, table):
    for label in LABELS:
        np.testing.assert_array_equal(values(table, MEAN_ROW, label), curves.mean[label])
    assert row(table, MEAN_ROW, "third")["n_events"] == 3


def test_mean_across_mice_averages_the_mouse_rows_not_the_events(table):
    # t0 prev_best: mouse a 0.5, mouse b 1 -> 0.75; pooling the 3 events would give 2/3.
    assert values(table, MEAN_ROW, "prev_best")[2] == pytest.approx(0.75)
    # t-1 prev_best: mouse a 0.5, mouse b 0 -> 0.25; pooling the 3 events would give 1/3.
    assert values(table, MEAN_ROW, "prev_best")[1] == pytest.approx(0.25)


def test_mean_row_n_events_at_t_sums_over_mice(curves, table):
    np.testing.assert_array_equal(values(table, MEAN_ROW, N_EVENTS_LABEL), curves.n_events_per_bin)


def test_rows_are_grouped_by_mouse_with_the_mean_last(table):
    assert list(table["mouse"].unique()) == ["a", "b", MEAN_ROW]
    labels = [*LABELS, N_EVENTS_LABEL]
    assert list(table.loc[table["mouse"] == "a", "label"]) == labels
    assert list(table.loc[table["mouse"] == MEAN_ROW, "label"]) == labels


def test_curves_expose_the_per_mouse_valid_event_counts(curves):
    np.testing.assert_array_equal(curves.per_subject_n_events["a"], [1, 2, 2, 2, 0])
    np.testing.assert_array_equal(curves.per_subject_n_events["b"], [1, 1, 1, 0, 0])
    assert curves.per_subject_total_events == {"a": 2, "b": 1}


def test_event_weighted_curves_are_rejected(windows):
    by_event = summarize_windows(windows, weighting="event")
    with pytest.raises(ValueError, match="weighting"):
        mouse_curves_table(by_event, scope="s")


def test_curves_without_events_give_an_empty_table_with_the_columns():
    empty = summarize_windows(ValueChangeWindows(CONFIG, PRE, POST, False, {}, {}))
    table = mouse_curves_table(empty, scope="s")
    assert table.empty and list(table.columns[-5:]) == T_COLUMNS


def test_tables_of_several_scopes_concatenate(curves):
    both = pd.concat([mouse_curves_table(curves, scope=s) for s in ("p1", "p2")])
    assert set(both["scope"]) == {"p1", "p2"}
