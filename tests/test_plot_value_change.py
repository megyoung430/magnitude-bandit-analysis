"""Tests for the value-change plots and the run_value_change_analysis entry point."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis.value_change_windows import (
    get_value_change_windows,
    summarize_windows,
)
from src.behavior_visualization.plot_value_change_choice_probs import (
    plot_value_change_choice_probs,
    plot_value_change_panel,
)
from src.pipelines.value_change_analysis import run_value_change_analysis
from tests.test_value_change_windows import BASE, key, mag, recording, script, session


def mouse_sessions(n_sessions=2, rotate=0):
    """Sessions with one best increase (10), then A falls below B (14): a swap in its window."""
    rms = script(60, BASE, {10: mag(5, 2, 1), 14: mag(1, 2, 0)})
    return {key(i + 1): session(recording(rms)) for i in range(n_sessions)}


def two_mice(n_sessions=2):
    return {"m1": mouse_sessions(n_sessions), "m2": mouse_sessions(n_sessions)}


def curves_for(subjects, truncate=False, rank="best", direction="increase", rank_filter="all"):
    w = get_value_change_windows(
        subjects,
        rank,
        direction,
        rank_filter,
        pre=5,
        post=15,
        truncate_at_rank_swap=truncate,
    )
    return summarize_windows(w)


# --------------------------------------------------------------------------- panel


def test_panel_draws_into_the_given_axes_and_creates_no_figure():
    fig, ax = plt.subplots()
    before = plt.get_fignums()
    out = plot_value_change_panel(curves_for(two_mice()), ax=ax)
    assert out is ax and plt.get_fignums() == before
    assert {line.get_label() for line in ax.get_lines()} >= {"Best", "Second", "Third"}
    plt.close(fig)


def test_panel_without_ax_makes_its_own_figure():
    ax = plot_value_change_panel(curves_for(two_mice()))
    assert ax.figure is not None
    plt.close(ax.figure)


def test_single_subject_draws_the_mean_without_an_error_band():
    ax = plot_value_change_panel(curves_for({"m1": mouse_sessions()}))
    assert len(ax.collections) == 0  # no fill_between: the SE is NaN with one subject
    assert any(line.get_label() == "Best" for line in ax.get_lines())
    plt.close(ax.figure)


def test_error_band_is_drawn_with_two_subjects():
    ax = plot_value_change_panel(curves_for(two_mice()))
    assert len(ax.collections) > 0
    plt.close(ax.figure)


def test_swap_panel_uses_previous_best_and_second_legend_entries():
    swap = {"m1": {key(1): session(recording(script(40, BASE, {10: mag(4, 6, 1)})))}}
    ax = plot_value_change_panel(
        curves_for(swap, rank="second", direction="increase", rank_filter="rank_swap")
    )
    labels = {line.get_label() for line in ax.get_lines()}
    assert {"Prev Best", "Prev Second Best", "Third"} <= labels
    assert "Best" not in labels
    plt.close(ax.figure)


def test_individual_mice_lines_are_optional():
    plain = plot_value_change_panel(curves_for(two_mice()))
    with_mice = plot_value_change_panel(curves_for(two_mice()), show_individual_mice=True)
    assert len(with_mice.get_lines()) > len(plain.get_lines())
    plt.close("all")


def test_title_shows_the_truncated_event_count():
    title = plot_value_change_panel(curves_for(two_mice(), truncate=True)).get_title()
    assert "n=2 subjects" in title and "n=4 events" in title
    assert "4 truncated at rank swap" in title
    plain = plot_value_change_panel(curves_for(two_mice(), truncate=False)).get_title()
    assert "truncated" not in plain
    plt.close("all")


def test_wrapper_saves_png_and_pdf_and_closes(tmp_path):
    plot_value_change_choice_probs(curves_for(two_mice()), save_path=tmp_path / "sub" / "fig")
    assert (tmp_path / "sub" / "fig.png").exists() and (tmp_path / "sub" / "fig.pdf").exists()
    assert plt.get_fignums() == []


# ------------------------------------------------------------------ entry point


def expected_files(out_dir, modes=("untruncated", "truncated_at_rank_swap")):
    return {
        out_dir / f"value_change_{c}_{m}.{ext}"
        for c in (
            "best_increase_all_events",
            "best_increase_rank_preserving",
            "best_decrease_all_events",
            "best_decrease_rank_preserving",
            "best_decrease_rank_swap",
            "second_increase_all_events",
            "second_increase_rank_preserving",
            "second_increase_rank_swap",
            "second_decrease_all_events",
            "second_decrease_rank_preserving",
        )
        for m in modes
        for ext in ("png", "pdf")
    }


def swap_rich_mouse():
    """Events of every kind: best up/down, second up/down, preserving and swaps."""
    rms = script(
        200,
        BASE,
        {
            10: mag(5, 2, 1),  # best up, preserving
            30: mag(5, 3, 1),  # second up, preserving
            50: mag(5, 3, 0),  # third moves only
            60: mag(4, 3, 0),  # best down, preserving
            80: mag(4, 2, 0),  # second down, preserving
            100: mag(4, 6, 0),  # second up + swap
            125: mag(2, 6, 0),  # second (A) down, preserving
            150: mag(5, 6, 0),  # second up, preserving (A 2 -> 5)
            170: mag(5, 3, 0),  # best down + swap (B drops below A)
        },
    )
    return {key(1): session(recording(rms))}


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    """One complete run (both truncation modes, one mouse): figures are slow, so share it."""
    out_dir = tmp_path_factory.mktemp("value_change")
    return out_dir, run_value_change_analysis({1: {"m1": swap_rich_mouse()}}, out_dir)


@pytest.fixture
def no_figures(monkeypatch):
    """Skip drawing for tests that only look at the returned summaries."""
    monkeypatch.setattr(
        "src.pipelines.value_change_analysis.plot_value_change_choice_probs",
        lambda curves, base, **kwargs: None,
    )


def test_entry_writes_every_figure_and_the_csv(full_run):
    out_dir, result = full_run
    written = set(out_dir.iterdir())
    assert expected_files(out_dir) <= written
    assert set(result["files"]) == expected_files(out_dir)
    assert plt.get_fignums() == []
    table = pd.read_csv(out_dir / "value_change_contamination.csv")
    assert len(table) == 10
    assert {"config", "n_events", "n_contaminated", "frac_contaminated", "n_rank_swap"} <= set(
        table.columns
    )
    assert set(result["contamination"]) == set(table["config"])
    assert result["contamination_csv"] == out_dir / "value_change_contamination.csv"


def test_entry_summaries_report_counts_per_config_and_mode(full_run):
    _, result = full_run
    s = result["summaries"]["best/decrease/rank_swap"]
    assert set(s) == {"untruncated", "truncated_at_rank_swap"}
    assert s["untruncated"]["n_events"] == s["truncated_at_rank_swap"]["n_events"] == 1
    assert s["untruncated"]["counts"]["n_dropped_no_post_room"] == 0
    assert s["untruncated"]["n_truncated"] == 0
    assert result["summaries"]["second/increase/rank_swap"]["untruncated"]["n_events"] == 1


def test_entry_curves_match_a_hand_computation(full_run):
    # best/increase/all events: trial 10 (A 4 -> 5) and trial 100 (B 2 -> 6 passes A).
    # The animal cycles A, B, C by trial number: trial 10 -> B (second), 99 -> A, 9 -> A,
    # 100 -> B (now the best arm).
    _, result = full_run
    s = result["summaries"]["best/increase/all"]["untruncated"]
    assert s["n_events"] == 2
    assert list(s["x"]) == list(range(-5, 16))
    t0, tm1 = 5, 4
    assert [s["mean"][r][t0] for r in ("best", "second", "third")] == [0.5, 0.5, 0.0]
    assert [s["mean"][r][tm1] for r in ("best", "second", "third")] == [1.0, 0.0, 0.0]
    assert np.isnan(s["se"]["best"]).all()  # one mouse: no spread across mice


def test_entry_truncate_modes_select_the_figures(tmp_path):
    run_value_change_analysis({1: {"m1": swap_rich_mouse()}}, tmp_path, truncate_modes=(True,))
    names = {p.name for p in tmp_path.glob("*.png")}
    assert len(names) == 10 and all("truncated_at_rank_swap" in n for n in names)


def test_entry_pools_problems_per_mouse(tmp_path, no_figures):
    one = run_value_change_analysis({1: {"m1": swap_rich_mouse()}}, tmp_path / "one")
    both = run_value_change_analysis(
        {1: {"m1": swap_rich_mouse()}, 2: {"m1": swap_rich_mouse()}}, tmp_path / "both"
    )
    label = "best/decrease/all"
    n1 = one["summaries"][label]["untruncated"]
    n2 = both["summaries"][label]["untruncated"]
    assert n2["n_events"] == 2 * n1["n_events"]
    assert n2["n_subjects"] == 1  # one mouse across two problems


def test_entry_records_configs_without_events_instead_of_dropping_them(tmp_path):
    quiet = {key(1): session(recording(script(60, BASE, {})))}
    result = run_value_change_analysis({1: {"m1": quiet}}, tmp_path)
    s = result["summaries"]["best/increase/all"]["untruncated"]
    assert s["n_events"] == 0 and s["plotted"] is False
    assert not list(tmp_path.glob("*.png"))
    table = pd.read_csv(tmp_path / "value_change_contamination.csv")
    assert (table["n_events"] == 0).all()
    assert table["frac_contaminated"].isna().all()


def test_entry_validates_its_arguments(tmp_path, no_figures):
    good = {1: {"m1": swap_rich_mouse()}}
    with pytest.raises(ValueError, match="no problems"):
        run_value_change_analysis({}, tmp_path)
    with pytest.raises(ValueError, match="truncate_modes"):
        run_value_change_analysis(good, tmp_path, truncate_modes=())
    with pytest.raises(ValueError, match="pre"):
        run_value_change_analysis(good, tmp_path, pre=0)
