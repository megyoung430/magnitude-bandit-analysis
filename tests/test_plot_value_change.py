"""Tests for the value-change plots and the run_value_change_analysis entry point."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PolyCollection

from src.behavior_analysis.value_change_windows import (
    ValueChangeConfig,
    describe_value_change,
    get_value_change_windows,
    summarize_windows,
)
from src.behavior_visualization.plot_helpers import figure_caption
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


def always_a_mouse(n_sessions):
    """Like mouse_sessions, but the animal always chooses A (the best arm after the increase)."""
    rms = script(60, BASE, {10: mag(5, 2, 1), 14: mag(1, 2, 0)})
    return {key(i + 1): session(recording(rms, choices=["A"] * 60)) for i in range(n_sessions)}


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


def test_title_states_what_is_aligned_and_how_many_mice_and_events():
    ax = plot_value_change_panel(curves_for(two_mice(), truncate=True))
    assert ax.get_title() == (
        "Choice Probabilities Around Best-Value Increases (All Events)\n"
        "mean ± se across mice, n=2 mice, n=4 events"
    )
    assert figure_caption(ax.figure) == (
        "Window: truncated at the first rank swap (4 events truncated)."
    )
    plain = plot_value_change_panel(curves_for(two_mice(), truncate=False))
    assert figure_caption(plain.figure) == "Window: full window."
    plt.close("all")


def test_title_says_mean_only_for_one_mouse_matching_the_drawn_band():
    one = plot_value_change_panel(curves_for({"m1": mouse_sessions()}, truncate=True))
    two = plot_value_change_panel(curves_for(two_mice(), truncate=True))
    assert one.get_title().splitlines()[1] == "mean across mice, n=1 mouse, n=2 events"
    assert two.get_title().splitlines()[1].startswith("mean ± se across mice, n=2 mice")
    bands = [[c for c in ax.collections if isinstance(c, PolyCollection)] for ax in (one, two)]
    assert not bands[0] and bands[1]
    plt.close("all")


def test_event_weighted_title_says_across_events_and_the_caption_has_the_caveat():
    w = get_value_change_windows(two_mice(), "best", "increase", "all", pre=5, post=15)
    ax = plot_value_change_panel(summarize_windows(w, weighting="event"))
    assert ax.get_title().splitlines()[1] == "mean ± se across events, n=4 events, n=2 mice"
    caption = figure_caption(ax.figure).splitlines()
    assert caption[1:] == [
        "Each event is weighted equally.",
        "se treats events as independent, so it is narrower than the true uncertainty.",
    ]
    plt.close("all")


def test_the_heading_has_the_filter_and_the_caption_the_mode_for_a_swap_configuration():
    swap = {"m1": {key(1): session(recording(script(40, BASE, {10: mag(4, 6, 1)})))}}
    curves = curves_for(
        swap, rank="second", direction="increase", rank_filter="rank_swap", truncate=True
    )
    ax = plot_value_change_panel(curves)
    lines = ax.get_title().split("\n")
    assert len(lines) == 2
    assert lines[0] == "Choice Probabilities Around Second-Best-Value Increases (Rank Swap)"
    assert figure_caption(ax.figure).startswith("Window: truncated at the first rank swap")
    plt.close("all")


def test_description_names_every_rank_direction_filter_and_mode():
    cfg = ValueChangeConfig
    assert describe_value_change(cfg("best", "decrease", "rank_swap"), True) == (
        "Best-Value Decreases (Rank Swap, Truncated at Swap)"
    )
    assert describe_value_change(cfg("second", "decrease", "rank_preserving"), False) == (
        "Second-Best-Value Decreases (Rank-Preserving, Full Window)"
    )
    assert describe_value_change(cfg("best", "increase", "all"), False) == (
        "Best-Value Increases (All Events, Full Window)"
    )


def test_x_ticks_are_integers_and_the_legend_is_outside_the_axes():
    ax = plot_value_change_panel(curves_for(two_mice()))
    ax.figure.canvas.draw()
    ticks = [t for t in ax.get_xticks() if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
    assert all(float(t).is_integer() for t in ticks)
    legend_left = ax.get_legend().get_window_extent().x0
    assert legend_left > ax.get_window_extent().x1
    plt.close("all")


def test_wrapper_saves_png_and_pdf_and_closes(tmp_path):
    plot_value_change_choice_probs(curves_for(two_mice()), save_path=tmp_path / "sub" / "fig")
    assert (tmp_path / "sub" / "fig.png").exists() and (tmp_path / "sub" / "fig.pdf").exists()
    assert plt.get_fignums() == []


# ------------------------------------------------------------------ entry point


def expected_files(out_dir, modes=("Full Window", "Truncated at Swap", "-2 to +4 Trials")):
    return {
        out_dir / f"Choice Probabilities Around {c} ({f}, {m}).{ext}"
        for c, f in (
            ("Best-Value Increases", "All Events"),
            ("Best-Value Increases", "Rank-Preserving"),
            ("Best-Value Decreases", "All Events"),
            ("Best-Value Decreases", "Rank-Preserving"),
            ("Best-Value Decreases", "Rank Swap"),
            ("Second-Best-Value Increases", "All Events"),
            ("Second-Best-Value Increases", "Rank-Preserving"),
            ("Second-Best-Value Increases", "Rank Swap"),
            ("Second-Best-Value Decreases", "All Events"),
            ("Second-Best-Value Decreases", "Rank-Preserving"),
        )
        for m in modes
        for ext in ("pdf", "png", "svg", "caption.txt")
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


def test_entry_writes_only_figures_to_the_folder(full_run):
    out_dir, result = full_run
    by_mouse, by_event = out_dir / "mouse-weighted", out_dir / "event-weighted"
    assert set(out_dir.iterdir()) == {by_mouse, by_event}  # nothing at the top level
    assert set(by_mouse.iterdir()) == expected_files(by_mouse)
    assert set(by_event.iterdir()) == expected_files(by_event)
    figures = {p for p in expected_files(by_mouse) | expected_files(by_event) if p.suffix != ".txt"}
    assert set(result["files"]) == figures
    assert plt.get_fignums() == []


def test_entry_returns_event_weighted_curves(full_run):
    _, result = full_run
    s = result["summaries"]["best/increase/all"]["untruncated"]
    # one mouse: weighting by event or by mouse gives the same mean
    for label, mean in s["event_weighted"]["mean"].items():
        np.testing.assert_allclose(mean, s["mean"][label])


def test_entry_event_weighted_summary_weights_events(no_figures, tmp_path):
    # m1 has one session of events, m2 two: the weightings must differ where they disagree
    data = {1: {"m1": mouse_sessions(1), "m2": always_a_mouse(2)}}
    s = run_value_change_analysis(data, tmp_path)["summaries"]["best/increase/all"]
    s = s["untruncated"]
    w = get_value_change_windows(data[1], "best", "increase", "all", pre=5, post=15)
    by_event = summarize_windows(w, weighting="event")
    assert by_event.weighting == "event"
    for label in w.labels:
        np.testing.assert_allclose(s["event_weighted"]["mean"][label], by_event.mean[label])
        np.testing.assert_allclose(s["mean"][label], summarize_windows(w).mean[label])
    assert not np.allclose(s["event_weighted"]["mean"]["best"], s["mean"]["best"], equal_nan=True)


def test_entry_returns_the_contamination_table(full_run):
    _, result = full_run
    table = result["contamination_table"]
    assert isinstance(table, pd.DataFrame)
    assert len(table) == 10
    assert {"config", "n_events", "n_contaminated", "frac_contaminated", "n_rank_swap"} <= set(
        table.columns
    )
    assert set(result["contamination"]) == set(table["config"])
    assert "contamination_csv" not in result


def test_entry_summaries_report_counts_per_config_and_mode(full_run):
    _, result = full_run
    s = result["summaries"]["best/decrease/rank_swap"]
    assert set(s) == {"untruncated", "truncated_at_rank_swap", "complete_pre2_post4"}
    assert s["untruncated"]["n_events"] == s["truncated_at_rank_swap"]["n_events"] == 1
    assert s["untruncated"]["counts"]["n_third_arm_changed"] == 0
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


def test_entry_returns_per_bin_counts_that_show_the_thinning(full_run):
    _, result = full_run
    table = result["bin_counts_table"]
    assert "bin_counts_csv" not in result
    assert set(table.columns) == {"config", "mode", "t", "n_events", "n_subjects"}
    best_up = table[(table["config"] == "best/increase/all") & (table["mode"] == "untruncated")]
    assert list(best_up["t"]) == list(range(-5, 16)) and (best_up["n_events"] == 2).all()
    truncated = table[
        (table["config"] == "best/decrease/rank_swap") & (table["mode"] == "truncated_at_rank_swap")
    ]
    assert (truncated["n_events"] == 1).all()  # its single swap is the t = 0 exchange itself


def test_entry_refuses_a_contamination_table_that_disagrees_with_the_counts(
    tmp_path, no_figures, monkeypatch
):
    monkeypatch.setattr(
        "src.pipelines.value_change_analysis.contamination_summary",
        lambda events: {"n_events": -1},
    )
    with pytest.raises(ValueError, match="summarised"):
        run_value_change_analysis({1: {"m1": swap_rich_mouse()}}, tmp_path)


def test_entry_truncate_modes_select_the_figures(tmp_path):
    run_value_change_analysis(
        {1: {"m1": swap_rich_mouse()}}, tmp_path, truncate_modes=(True,), complete_window=None
    )
    for folder in ("mouse-weighted", "event-weighted"):
        names = {p.name for p in (tmp_path / folder).glob("*.png")}
        assert len(names) == 10 and all("Truncated at Swap" in n for n in names)


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
    table = result["contamination_table"]
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
