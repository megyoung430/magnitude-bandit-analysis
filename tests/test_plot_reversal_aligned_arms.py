"""Tests for the reversal-aligned arm figure: colours, twin axis, SE bands, title and saving."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_hex

from src.behavior_analysis.reversal_aligned_arms import compute_reversal_aligned_arms
from src.behavior_visualization.plot_helpers import figure_caption
from src.behavior_visualization.plot_reversal_aligned_arms import (
    ROLE_COLORS,
    figure_heading,
    figure_stem,
    plot_reversal_aligned_arms,
    save_reversal_aligned_arms,
)
from tests.plot_capture import capture_figures
from tests.reversal_arm_fixtures import make_reversal, stamp_windows


def result_for(scheme, subjects=2, reversal_type="all", moving_avg_window=1, weighting="subject"):
    windows = {
        f"m{i}": [make_reversal(["A", "A", "B", "A"] if i else ["A"] * 4, ["C", "C", "A"])]
        for i in range(subjects)
    }
    return compute_reversal_aligned_arms(
        stamp_windows(windows, pre=4, post=3),
        scheme=scheme,
        reversal_type=reversal_type,
        pre=4,
        post=3,
        moving_avg_window=moving_avg_window,
        weighting=weighting,
    )


def choice_lines(ax):
    return [line for line in ax.lines if line.get_linestyle() == "-" and line.get_label()[0] != "_"]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_prev_scheme_uses_blue_green_grey_for_best_second_third():
    ax, _ = plot_reversal_aligned_arms(result_for("prev"))
    lines = choice_lines(ax)
    assert [line.get_label() for line in lines] == ["Prev Best", "Prev Second", "Prev Third"]
    assert [to_hex(line.get_color()) for line in lines] == ["#5da5da", "#60bd68", "#7f7f7f"]


def test_new_scheme_draws_the_third_arm_in_the_prev_third_grey():
    ax, _ = plot_reversal_aligned_arms(result_for("new"))
    lines = choice_lines(ax)
    assert [line.get_label() for line in lines] == ["New Best", "New Second", "New Third"]
    assert [to_hex(line.get_color()) for line in lines] == ["#5da5da", "#60bd68", "#7f7f7f"]
    assert ROLE_COLORS["new_third"] == ROLE_COLORS["prev_third"]


@pytest.mark.parametrize("scheme", ["prev", "new"])
def test_a_figure_draws_the_roles_of_one_scheme_only(scheme):
    # prev third and new third share a colour, so they must never share a panel
    ax, _ = plot_reversal_aligned_arms(result_for(scheme))
    labels = [line.get_label() for line in choice_lines(ax)]
    assert all(label.startswith(scheme.title()) for label in labels)


def test_magnitudes_are_dashed_steps_of_the_same_colour_on_a_twin_axis():
    ax, ax2 = plot_reversal_aligned_arms(result_for("new"))
    assert ax2.get_shared_x_axes().joined(ax, ax2)
    assert ax2.yaxis.get_label_text() == "Mean Reward Magnitude"
    colours = [to_hex(line.get_color()) for line in choice_lines(ax)]
    steps = [line for line in ax2.lines if line.get_linestyle() == "--"]
    assert [to_hex(line.get_color()) for line in steps] == colours
    # new_best is arm C: magnitude 1 before the reversal, 3 after
    np.testing.assert_allclose(steps[0].get_ydata()[[0, -1]], [1, 3])


def test_se_bands_are_drawn_only_where_the_se_exists():
    one, _ = plot_reversal_aligned_arms(result_for("prev", subjects=1))
    assert not [c for c in one.collections if isinstance(c, PolyCollection)]
    two, _ = plot_reversal_aligned_arms(result_for("prev", subjects=2))
    assert [c for c in two.collections if isinstance(c, PolyCollection)]


def test_the_title_keeps_the_heading_and_the_statistics_and_the_rest_goes_to_the_caption():
    for scheme, words in (("prev", "before the block change"), ("new", "after the block change")):
        ax, _ = plot_reversal_aligned_arms(result_for(scheme, moving_avg_window=4), scope="pooled")
        assert ax.get_title() == (
            "Choice Probabilities Around Block Changes (All)\n"
            "mean ± SE across mice, n = 2 mice, n = 2 block changes"
        )
        caption = figure_caption(ax.figure).splitlines()
        assert caption == [
            "Data: pooled.",
            f"Arms labelled by rank {words}.",
            "Moving average over 4 trials (centered).",
        ]


def test_the_event_weighted_title_averages_across_block_changes_and_the_caption_has_the_caveat():
    ax, _ = plot_reversal_aligned_arms(result_for("prev", weighting="event"), scope="pooled")
    stats = ax.get_title().splitlines()[1]
    assert stats == "mean ± SE across block changes, n = 2 block changes, n = 2 mice"
    caption = figure_caption(ax.figure)
    assert "Each block change is weighted equally." in caption
    assert caption.splitlines()[-1].startswith("SE treats block changes as independent")


def test_a_single_mouse_is_a_mean_without_se_and_called_a_mouse():
    ax, _ = plot_reversal_aligned_arms(result_for("prev", subjects=1))
    # one mouse has no SE band, so the title does not claim one
    assert not [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert ax.get_title().splitlines()[1] == "mean across mice, n = 1 mouse, n = 1 block change"


def test_the_stated_n_matches_the_result_and_the_band_matches_the_statistic():
    result = result_for("prev", subjects=2)
    ax, _ = plot_reversal_aligned_arms(result)
    stats = ax.get_title().splitlines()[1]
    assert f"n = {result.num_subjects} mice" in stats
    assert f"n = {result.num_reversals} block changes" in stats
    assert stats.startswith("mean ± SE") == bool(
        [c for c in ax.collections if isinstance(c, PolyCollection)]
    )


def test_the_event_weighted_se_band_is_drawn_for_two_block_changes_of_one_mouse():
    windows = {"m": [make_reversal(["A"] * 4, ["C"] * 3), make_reversal(["B"] * 4, ["C"] * 3)]}
    result = compute_reversal_aligned_arms(
        stamp_windows(windows, pre=4, post=3),
        scheme="prev",
        pre=4,
        post=3,
        moving_avg_window=1,
        weighting="event",
    )
    ax, _ = plot_reversal_aligned_arms(result)
    assert [c for c in ax.collections if isinstance(c, PolyCollection)]


def test_the_heading_names_the_block_change_kind_of_each_scheme():
    prev = result_for("prev", reversal_type="best_to_third")
    assert figure_heading(prev) == "Choice Probabilities Around Block Changes (Prev Best to Third)"
    new = result_for("new", reversal_type="third_to_second")
    assert figure_heading(new).endswith("(New Second from Third)")
    assert figure_stem(new, "problem-04") == (
        "Choice Probabilities Around Block Changes (New Second from Third) "
        "(Arms Labelled by Rank After) (problem-04)"
    )


def test_the_caption_reports_reversals_left_out_for_ties():
    tied = make_reversal(["A"] * 4, ["C"] * 3, before={"A": 3.0, "B": 2.0, "C": 2.0})
    windows = {"m0": [tied, make_reversal(["A"] * 4, ["C"] * 3)]}
    result = compute_reversal_aligned_arms(
        stamp_windows(windows, pre=4, post=3), scheme="prev", pre=4, post=3
    )
    ax, _ = plot_reversal_aligned_arms(result)
    assert "1 block change left out for tied magnitudes." in figure_caption(ax.figure)
    assert "tied" not in ax.get_title()


def test_draws_into_the_axes_it_is_given():
    fig, ax = plt.subplots()
    out_ax, _ = plot_reversal_aligned_arms(result_for("prev"), ax=ax)
    assert out_ax is ax and plt.get_fignums() == [fig.number]


def test_an_empty_result_cannot_be_drawn():
    with pytest.warns(UserWarning):
        empty = result_for("prev", reversal_type="best_to_third", subjects=0)
    with pytest.raises(ValueError, match="no block changes"):
        plot_reversal_aligned_arms(empty)


def test_save_writes_every_format_and_a_caption(tmp_path):
    save_reversal_aligned_arms(result_for("prev"), tmp_path / "sub" / "fig")
    for ext in ("pdf", "png", "svg", "caption.txt"):
        assert (tmp_path / "sub" / f"fig.{ext}").exists(), ext
    assert plt.get_fignums() == []


@pytest.mark.parametrize("save_path", [None, ""])
def test_no_save_path_shows_the_figure(monkeypatch, save_path):
    capture = capture_figures(monkeypatch)
    save_reversal_aligned_arms(result_for("prev"), save_path)
    assert len(capture.shown) == 1 and capture.saved == []


def test_the_last_magnitude_step_has_a_width_of_one_trial():
    result = result_for("new")
    _, ax2 = plot_reversal_aligned_arms(result)
    step = next(line for line in ax2.lines if line.get_linestyle() == "--")
    assert step.get_xdata()[-1] == result.trial_offsets[-1] + 1
    assert step.get_ydata()[-1] == step.get_ydata()[-2]


def test_the_legend_sits_below_the_axes_and_the_title_has_no_doubled_bracket():
    ax, _ = plot_reversal_aligned_arms(result_for("prev", moving_avg_window=4))
    ax.figure.canvas.draw()
    legend_box = ax.get_legend().get_window_extent()
    assert legend_box.y1 <= ax.get_window_extent().y0
    assert "))" not in ax.get_title()
