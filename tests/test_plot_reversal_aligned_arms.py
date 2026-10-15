"""Tests for the reversal-aligned arm figure: colours, twin axis, SE bands, title and saving."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_hex

from src.behavior_analysis.reversal_aligned_arms import compute_reversal_aligned_arms
from src.behavior_visualization.plot_reversal_aligned_arms import (
    plot_reversal_aligned_arms,
    save_reversal_aligned_arms,
)
from tests.plot_capture import capture_figures
from tests.reversal_arm_fixtures import make_reversal


def result_for(scheme, subjects=2, reversal_type="all", moving_avg_window=1):
    windows = {
        f"m{i}": [make_reversal(["A", "A", "B", "A"] if i else ["A"] * 4, ["C", "C", "A"])]
        for i in range(subjects)
    }
    return compute_reversal_aligned_arms(
        windows,
        scheme=scheme,
        reversal_type=reversal_type,
        pre=4,
        post=3,
        moving_avg_window=moving_avg_window,
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


def test_new_scheme_draws_the_third_arm_in_black():
    ax, _ = plot_reversal_aligned_arms(result_for("new"))
    lines = choice_lines(ax)
    assert [line.get_label() for line in lines] == ["New Best", "New Second", "New Third"]
    assert [to_hex(line.get_color()) for line in lines] == ["#5da5da", "#60bd68", "#000000"]


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


def test_the_title_says_how_arms_are_labelled_and_how_types_are_defined():
    for scheme, words in (("prev", "before the reversal"), ("new", "after the reversal")):
        ax, _ = plot_reversal_aligned_arms(result_for(scheme, moving_avg_window=4))
        title = ax.get_title()
        assert words in title
        assert "type defined by the prev ranking" in title
        assert "n=2 subjects" in title and "n=2 reversals" in title
        assert "moving average over 4 trials (centered)" in title


def test_the_title_reports_reversals_left_out_for_ties():
    tied = make_reversal(["A"] * 4, ["C"] * 3, before={"A": 3.0, "B": 2.0, "C": 2.0})
    windows = {"m0": [tied, make_reversal(["A"] * 4, ["C"] * 3)]}
    result = compute_reversal_aligned_arms(windows, scheme="prev", pre=4, post=3)
    ax, _ = plot_reversal_aligned_arms(result)
    assert "1 left out for tied magnitudes" in ax.get_title()


def test_draws_into_the_axes_it_is_given():
    fig, ax = plt.subplots()
    out_ax, _ = plot_reversal_aligned_arms(result_for("prev"), ax=ax)
    assert out_ax is ax and plt.get_fignums() == [fig.number]


def test_an_empty_result_cannot_be_drawn():
    with pytest.warns(UserWarning):
        empty = result_for("prev", reversal_type="best_to_third", subjects=0)
    with pytest.raises(ValueError, match="no reversals"):
        plot_reversal_aligned_arms(empty)


def test_save_writes_png_and_pdf(tmp_path):
    save_reversal_aligned_arms(result_for("prev"), tmp_path / "sub" / "fig")
    assert (tmp_path / "sub" / "fig.png").exists() and (tmp_path / "sub" / "fig.pdf").exists()
    assert plt.get_fignums() == []


@pytest.mark.parametrize("save_path", [None, ""])
def test_no_save_path_shows_the_figure(monkeypatch, save_path):
    capture = capture_figures(monkeypatch)
    save_reversal_aligned_arms(result_for("prev"), save_path)
    assert len(capture.shown) == 1 and capture.saved == []
