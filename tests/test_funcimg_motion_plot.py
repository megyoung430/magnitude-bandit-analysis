"""Motion-check figure: motion, corrXY and the cells' shared signal against time, with the cut."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.funcimg_analysis.motion_check import motion_check  # noqa: E402
from src.funcimg_visualization.plot_funcimg_motion import (  # noqa: E402
    motion_check_png_name,
    plot_motion_check,
)
from tests.test_funcimg_motion_check import PERIOD, SETTINGS, _artefact_run  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def result():
    ops, dff = _artefact_run(np.random.default_rng(0))
    return motion_check(dff, ops, PERIOD, **SETTINGS)


def test_three_panels_share_the_time_axis_and_mark_the_cut(result):
    fig = plot_motion_check(result, title="run")
    axes = fig.axes
    assert len(axes) == 3
    assert axes[0].get_shared_x_axes().joined(axes[0], axes[2])
    for ax in axes:
        cut_lines = [
            line for line in ax.get_lines() if list(line.get_xdata()) == [result["cut_s"]] * 2
        ]
        assert cut_lines, "every panel marks the cut"


def test_the_shared_signal_panel_draws_both_channels_and_the_threshold(result):
    fig = plot_motion_check(result, threshold=3.0)
    labels = [line.get_label() for line in fig.axes[2].get_lines()]
    assert any("red" in label for label in labels) and any("green" in label for label in labels)
    flat = [line for line in fig.axes[2].get_lines() if set(line.get_ydata()) == {3.0}]
    assert flat, "the artefact threshold is drawn"


def test_given_axes_are_drawn_into_without_a_new_figure(result):
    fig, axes = plt.subplots(3, 1, sharex=True)
    assert plot_motion_check(result, axes=axes) is fig
    assert len(plt.get_fignums()) == 1


def test_axes_of_the_wrong_number_fail_loudly(result):
    _, axes = plt.subplots(1, 2)
    with pytest.raises(ValueError, match="three"):
        plot_motion_check(result, axes=axes)


def test_non_rigid_and_corrxy_panels_cope_with_runs_that_lack_them(result):
    lacking = {
        **result,
        "motion": {**result["motion"], "xoff1": None, "yoff1": None, "corrXY": None},
    }
    fig = plot_motion_check(lacking)
    assert len(fig.axes) == 3


def test_file_name_carries_the_cut():
    assert motion_check_png_name("ses-224-224", "without_destriping", 212.4) == (
        "motion_check_ses-224-224_without_destriping_cut212s.png"
    )
