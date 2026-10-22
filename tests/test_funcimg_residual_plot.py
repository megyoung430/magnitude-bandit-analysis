"""Figures of the background-corrected ΔF/F0: pooled responses next to Suite2p's, example traces."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.funcimg_analysis.event_frames import pooled_mean_sem  # noqa: E402
from src.funcimg_analysis.run_responses import PULSE_CATEGORIES  # noqa: E402
from src.funcimg_visualization.plot_funcimg_residual import (  # noqa: E402
    dff_comparison_png_name,
    plot_dff_comparison,
    plot_example_traces,
)
from src.funcimg_visualization.plot_funcimg_responses import (  # noqa: E402
    CHANNEL_TITLES,
    PULSE_COLORS,
)

MEASURE = "dF/F0 z"
OFFSETS = np.arange(-10, 41)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def pooled(level):
    return {
        (channel, MEASURE): pooled_mean_sem(
            [{p: np.vstack([level * p * (OFFSETS >= 0)] * 3) for p in PULSE_CATEGORIES}],
            PULSE_CATEGORIES,
        )
        for channel in ("green", "red")
    }


def test_the_comparison_has_suite2p_and_corrected_side_by_side_for_green_and_red():
    fig = plot_dff_comparison(pooled(1.0), pooled(0.1), OFFSETS, 0.133)
    titles = [ax.get_title() for ax in fig.axes]
    assert len(fig.axes) == 4
    assert [("Suite2p" in t) for t in titles] == [True, False, True, False]
    assert [("bg-corrected" in t) for t in titles] == [False, True, False, True]
    assert CHANNEL_TITLES["green"] in titles[0] and CHANNEL_TITLES["red"] in titles[2]


def test_every_panel_shares_the_y_limits_and_shows_all_pulse_counts_in_their_colours():
    fig = plot_dff_comparison(pooled(1.0), pooled(0.1), OFFSETS, 0.133)
    assert len({ax.get_ylim() for ax in fig.axes}) == 1
    top = max(np.max(line.get_ydata()) for line in fig.axes[0].lines if len(line.get_xdata()) > 2)
    assert fig.axes[0].get_ylim()[1] >= top  # the larger response fits
    colours = {line.get_color() for line in fig.axes[1].lines if len(line.get_xdata()) > 2}
    assert colours == set(PULSE_COLORS.values())


def test_the_comparison_draws_into_given_axes_and_refuses_the_wrong_shape():
    _, axes = plt.subplots(1, 4)
    fig = plot_dff_comparison(pooled(1.0), pooled(0.1), OFFSETS, 0.133, axes=axes[None, :])
    assert fig is axes[0].figure
    with pytest.raises(ValueError, match="1 x 4"):
        plot_dff_comparison(pooled(1.0), pooled(0.1), OFFSETS, 0.133, axes=axes[None, :2])


def example(k=3, n_frames=400):
    rng = np.random.default_rng(0)
    dff = rng.normal(0, 0.1, (k, n_frames))
    return {"cells": [7, 3, 9][:k], "suite2p": 5 * dff + 0.3, "corrected": dff}


def test_a_channel_without_usable_cells_says_so_in_its_panels():
    green_only = {k: v for k, v in pooled(1.0).items() if k[0] == "green"}
    fig = plot_dff_comparison(
        green_only, {k: v for k, v in pooled(0.1).items() if k[0] == "green"}, OFFSETS, 0.133
    )
    assert len(fig.axes) == 4
    red_axes = [fig.axes[2], fig.axes[3]]
    assert all("no usable cells" in " ".join(t.get_text() for t in ax.texts) for ax in red_axes)
    assert not any(line for line in fig.axes[0].texts)


def test_example_traces_overlay_both_dffs_in_robust_z_on_one_axis_per_cell():
    time_s = 10 + 0.133 * np.arange(400)
    fig = plot_example_traces(example(), time_s, n_frames_shown=100)
    assert len(fig.axes) == 3
    for ax, roi in zip(fig.axes, [7, 3, 9], strict=True):
        assert f"ROI {roi}" in ax.get_ylabel() and "robust z" in ax.get_ylabel()
        first, second = ax.lines[:2]
        assert len(first.get_xdata()) == len(second.get_xdata()) == 100
        assert first.get_xdata()[0] == pytest.approx(time_s[150])  # the centred 100 frames
        assert first.get_color() != second.get_color()
        # the two traces differ 5-fold in scale and by an offset; in z they are the same trace
        np.testing.assert_allclose(first.get_ydata(), second.get_ydata(), atol=1e-6)
        assert abs(np.median(first.get_ydata())) < 0.5
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels == ["Suite2p dF/F0", "bg-corrected dF/F0"]


def test_example_traces_state_that_the_corrected_baseline_includes_the_neuropil():
    fig = plot_example_traces(example(), 10 + 0.133 * np.arange(400))
    text = " ".join(t.get_text() for t in fig.findobj(matplotlib.text.Text))
    assert "neuropil" in text and "robust z" in text


def test_the_example_traces_caveat_does_not_cover_the_time_axis_label():
    fig = plot_example_traces(example(), 10 + 0.133 * np.arange(400))
    fig.canvas.draw()
    caveat = next(t for t in fig.findobj(matplotlib.text.Text) if "neuropil" in t.get_text())
    label = fig.axes[-1].xaxis.label
    assert not caveat.get_window_extent().overlaps(label.get_window_extent())


def test_example_traces_can_show_every_frame_and_need_matching_shapes():
    time_s = 10 + 0.133 * np.arange(400)
    fig = plot_example_traces(example(), time_s, n_frames_shown=None)
    assert len(fig.axes[0].lines[0].get_xdata()) == 400
    bad = {**example(), "corrected": example()["corrected"][:, :300]}
    with pytest.raises(ValueError, match="same shape"):
        plot_example_traces(bad, time_s)


def test_example_traces_draw_into_given_axes_and_refuse_no_cells():
    time_s = 10 + 0.133 * np.arange(400)
    _, axes = plt.subplots(3, 1)
    assert plot_example_traces(example(), time_s, axes=axes) is axes[0].figure
    with pytest.raises(ValueError, match="no example cell"):
        plot_example_traces(
            {"cells": [], "suite2p": np.empty((0, 400)), "corrected": np.empty((0, 400))}, time_s
        )


def test_file_name_carries_the_window_and_units():
    name = dff_comparison_png_name("ses-224-224_run-01", "without_destriping", "s", (-10.0, 30.0))
    assert name == "bg_corrected_dff_ses-224-224_run-01_without_destriping_s_win-10to30s.png"
