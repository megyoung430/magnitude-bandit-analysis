"""Figures for the shared-signal analysis: scatter, histograms and residual responses."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.funcimg_analysis.event_frames import pooled_mean_sem  # noqa: E402
from src.funcimg_analysis.run_responses import PULSE_CATEGORIES  # noqa: E402
from src.funcimg_analysis.shared_signal import shared_signal_stats, shift_offsets  # noqa: E402
from src.funcimg_visualization.plot_funcimg_shared import (  # noqa: E402
    plot_partial_r_histograms,
    plot_r2_histograms,
    plot_residual_responses,
    plot_scatter_cell_vs_background,
    residual_png_name,
    settings_tag,
    shared_png_name,
)
from tests.test_funcimg_shared_signal import N_FRAMES, make_cells  # noqa: E402

CHANNELS = ("green", "red")
MEASURES = ("F", "dF/F0")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def runs():
    """Two runs with statistics for both channels and raw measures, from synthetic traces."""
    out = []
    for index, common in enumerate((0.0, 1.5)):
        rng = np.random.default_rng(index)
        offsets = shift_offsets(N_FRAMES, 450, 30, np.random.default_rng(10 + index))
        stats = {}
        for channel in CHANNELS:
            for measure in MEASURES:
                cells, field = make_cells(rng, n_cells=10, common_weight=common)
                stats[(channel, measure)] = shared_signal_stats(cells, field, offsets)
        out.append({"session": "ses-224_date-20260809", "run": f"run-0{index + 1}", "stats": stats})
    return out


def test_the_scatter_has_one_panel_per_channel_and_measure_with_an_identity_line(runs):
    fig = plot_scatter_cell_vs_background(runs, MEASURES)
    axes = fig.axes
    assert len(axes) == 4
    for ax in axes:
        diagonal = [line for line in ax.lines if np.allclose(line.get_xdata(), line.get_ydata())]
        assert diagonal, "identity line missing"
        assert len(ax.collections) == len(runs)  # one scatter per run
        assert "Wilcoxon" in "".join(t.get_text() for t in ax.texts)
        assert ax.get_xlabel() and ax.get_ylabel()


def test_the_scatter_plots_each_cells_two_correlations(runs):
    fig = plot_scatter_cell_vs_background(runs, MEASURES)
    points = fig.axes[0].collections[0].get_offsets()
    stats = runs[0]["stats"][("green", "F")]
    np.testing.assert_allclose(points[:, 0], stats["r_bg"])
    np.testing.assert_allclose(points[:, 1], stats["r_loo"])


def test_plots_draw_into_given_axes_without_making_a_new_figure(runs):
    _, axes = plt.subplots(2, 2)
    fig = plot_scatter_cell_vs_background(runs, MEASURES, axes=axes)
    assert fig is axes[0, 0].figure
    assert len(plt.get_fignums()) == 1


def test_axes_of_the_wrong_shape_raise(runs):
    _, axes = plt.subplots(1, 3)
    for plot in (plot_scatter_cell_vs_background, plot_partial_r_histograms, plot_r2_histograms):
        with pytest.raises(ValueError, match="axes"):
            plot(runs, MEASURES, axes=axes)


def test_the_partial_r_histograms_mark_each_runs_pooled_null_95th_percentile(runs):
    fig = plot_partial_r_histograms(runs, MEASURES)
    ax = fig.axes[0]
    null = runs[0]["stats"][("green", "F")]["null_partial_average"]
    expected = np.nanpercentile(null, 95)
    marks = [line.get_xdata()[0] for line in ax.lines if line.get_linestyle() == "--"]
    assert any(np.isclose(m, expected) for m in marks)
    assert len(marks) == len(
        runs
    )  # one null mark per run, plus a solid line at 0 that is not dashed


def test_the_r2_histograms_have_one_histogram_per_run_in_each_panel(runs):
    fig = plot_r2_histograms(runs, MEASURES)
    for ax in fig.axes:
        assert len(ax.patches) > 0
        assert ax.get_xlim()[0] <= 0 and ax.get_xlim()[1] >= 0.5


def make_pooled(level, channels=CHANNELS, measures=("F z", "dF/F0 z"), n_window=20):
    offsets = np.arange(-5, n_window - 5)
    pooled = {}
    for channel in channels:
        for measure in measures:
            pooled[(channel, measure)] = pooled_mean_sem(
                [{p: np.vstack([level * p * (offsets >= 0)] * 3) for p in PULSE_CATEGORIES}],
                PULSE_CATEGORIES,
            )
    return pooled, offsets


def test_the_residual_figure_puts_original_and_residual_side_by_side_on_shared_y_limits():
    original, offsets = make_pooled(1.0)
    residual, _ = make_pooled(0.1)
    fig = plot_residual_responses(original, residual, offsets, 0.133, x_units="s")
    assert len(fig.axes) == 8  # 2 measures x (green, red) x (original, residual)
    limits = {ax.get_ylim() for ax in fig.axes}
    assert len(limits) == 1
    titles = [ax.get_title() for ax in fig.axes]
    assert sum("Original" in t for t in titles) == 4 and sum("Residual" in t for t in titles) == 4
    top = max(np.max(line.get_ydata()) for line in fig.axes[0].lines if len(line.get_xdata()) > 2)
    assert fig.axes[0].get_ylim()[1] >= top


def test_the_residual_figure_accepts_one_measure_and_given_axes():
    original, offsets = make_pooled(1.0, measures=("F z",))
    residual, _ = make_pooled(0.1, measures=("F z",))
    _, axes = plt.subplots(1, 4)
    fig = plot_residual_responses(
        original, residual, offsets, 0.133, measures=("F z",), axes=axes.reshape(1, 4)
    )
    assert fig is axes[0].figure


def test_file_names_carry_the_settings_that_shape_the_figure():
    tag = settings_tag(60.0, 100, 30.0, cells_only=False)
    assert tag == "detrend60s_shifts100_min30s_allrois"
    assert settings_tag(60.0, 100, 30.0, cells_only=True).endswith("_cellsonly")
    assert shared_png_name("scatter", "ses-224-224_run-01", "without_destriping", tag) == (
        "shared_scatter_ses-224-224_run-01_without_destriping_detrend60s_shifts100_min30s_allrois.png"
    )
    name = residual_png_name("ses-224-224", "with_destriping", "s", (-10.0, 30.0), tag)
    assert "win-10to30s" in name and name.endswith(f"{tag}.png") and "_s_" in name
