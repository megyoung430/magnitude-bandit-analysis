"""Figures for the shared-signal analysis: scatter, histograms and residual responses."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.funcimg_analysis.event_frames import pooled_mean_sem  # noqa: E402
from src.funcimg_analysis.run_responses import PULSE_CATEGORIES  # noqa: E402
from src.funcimg_analysis.run_statistics import shared_signal_stats  # noqa: E402
from src.funcimg_analysis.shared_signal_null import shift_offsets  # noqa: E402
from src.funcimg_visualization.plot_funcimg_responses import CHANNEL_TITLES, MEASURE_LABELS  # noqa: E402
from src.funcimg_visualization.plot_funcimg_shared import (  # noqa: E402
    plot_partial_r_histograms,
    plot_r2_histograms,
    plot_residual_responses,
    plot_scatter_cell_vs_background,
    residual_png_name,
    settings_tag,
    shared_png_name,
)
from tests.test_funcimg_shared_signal import N_FRAMES, make_cells, smooth_noise  # noqa: E402

CHANNELS = ("green", "red")
MEASURES = ("F", "dF/F0")
SESSION = "ses-224_date-20260809"


def run_stats(rng, common, n_cells=10):
    """Statistics of synthetic cells with their neuropil and a two-half background."""
    offsets = shift_offsets(N_FRAMES, 450, 30, np.random.default_rng(10))
    cells, field = make_cells(rng, n_cells=n_cells, common_weight=common)
    neuropil = np.array([field + 0.5 * smooth_noise(rng) for _ in range(n_cells)])
    halves = np.array([field + 0.3 * smooth_noise(rng), field + 0.3 * smooth_noise(rng)])
    return shared_signal_stats(
        cells,
        field,
        offsets,
        rng=np.random.default_rng(3),
        neuropil_z=neuropil,
        background_halves_z=halves,
    )


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
        stats = {(c, m): run_stats(rng, common) for c in CHANNELS for m in MEASURES}
        out.append({"session": SESSION, "run": f"run-0{index + 1}", "stats": stats})
    return out


# --- scatter ------------------------------------------------------------------------------------


def test_the_scatter_has_one_panel_per_channel_and_measure_with_an_identity_line(runs):
    fig = plot_scatter_cell_vs_background(runs, MEASURES)
    assert len(fig.axes) == 4
    for ax in fig.axes:
        diagonal = [line for line in ax.lines if np.allclose(line.get_xdata(), line.get_ydata())]
        assert diagonal, "identity line missing"
        assert len(ax.collections) == 2 * len(runs)  # corrected (filled) and raw (hollow) per run
        text = "".join(t.get_text() for t in ax.texts)
        assert "Wilcoxon" in text and "corrected" in text and "raw" in text
        assert ax.get_xlabel() and ax.get_ylabel()


def test_the_scatter_plots_the_corrected_and_the_raw_correlation_against_the_background(runs):
    ax = plot_scatter_cell_vs_background(runs, MEASURES).axes[0]
    stats = runs[0]["stats"][("green", "F")]
    corrected, raw = ax.collections[0].get_offsets(), ax.collections[1].get_offsets()
    np.testing.assert_allclose(corrected[:, 0], stats["r_bg"])
    np.testing.assert_allclose(corrected[:, 1], stats["r_loo_corrected"])
    np.testing.assert_allclose(raw[:, 1], stats["r_loo"])
    assert np.median(stats["r_loo_corrected"]) > np.median(
        stats["r_loo"]
    )  # the correction raises it


def test_the_scatter_marks_the_shifted_null_95th_percentile_on_both_axes(runs):
    ax = plot_scatter_cell_vs_background(runs, MEASURES).axes[0]
    stats = runs[0]["stats"][("green", "F")]
    x_level = np.nanpercentile(stats["null_r_bg"], 95)
    y_null = np.clip(stats["null_r_loo"] / np.sqrt(stats["loo_reliability"])[None, :], -1, 1)
    y_level = np.nanpercentile(y_null, 95)
    vertical = [
        line.get_xdata()[0] for line in ax.lines if line.get_xdata()[0] == line.get_xdata()[-1]
    ]
    horizontal = [
        line.get_ydata()[0] for line in ax.lines if line.get_ydata()[0] == line.get_ydata()[-1]
    ]
    assert any(np.isclose(x, x_level) for x in vertical)
    assert any(np.isclose(y, y_level) for y in horizontal)


def test_the_scatter_names_each_run_by_session_and_run_and_counts_cells_off_the_axes(runs):
    far = {
        **runs[0],
        "stats": {key: {**s, "r_bg": s["r_bg"].copy()} for key, s in runs[0]["stats"].items()},
    }
    for s in far["stats"].values():
        s["r_bg"][0] = -0.95  # beyond the axis limit
    text = "".join(
        t.get_text() for t in plot_scatter_cell_vs_background([far], MEASURES).axes[0].texts
    )
    assert "ses-224 run-01" in text and "1 cell(s) outside the axes" in text
    ordinary = "".join(
        t.get_text() for t in plot_scatter_cell_vs_background(runs, MEASURES).axes[0].texts
    )
    assert "ses-224 run-02" in ordinary


def test_a_scatter_without_the_null_levels_still_draws(runs):
    bare = [
        {
            **r,
            "stats": {
                k: {n: v for n, v in s.items() if not n.startswith("null_")}
                for k, s in r["stats"].items()
            },
        }
        for r in runs
    ]
    assert len(plot_scatter_cell_vs_background(bare, MEASURES).axes) == 4


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


# --- histograms ---------------------------------------------------------------------------------


def line_positions(ax, linestyle):
    return [line.get_xdata()[0] for line in ax.lines if line.get_linestyle() == linestyle]


def test_the_partial_r_histograms_mark_each_runs_pooled_null_95th_percentile_for_both_controls(
    runs,
):
    ax = plot_partial_r_histograms(runs, MEASURES).axes[0]
    stats = runs[0]["stats"][("green", "F")]
    assert any(
        np.isclose(m, np.nanpercentile(stats["null_partial_average"], 95))
        for m in line_positions(ax, "--")
    )
    assert any(
        np.isclose(m, np.nanpercentile(stats["null_partial_neuropil_average"], 95))
        for m in line_positions(ax, "-.")
    )
    assert len(line_positions(ax, "--")) == len(runs) == len(line_positions(ax, "-."))


def test_the_partial_r_histograms_state_the_backgrounds_reliability_and_both_medians(runs):
    ax = plot_partial_r_histograms(runs, MEASURES).axes[0]
    stats = runs[0]["stats"][("green", "F")]
    text = "".join(t.get_text() for t in ax.texts)
    assert f"{stats['bg_half_r']:.2f}" in text and f"{stats['bg_reliability']:.2f}" in text
    assert f"{np.median(stats['partial_r']):.2f}" in text
    assert f"{np.median(stats['partial_r_neuropil']):.2f}" in text
    assert "ses-224 run-01" in text


def test_cells_beyond_the_histogram_bins_are_counted_in_the_panel(runs):
    outlier = {
        **runs[0],
        "stats": {key: {**s, "r2": s["r2"].copy()} for key, s in runs[0]["stats"].items()},
    }
    for s in outlier["stats"].values():
        s["r2"][0] = 1.7
    ax = plot_r2_histograms([outlier], MEASURES).axes[0]
    assert "1 cell(s) outside the bins" in "".join(t.get_text() for t in ax.texts)
    clean = plot_r2_histograms(runs, MEASURES).axes[0]
    assert "outside" not in "".join(t.get_text() for t in clean.texts)


def test_the_r2_histograms_have_one_histogram_per_run_in_each_panel(runs):
    fig = plot_r2_histograms(runs, MEASURES)
    for ax in fig.axes:
        assert len(ax.patches) > 0
        assert ax.get_xlim()[0] <= 0 and ax.get_xlim()[1] >= 0.5


# --- residual responses -------------------------------------------------------------------------


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
    assert len({ax.get_ylim() for ax in fig.axes}) == 1
    top = max(np.max(line.get_ydata()) for line in fig.axes[0].lines if len(line.get_xdata()) > 2)
    assert fig.axes[0].get_ylim()[1] >= top


def test_the_residual_panel_titles_are_built_from_the_channel_and_measure_labels():
    original, offsets = make_pooled(1.0)
    residual, _ = make_pooled(0.1)
    fig = plot_residual_responses(original, residual, offsets, 0.133)
    expected = {
        f"{name}, {CHANNEL_TITLES[channel]}\n{MEASURE_LABELS[measure][0]}"
        for name in ("Original", "Residual")
        for channel in CHANNELS
        for measure in ("F z", "dF/F0 z")
    }
    assert {ax.get_title() for ax in fig.axes} == expected


def test_the_residual_figure_accepts_one_measure_and_given_axes():
    original, offsets = make_pooled(1.0, measures=("F z",))
    residual, _ = make_pooled(0.1, measures=("F z",))
    _, axes = plt.subplots(1, 4)
    fig = plot_residual_responses(
        original, residual, offsets, 0.133, measures=("F z",), axes=axes.reshape(1, 4)
    )
    assert fig is axes[0].figure


def test_a_title_is_put_above_each_figures_own_description(runs):
    for plot in (plot_scatter_cell_vs_background, plot_partial_r_histograms, plot_r2_histograms):
        text = plot(runs, MEASURES, title="sub-03, ses-224").texts[0].get_text()
        assert text.startswith("sub-03, ses-224\n") and len(text) > 30


# --- file names ---------------------------------------------------------------------------------


def test_file_names_carry_the_settings_that_shape_the_figure():
    tag = settings_tag(60.0, 100, 30.0, cells_only=False)
    assert tag == "detrend60s_shifts100_min30s_allrois_fitoutside"
    assert settings_tag(60.0, 100, 30.0, cells_only=True).endswith("_cellsonly_fitoutside")
    assert settings_tag(60.0, 100, 30.0, False, fit_exclude_outcomes=False).endswith("_fitall")
    assert settings_tag(60.0, 100, 30.0, False, fit_exclude_window_s=(-0.5, 2.0)).endswith(
        "_fitoutside-0.5to2s"
    )
    assert shared_png_name("scatter", "ses-224-224_run-01", "without_destriping", tag) == (
        "shared_scatter_ses-224-224_run-01_without_destriping_" + tag + ".png"
    )
    name = residual_png_name("ses-224-224", "with_destriping", "s", (-10.0, 30.0), tag)
    assert "win-10to30s" in name and name.endswith(f"{tag}.png") and "_s_" in name
