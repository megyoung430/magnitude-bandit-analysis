"""Transient-check figures: skewness and event-rate histograms, and example traces by eye."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.funcimg_analysis.event_frames import robust_z  # noqa: E402
from src.funcimg_analysis.transients import run_transients  # noqa: E402
from src.funcimg_visualization.plot_funcimg_transients import (  # noqa: E402
    GREEN,
    RED,
    plot_transient_examples,
    plot_transient_histograms,
    transients_png_name,
)
from tests.test_funcimg_transients import (  # noqa: E402
    N_CELLS,
    N_FRAMES,
    PERIOD,
    SETTINGS,
    add_transients,
    noise,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def run():
    rng = np.random.default_rng(0)
    green = add_transients(noise(rng), rng, n_planted=15)
    red = noise(rng)
    red[5] = np.nan  # one cell without usable red
    return {
        "time_s": 20.0 + PERIOD * np.arange(N_FRAMES),
        "channels": {
            "green": {"dff": 0.1 * green, "kept": np.ones(N_CELLS, bool)},
            "red": {"dff": 0.1 * red, "kept": ~np.isnan(red).all(axis=1)},
        },
    }


@pytest.fixture(scope="module")
def stats(run):
    return run_transients(run, **SETTINGS)


def bars(ax, color):
    return [p for p in ax.patches if matplotlib.colors.to_hex(p.get_facecolor()[:3]) == color]


def test_two_histograms_overlay_green_and_red_cells(stats):
    fig = plot_transient_histograms([stats])
    skew_ax, rate_ax = fig.axes
    for ax in (skew_ax, rate_ax):
        assert sum(p.get_height() for p in bars(ax, GREEN)) == N_CELLS
        assert sum(p.get_height() for p in bars(ax, RED)) == N_CELLS - 1  # the NaN cell is absent
    assert "skewness" in skew_ax.get_xlabel().lower() and "per min" in rate_ax.get_xlabel()


def test_the_skewness_panel_marks_the_red_95th_percentile(stats):
    skew_ax = plot_transient_histograms([stats]).axes[0]
    lines = [line for line in skew_ax.lines if len(set(line.get_xdata())) == 1]
    expected = np.nanpercentile(stats["red"]["skew"], 95)
    assert any(line.get_xdata()[0] == pytest.approx(expected) for line in lines)


def test_runs_are_pooled_and_axes_can_be_given(stats):
    _, axes = plt.subplots(1, 2)
    fig = plot_transient_histograms([stats, stats], axes=axes)
    assert fig is axes[0].figure
    assert sum(p.get_height() for p in bars(axes[0], GREEN)) == 2 * N_CELLS
    with pytest.raises(ValueError, match="2 axes"):
        plot_transient_histograms([stats], axes=axes[:1])


def test_examples_show_the_six_most_skewed_green_cells_each_with_its_red_trace_below(run, stats):
    fig = plot_transient_examples(run, stats, n=6, n_frames_shown=500)
    assert len(fig.axes) == 12
    order = np.argsort(-stats["green"]["skew"])
    expected = [int(c) for c in order if np.isfinite(stats["red"]["skew"][c])][:6]
    for pair, cell in enumerate(expected):
        green_ax, red_ax = fig.axes[2 * pair], fig.axes[2 * pair + 1]
        assert f"ROI {cell}" in green_ax.get_ylabel() and "red" in red_ax.get_ylabel()
        (g,), (r,) = green_ax.lines, red_ax.lines
        assert len(g.get_xdata()) == len(r.get_xdata()) == 500
        z_green = robust_z(run["channels"]["green"]["dff"])[0][cell][1750:2250]
        np.testing.assert_allclose(g.get_ydata(), z_green)
        assert g.get_color() == GREEN and r.get_color() == RED


def test_a_cell_without_red_data_is_not_among_the_examples(run, stats):
    fig = plot_transient_examples(run, stats, n=40)
    labels = " ".join(ax.get_ylabel() for ax in fig.axes)
    assert "ROI 5 " not in labels and len(fig.axes) == 2 * (N_CELLS - 1)


def test_examples_can_show_every_frame(run, stats):
    fig = plot_transient_examples(run, stats, n=2, n_frames_shown=None)
    assert len(fig.axes[0].lines[0].get_xdata()) == N_FRAMES


def test_file_name_carries_the_thresholds():
    assert (
        transients_png_name("histograms", "ses-224-224_run-01", "without_destriping", 3.0, 1.0)
        == "transients_histograms_ses-224-224_run-01_without_destriping_z3_gap1s.png"
    )


def test_example_traces_are_in_robust_z_on_one_shared_y_scale():
    rng = np.random.default_rng(0)
    run = {
        "time_s": 20.0 + PERIOD * np.arange(N_FRAMES),
        "channels": {
            "green": {"dff": 0.1 * add_transients(noise(rng), rng, n_planted=40)},
            "red": {"dff": 100 * noise(rng)},  # a very different raw scale
        },
    }
    stats = run_transients(run, **SETTINGS)
    fig = plot_transient_examples(run, stats, n=3, n_frames_shown=500)
    assert len({ax.get_ylim() for ax in fig.axes}) == 1
    assert all("robust z" in ax.get_ylabel() for ax in fig.axes)
    red = fig.axes[1].lines[0].get_ydata()
    assert np.abs(red).max() < 8  # not 100 x larger than the green's scale


def test_examples_draw_into_given_axes():
    rng = np.random.default_rng(0)
    run = {
        "time_s": 20.0 + PERIOD * np.arange(N_FRAMES),
        "channels": {
            "green": {"dff": add_transients(noise(rng), rng, 40)},
            "red": {"dff": noise(rng)},
        },
    }
    stats = run_transients(run, **SETTINGS)
    _, axes = plt.subplots(4, 1)
    assert plot_transient_examples(run, stats, n=2, axes=axes) is axes[0].figure
    with pytest.raises(ValueError, match="Need 6 axes"):
        plot_transient_examples(run, stats, n=3, axes=axes)
