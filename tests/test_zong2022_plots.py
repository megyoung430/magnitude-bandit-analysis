"""Figures of the Zong et al. 2022 pipeline: panel counts, shapes, refusals, rendering."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("mini2p_preprocessing")  # the run fixture needs the funcimg extra

from src.funcimg_analysis.zong2022_motion import MotionSettings, nonrigid_motion  # noqa: E402
from src.funcimg_analysis.zong2022_responses import ResponseSettings  # noqa: E402
from src.funcimg_analysis.zong2022_run import analyse_run, top_responsive_cells  # noqa: E402
from src.funcimg_analysis.zong2022_signal import Zong2022Settings, event_counts_by_tau  # noqa: E402
from src.funcimg_visualization.plot_funcimg_zong2022 import (  # noqa: E402
    plot_baseline_example,
    plot_clear_responses,
    plot_motion_summary,
    plot_score_vs_null,
    plot_signal_examples,
    plot_snr_histogram,
    plot_tau_sensitivity,
    plot_top_responsive,
    plot_zdrift,
    zong2022_png_name,
)
from tests.zong2022_run_fixtures import outcome_table, write_synthetic_run  # noqa: E402


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    paths = write_synthetic_run(tmp_path_factory.mktemp("der"))
    return analyse_run(
        paths["plane_dir"], paths["run_dir"], "without_destriping",
        settings=Zong2022Settings(), responses=ResponseSettings(),
        motion=MotionSettings(n_pcs=4, n_extremes=50, max_frames=400),
        cache_dir=tmp_path_factory.mktemp("cache"), load_outcomes=lambda _: outcome_table(),
    )  # fmt: skip


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def image_axes(fig):
    return [ax for ax in fig.axes if ax.images]


def test_the_png_name_carries_kind_tag_run_and_variant():
    name = zong2022_png_name("clear_responses", "ses-224-224_run-01", "ses-224_run-01", "v")
    assert name == "clear_responses_ses-224-224_run-01_ses-224_run-01_v.png"


def test_zdrift_has_one_panel_per_requested_pc(result):
    motion = result["motion"]
    y0, y1, x0, x1 = motion["crop"]
    labels = result["labels"][y0:y1, x0:x1]
    fig = plot_zdrift(motion["i_top"], motion["i_bottom"], labels, pcs=[0, 1, 2], n_cols=2)
    assert len(image_axes(fig)) == 3  # one panel per PC, each with the outline overlay
    titled = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert titled == ["PC 1: Ibottom - Itop", "PC 2: Ibottom - Itop", "PC 3: Ibottom - Itop"]


def test_zdrift_shows_the_difference_of_the_two_images_symmetrically(result):
    motion = result["motion"]
    y0, y1, x0, x1 = motion["crop"]
    fig = plot_zdrift(motion["i_top"], motion["i_bottom"], result["labels"][y0:y1, x0:x1], [0])
    first = fig.axes[0].images[0]
    np.testing.assert_allclose(first.get_array(), motion["i_bottom"][0] - motion["i_top"][0])
    assert first.get_clim()[0] == -first.get_clim()[1]


def test_clear_responses_draw_a_line_per_pulse_count_in_each_of_four_panels(result):
    reward = result["reward"]
    offsets = reward["aligned"]["dff_clear"].offsets
    fig = plot_clear_responses(reward["pooled"], offsets, result["period_s"])
    assert len(fig.axes) == 4
    assert all(len(ax.lines) >= 5 for ax in fig.axes)  # 5 pulse counts + the zero line


def test_clear_responses_refuse_axes_of_the_wrong_shape(result):
    reward = result["reward"]
    _, axes = plt.subplots(1, 3)
    with pytest.raises(ValueError, match="2 x 2"):
        plot_clear_responses(
            reward["pooled"], reward["aligned"]["dff_clear"].offsets, 0.133, axes=axes
        )


def test_motion_summary_has_three_panels_and_handles_missing_nonrigid_shifts(result):
    motion = result["motion"]
    fig = plot_motion_summary(motion["drift"], motion["offsets"], None, result["period_s"])
    assert len(fig.axes) == 3
    xoff1 = np.zeros((50, 9))
    fig2 = plot_motion_summary(
        motion["drift"], motion["offsets"], nonrigid_motion(xoff1, xoff1), 0.133
    )
    assert len(fig2.axes[2].lines) == 1


def test_top_responsive_titles_carry_the_snr_and_the_score(result):
    table = result["table"]
    rois = top_responsive_cells(table, "reward", 3)
    assert rois.tolist() == [0]
    aligned = result["reward"]["aligned"]["dff_clear"]
    fig = plot_top_responsive(
        aligned.windows, aligned.pulses, rois, table, "reward", aligned.offsets,
        result["period_s"], result["mean_image"], result["labels"],
    )  # fmt: skip
    titles = [ax.get_title() for ax in fig.axes if ax.get_title().startswith("ROI")]
    assert len(titles) == 1
    assert f"SNR {table['snr'].iloc[0]:.1f}" in titles[0]
    assert f"score {table['reward_score'].iloc[0]:.2f}" in titles[0]


def test_small_panels_draw_into_a_given_axis(result):
    _, ax = plt.subplots()
    assert plot_snr_histogram(result["table"], 3.0, ax=ax) is ax.figure
    assert plot_score_vs_null(result["table"], "reward", ax=ax) is ax.figure
    assert plot_baseline_example(result["signal"], 0, ax=ax) is ax.figure
    counts = event_counts_by_tau(
        result["signal"].delta_f, result["signal"].transient_mask, 1 / 0.133, (0.45, 0.7)
    )
    assert plot_tau_sensitivity(counts, 0.7, ax=ax) is ax.figure


def test_signal_examples_give_a_row_per_cell(result):
    fig = plot_signal_examples(result["signal"], [0, 1], (0, 200))
    assert len(fig.axes) == 2
