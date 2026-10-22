"""Figures of the Zong et al. 2022 pipeline: panel counts, shapes, refusals, rendering."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
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


def test_the_png_name_carries_kind_one_tag_and_variant():
    name = zong2022_png_name("clear_responses", "ses-224-224_run-01", "v")
    assert name == "clear_responses_ses-224-224_run-01_v.png"


def test_zdrift_gives_each_pc_a_plain_panel_and_a_panel_with_contours(result):
    motion = result["motion"]
    y0, y1, x0, x1 = motion["crop"]
    labels = result["labels"][y0:y1, x0:x1]
    fig = plot_zdrift(motion["i_top"], motion["i_bottom"], labels, pcs=[0, 1, 2], n_cols=2)
    panels = image_axes(fig)
    assert len(panels) == 6
    plain = [ax for ax in panels if not ax.collections]
    outlined = [ax for ax in panels if ax.collections]
    assert len(plain) == 3 and len(outlined) == 3  # contours are line collections, not images
    assert all(len(ax.images) == 1 for ax in panels)  # nothing is drawn over the image pixels
    assert [ax.get_title() for ax in plain] == [f"PC {k}: Ibottom - Itop" for k in (1, 2, 3)]
    assert all("outlines" in ax.get_title() for ax in outlined)


def test_zdrift_contours_lie_just_outside_the_rois_not_on_their_pixels():
    labels = np.zeros((40, 40), int)
    labels[15:25, 15:25] = 1
    image = np.random.default_rng(0).normal(size=(1, 40, 40))
    fig = plot_zdrift(image, image * 0, labels, pcs=[0])
    (outlined,) = [ax for ax in image_axes(fig) if ax.collections]
    paths = [path for c in outlined.collections for path in c.get_paths()]
    x, y = np.vstack([path.vertices for path in paths]).T
    inside = (x > 14.5) & (x < 24.5) & (y > 14.5) & (y < 24.5)
    assert not inside.any()  # the contour surrounds the ROI (pixels 15..24), it does not cross it


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


def test_top_responsive_names_cells_by_their_cellpose_label():
    table = pd.DataFrame({"snr": [5.0, 6.0], "cellpose_label": [3, 7], "reward_score": [0.4, 0.8]})
    labels = np.zeros((20, 20), int)
    labels[2:6, 2:6], labels[10:14, 10:14] = 1, 2  # renumbered by row
    windows = np.random.default_rng(0).normal(size=(2, 4, 30))
    fig = plot_top_responsive(
        windows, np.array([0, 1, 2, 3]), [1], table, "reward", np.arange(-10, 20), 0.133,
        np.random.default_rng(1).random((20, 20)), labels,
    )  # fmt: skip
    titles = [ax.get_title() for ax in fig.axes if ax.get_title().startswith("ROI")]
    assert titles == ["ROI 7 | SNR 6.0 | score 0.80"]  # row 1 is Cellpose label 7
    numbers = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert numbers == ["7"]


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
