import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.container import ErrorbarContainer

from src.behavior_visualization import plot_task_statistics as ts
from src.behavior_visualization.plot_style import MOUSE_COLORS
from src.behavior_visualization.plot_task_statistics import (
    plot_block_length_panel,
    plot_block_lengths,
)
from tests.plot_capture import cap_size, capture_figures, title_pad_points, was_tight_laid_out


def summary(**overrides):
    base = dict(
        blocks=[1, 2, 3],
        meds=np.array([10.0, 20.0, 30.0]),
        ses=np.array([1.0, 2.0, 3.0]),
        per_mouse_blocklens={
            "a": {1: 8.0, 2: 20.0},
            "b": {1: 12.0, 2: 20.0, 3: 30.0},
            "c": {},
        },
        mice=["a", "b", "c"],
    )
    return {**base, **overrides}


@pytest.fixture
def capture(monkeypatch):
    return capture_figures(monkeypatch)


def draw(capture, data=None, **kwargs):
    plot_block_lengths(data or summary(), **kwargs)
    (fig,) = (capture.shown or [[s.figure] for s in capture.saved])[0]
    return fig, fig.axes[0]


def mouse_lines(ax):
    return [ln for ln in ax.lines if ln.get_marker() == "o"]


# ---------------------------------------------------------------- layout geometry


def test_tallest_bar_is_max_of_median_plus_se_and_one_without_bars():
    assert ts._tallest_bar(np.array([10.0, 20.0]), np.array([1.0, 5.0])) == 25.0
    assert ts._tallest_bar(np.array([]), np.array([])) == 1.0


def test_tallest_bar_counts_a_bar_whose_se_is_nan():
    assert ts._tallest_bar(np.array([10.0, 20.0]), np.array([1.0, np.nan])) == 20.0


def test_annotation_height_is_five_percent_above_tallest_bar_but_at_least_one():
    meds, ses = np.array([10.0, 20.0]), np.array([1.0, 5.0])
    assert ts._annotation_height(meds, ses) == pytest.approx(25.0 * 1.05)
    assert ts._annotation_height(np.array([0.1]), np.array([0.1])) == pytest.approx(1.05)


def test_y_upper_limit_fits_labels_and_bars():
    meds, ses = np.array([10.0]), np.array([2.0])
    assert ts._y_upper_limit(50.0, meds, ses) == pytest.approx(50.0 * 1.08)
    assert ts._y_upper_limit(5.0, meds, ses) == pytest.approx(12.0 * 1.08)


def test_mouse_points_keep_block_order_and_skip_missing_blocks():
    xs, ys = ts._mouse_points({3: 30.0, 1: 8.0}, [1, 2, 3])
    assert xs.tolist() == [0.0, 2.0] and ys.tolist() == [8.0, 30.0]
    xs, ys = ts._mouse_points({}, [1, 2])
    assert len(xs) == 0 and len(ys) == 0


# ---------------------------------------------------------------- figure


def test_figure_size_and_median_bars_with_se(capture):
    fig, ax = draw(capture, jitter=0)
    assert tuple(fig.get_size_inches()) == (10, 4.8) and len(fig.axes) == 1
    assert [b.get_height() for b in ax.patches] == [10.0, 20.0, 30.0]
    bar = ax.patches[0]
    assert bar.get_facecolor() == pytest.approx((0.6, 0.6, 0.6, 0.55))
    assert bar.get_edgecolor() == pytest.approx((0, 0, 0, 0.55))
    assert bar.get_linewidth() == 1.5
    (errorbar,) = [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
    segments = errorbar.lines[2][0].get_segments()
    assert [seg[:, 1].tolist() for seg in segments] == [[9.0, 11.0], [18.0, 22.0], [27.0, 33.0]]


def test_median_labels_default_to_five_percent_above_tallest_bar(capture):
    _, ax = draw(capture, jitter=0)
    top = 33.0 * 1.05
    assert [(t.get_text(), t.get_position()) for t in ax.texts] == [
        ("10.0", (0, top)),
        ("20.0", (1, top)),
        ("30.0", (2, top)),
    ]
    assert all(t.get_fontsize() == 11 and not t.get_clip_on() for t in ax.texts)
    assert ax.get_ylim() == (0, pytest.approx(top * 1.08))


def test_explicit_annotate_y_moves_labels_and_limit(capture):
    _, ax = draw(capture, jitter=0, annotate_y=300)
    assert {t.get_position()[1] for t in ax.texts} == {300}
    assert ax.get_ylim() == (0, pytest.approx(300 * 1.08))


def test_low_annotate_y_still_leaves_room_for_the_tallest_bar(capture):
    _, ax = draw(capture, jitter=0, annotate_y=1)
    assert ax.get_ylim() == (0, pytest.approx(33.0 * 1.08))


def test_mouse_lines_without_jitter(capture):
    _, ax = draw(capture, jitter=0)
    a, b = mouse_lines(ax)
    assert (a.get_xdata().tolist(), a.get_ydata().tolist()) == ([0, 1], [8.0, 20.0])
    assert (b.get_xdata().tolist(), b.get_ydata().tolist()) == ([0, 1, 2], [12.0, 20.0, 30.0])
    assert (a.get_color(), b.get_color()) == (MOUSE_COLORS[0], MOUSE_COLORS[1])
    for line in (a, b):
        assert (line.get_linewidth(), line.get_alpha(), line.get_marker()) == (2.5, 0.9, "o")
        assert line.get_markersize() == 6 and line.get_zorder() == 3


def test_jitter_is_uniform_within_the_requested_width(capture):
    expected = np.random.default_rng(0)
    _, ax = draw(capture, jitter=0.2)
    a, b = mouse_lines(ax)
    np.testing.assert_allclose(a.get_xdata(), [0, 1] + expected.uniform(-0.2, 0.2, 2))
    np.testing.assert_allclose(b.get_xdata(), [0, 1, 2] + expected.uniform(-0.2, 0.2, 3))


def test_legend_counts_blocks_per_mouse_and_skips_mice_without_data(capture):
    _, ax = draw(capture, jitter=0)
    legend = ax.get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["a (n=2 blocks)", "b (n=3 blocks)"]
    assert legend._loc == 1 and legend.get_texts()[0].get_fontsize() == 10


def test_axis_text_and_title(capture):
    _, ax = draw(capture, jitter=0)
    assert ax.get_xlabel() == "Block Number" and ax.get_ylabel() == "Block Length (Trials)"
    assert ax.xaxis.label.get_fontsize() == 12 and ax.yaxis.label.get_fontsize() == 12
    labels = ax.get_xticklabels()
    assert [t.get_text() for t in labels] == ["1", "2", "3"]
    assert all(t.get_fontsize() == 12 for t in labels)
    assert ax.get_title() == "Block Lengths\n(median ± se across mice | n=3 mice)"
    assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()


def test_show_without_save_path_and_save_with_one(capture, tmp_path):
    fig, _ = draw(capture, jitter=0)
    assert len(capture.shown) == 1 and not plt.fignum_exists(fig.number)
    base = tmp_path / "deep" / "Median Block Lengths"
    plot_block_lengths(summary(), jitter=0, save_path=base)
    pdf, png, svg = capture.saved
    assert svg.path == str(base) + ".svg" and svg.kwargs == {"bbox_inches": "tight"}
    assert pdf.path == str(base) + ".pdf" and pdf.kwargs == {"bbox_inches": "tight"}
    assert png.path == str(base) + ".png" and png.kwargs == {"dpi": 300, "bbox_inches": "tight"}
    assert base.parent.is_dir()


def test_no_blocks_gives_an_empty_chart_with_unit_scale(capture):
    empty = summary(blocks=[], meds=np.array([]), ses=np.array([]), per_mouse_blocklens={}, mice=[])
    _, ax = draw(capture, empty)
    assert len(ax.patches) == 0 and ax.get_ylim() == (0, pytest.approx(1.05 * 1.08))


def test_layout_is_tightened_and_padding_and_caps_are_kept(capture):
    fig, ax = draw(capture, jitter=0)
    assert was_tight_laid_out(fig)
    assert title_pad_points(ax) == pytest.approx(20)
    (bars,) = [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
    assert cap_size(bars) == 6


def test_panel_draws_into_the_given_axes_without_opening_a_figure(capture):
    fig, axes = plt.subplots(1, 2)
    ax = plot_block_length_panel(summary(), jitter=0, ax=axes[0])
    assert ax is axes[0]
    assert plt.get_fignums() == [fig.number] and len(fig.axes) == 2
    assert len(axes[0].patches) == 3 and len(axes[1].patches) == 0
    assert capture.shown == [] and capture.saved == []


def test_panel_creates_a_10_by_4_8_figure_when_no_axes_are_given(capture):
    ax = plot_block_length_panel(summary(), jitter=0)
    assert plt.get_fignums() == [ax.figure.number]
    assert tuple(ax.figure.get_size_inches()) == (10, 4.8)


def test_median_labels_stop_at_the_shorter_of_blocks_and_medians(capture):
    # Known issue pinned: a summary with fewer medians than blocks still draws, and
    # only the first block gets a value label.
    _, ax = draw(capture, summary(meds=np.array([5.0]), ses=np.array([1.0])), jitter=0)
    assert [t.get_text() for t in ax.texts] == ["5.0"]


def raise_while_drawing(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(ts, "_plot_median_bars", boom)


def test_a_figure_created_by_the_panel_is_closed_when_drawing_raises(capture, monkeypatch):
    raise_while_drawing(monkeypatch)
    with pytest.raises(RuntimeError, match="boom"):
        plot_block_length_panel(summary(), jitter=0)
    assert plt.get_fignums() == []


def test_a_figure_given_to_the_panel_stays_open_when_drawing_raises(capture, monkeypatch):
    raise_while_drawing(monkeypatch)
    fig, ax = plt.subplots()
    with pytest.raises(RuntimeError, match="boom"):
        plot_block_length_panel(summary(), jitter=0, ax=ax)
    assert plt.get_fignums() == [fig.number]


def test_the_public_function_closes_its_figure_when_drawing_or_saving_raises(
    capture, monkeypatch, tmp_path
):
    raise_while_drawing(monkeypatch)
    with pytest.raises(RuntimeError, match="boom"):
        plot_block_lengths(summary(), jitter=0)
    monkeypatch.undo()

    def cannot_save(self, fname, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(plt.Figure, "savefig", cannot_save)
    with pytest.raises(OSError, match="disk full"):
        plot_block_lengths(summary(), jitter=0, save_path=tmp_path / "x")
    assert plt.get_fignums() == []
