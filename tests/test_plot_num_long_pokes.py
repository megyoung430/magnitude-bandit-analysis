import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.container import ErrorbarContainer

from src.behavior_analysis.long_pokes_by_session import summarize_pokes_by_session
from src.behavior_visualization.plot_num_long_pokes import (
    plot_num_pokes_across_mice,
    plot_num_pokes_panel,
)
from src.behavior_visualization.plot_style import MOUSE_COLORS
from tests.plot_capture import cap_size, capture_figures, title_pad_points, was_tight_laid_out

S1, S2, S3 = "ses-01_date-20260301", "ses-02_date-20260302", "ses-10_date-20260310"
NAN = np.nan

POKES = {
    "mB": {S1: 3.0, S2: NAN, S3: 5.0},
    "mA": {S1: 1.0, S2: 3.0},
    "average_across_subjects": {S1: 2.0, S2: 3.0, S3: 5.0},
}


@pytest.fixture
def capture(monkeypatch):
    # The jitter generator is unseeded in production; fix it so x positions are exact.
    monkeypatch.setattr(np.random, "default_rng", lambda: np.random.Generator(np.random.PCG64(0)))
    return capture_figures(monkeypatch)


def mouse_lines(ax):
    """Per-mouse lines; error-bar caps and stems are also in ``ax.lines``."""
    return [ln for ln in ax.lines if ln.get_marker() == "o"]


def draw(capture, data=POKES, y_text=6, **kwargs):
    plot_num_pokes_across_mice(data, y_text, **kwargs)
    return (capture.shown or [[s.figure] for s in capture.saved])[0][0]


# ---------------------------------------------------------------- analysis


def test_summary_ignores_average_key_and_sorts_sessions_numerically():
    data = {"m": {"ses-10_date-x": 1.0, "ses-02_date-x": 2.0}, "average_across_subjects": {"a": 1}}
    summary = summarize_pokes_by_session(data)
    assert summary.mice == ["m"]
    assert summary.sessions == ["ses-02_date-x", "ses-10_date-x"]


def test_summary_mean_and_se_skip_missing_values():
    summary = summarize_pokes_by_session(POKES)
    assert summary.mice == ["mA", "mB"]
    assert summary.sessions == [S1, S2, S3]
    np.testing.assert_allclose(summary.mean, [2.0, 3.0, 5.0])
    # two values -> SD/sqrt(2); a single value -> NaN, never 0
    np.testing.assert_allclose(summary.se, [np.std([1.0, 3.0], ddof=1) / np.sqrt(2), NAN, NAN])
    np.testing.assert_array_equal(summary.by_mouse["mB"], [3.0, NAN, 5.0])
    np.testing.assert_array_equal(summary.by_mouse["mA"], [1.0, 3.0, NAN])


def test_summary_session_without_finite_values_is_nan():
    data = {"m1": {S1: 1.0, S2: None}, "m2": {S1: 2.0, S2: np.inf}}
    summary = summarize_pokes_by_session(data)
    assert summary.mean[0] == 1.5 and np.isnan(summary.mean[1]) and np.isnan(summary.se[1])


def test_summary_requires_mice_and_sessions():
    with pytest.raises(ValueError, match="has no mice"):
        summarize_pokes_by_session({"average_across_subjects": {S1: 1.0}})
    with pytest.raises(ValueError, match="No sessions found"):
        summarize_pokes_by_session({"m": {}})


# ---------------------------------------------------------------- figure


def test_bars_show_mean_with_se_error_bars(capture):
    fig = draw(capture)
    (ax,) = fig.axes
    assert tuple(fig.get_size_inches()) == (12, 5.5)
    bars = ax.patches
    assert [b.get_height() for b in bars] == [2.0, 3.0, 5.0]
    first = bars[0]
    assert first.get_edgecolor() == pytest.approx((0, 0, 0, 0.55))
    assert first.get_facecolor() == pytest.approx((0.6, 0.6, 0.6, 0.55))
    assert first.get_linewidth() == 1.5 and first.get_zorder() == 1
    (errorbar,) = [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
    segments = errorbar.lines[2][0].get_segments()
    # SE of [1, 3] is 1.0; sessions with one value have a NaN SE, so only their bar is drawn.
    assert segments[0][:, 1].tolist() == [1.0, 3.0]
    assert [len(seg) for seg in segments[1:]] == [0, 0]


def test_bar_value_labels_sit_at_y_text(capture):
    fig = draw(capture, y_text=7)
    (ax,) = fig.axes
    texts = [(t.get_text(), t.get_position(), t.get_fontsize()) for t in ax.texts]
    assert texts == [("2.00", (0, 7), 12.0), ("3.00", (1, 7), 12.0), ("5.00", (2, 7), 12.0)]
    assert all(t.get_va() == "bottom" and t.get_ha() == "center" for t in ax.texts)
    assert ax.get_ylim() == (0, 7)


def test_mouse_lines_are_jittered_and_coloured_in_sorted_order(capture):
    fig = draw(capture)
    (ax,) = fig.axes
    jitter = np.random.Generator(np.random.PCG64(0))
    expected = {
        "mA": ([0, 1], [1.0, 3.0], MOUSE_COLORS[0]),
        "mB": ([0, 2], [3.0, 5.0], MOUSE_COLORS[1]),
    }
    for line, (mouse, (xs, ys, color)) in zip(mouse_lines(ax), expected.items(), strict=True):
        np.testing.assert_allclose(line.get_xdata(), xs + jitter.uniform(-0.03, 0.03, len(xs)))
        np.testing.assert_array_equal(line.get_ydata(), ys)
        assert line.get_color() == color, mouse
        assert (line.get_linewidth(), line.get_alpha(), line.get_marker()) == (2.5, 0.9, "o")
        assert line.get_markersize() == 6 and line.get_zorder() == 4


def test_legend_lists_mice_upper_right(capture):
    fig = draw(capture)
    (ax,) = fig.axes
    legend = ax.get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["mA", "mB"]
    assert legend._loc == 1 and legend.get_frame_on()
    assert [h.get_color() for h in legend.legend_handles] == MOUSE_COLORS[:2]


def test_axis_text_and_ticks(capture):
    fig = draw(capture)
    (ax,) = fig.axes
    assert ax.get_xlabel() == "Session" and ax.get_ylabel() == "Non-Choice Long Pokes"
    assert list(ax.get_xticks()) == [0, 1, 2]
    labels = ax.get_xticklabels()
    assert [t.get_text() for t in labels] == ["ses-01", "ses-02", "ses-10"]
    assert all(t.get_rotation() == 45 and t.get_fontsize() == 12 for t in labels)
    assert ax.get_title() == (
        "Mean Non-Choice Long Pokes Across Sessions\n (mean ± se across subjects | n=2 subjects)"
    )
    assert ax.title.get_fontsize() == 16
    assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()


def test_mouse_without_finite_data_has_no_line_or_legend_entry(capture):
    data = {"m1": {S1: 1.0}, "m2": {S1: NAN}}
    fig = draw(capture, data)
    (ax,) = fig.axes
    assert len(mouse_lines(ax)) == 1
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["m1"]
    assert "n=2 subjects" in ax.get_title()


def test_more_mice_than_colours_cycle_the_palette(capture):
    data = {f"m{i:02d}": {S1: float(i)} for i in range(len(MOUSE_COLORS) + 2)}
    fig = draw(capture, data)
    (ax,) = fig.axes
    colors = [ln.get_color() for ln in mouse_lines(ax)]
    assert colors[len(MOUSE_COLORS)] == MOUSE_COLORS[0]
    assert colors[len(MOUSE_COLORS) + 1] == MOUSE_COLORS[1]


def test_show_when_no_path_and_figure_closed(capture):
    fig = draw(capture)
    assert len(capture.shown) == 1 and capture.saved == []
    assert not plt.fignum_exists(fig.number)


def test_save_path_writes_pdf_and_png_without_creating_directories(capture, tmp_path):
    base = tmp_path / "Pokes"
    plot_num_pokes_across_mice(POKES, 6, save_path=base)
    pdf, png = capture.saved
    assert pdf.path == str(base) + ".pdf" and pdf.kwargs == {"bbox_inches": "tight"}
    assert png.path == str(base) + ".png" and png.kwargs == {"dpi": 300, "bbox_inches": "tight"}
    assert capture.shown == []


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="has no mice"):
        plot_num_pokes_across_mice({}, 5)


def test_layout_is_tightened_and_padding_and_caps_are_kept(capture):
    fig = draw(capture)
    (ax,) = fig.axes
    assert was_tight_laid_out(fig)
    assert title_pad_points(ax) == pytest.approx(25)
    (bars,) = [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
    assert cap_size(bars) == 6


def test_panel_draws_into_the_given_axes_without_opening_a_figure(capture):
    fig, axes = plt.subplots(1, 2)
    ax = plot_num_pokes_panel(POKES, 6, ax=axes[1])
    assert ax is axes[1]
    assert plt.get_fignums() == [fig.number] and len(fig.axes) == 2
    assert len(axes[0].patches) == 0 and len(axes[1].patches) == 3
    assert capture.shown == [] and capture.saved == []


def test_panel_creates_a_12_by_5_5_figure_when_no_axes_are_given(capture):
    ax = plot_num_pokes_panel(POKES, 6)
    assert plt.get_fignums() == [ax.figure.number]
    assert tuple(ax.figure.get_size_inches()) == (12, 5.5)


def test_panel_jitter_comes_from_the_given_generator(capture):
    _, ax = plt.subplots()
    plot_num_pokes_panel(POKES, 6, ax=ax, rng=np.random.Generator(np.random.PCG64(7)))
    jitter = np.random.Generator(np.random.PCG64(7))
    for line, xs in zip(mouse_lines(ax), ([0, 1], [0, 2]), strict=True):
        np.testing.assert_allclose(line.get_xdata(), xs + jitter.uniform(-0.03, 0.03, 2))
