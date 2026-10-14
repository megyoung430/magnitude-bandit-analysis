import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.container import ErrorbarContainer

from src.behavior_analysis.first_leave_fractions import first_leave_fractions
from src.behavior_visualization.plot_first_leave_after_good_reversals import (
    plot_first_leave_after_good_reversals,
)
from src.behavior_visualization.plot_style import MOUSE_COLORS
from tests.plot_capture import capture_figures

MEAN = {"new_best": 0.7, "third": 0.3, "num_reversals": 20}
SE = {"new_best": 0.1, "third": 0.05}
COUNTS = {
    "s1": {"new_best": 3, "third": 1, "total": 4},
    "empty": {"new_best": 0, "third": 0, "total": 0},
    "s2": {"new_best": 5, "third": 2, "total": 8},
}


@pytest.fixture
def capture(monkeypatch):
    return capture_figures(monkeypatch)


def draw(capture, counts=COUNTS, **kwargs):
    plot_first_leave_after_good_reversals(MEAN, SE, counts, **kwargs)
    (fig,) = (capture.shown or [[s.figure] for s in capture.saved])[-1]
    return fig, fig.axes[0]


def subject_lines(ax):
    return [ln for ln in ax.lines if ln.get_marker() == "o"]


# ---------------------------------------------------------------- analysis


def test_first_leave_fractions_divide_by_total_and_skip_empty_subjects():
    assert first_leave_fractions({"new_best": 3, "third": 1, "total": 4}) == (0.75, 0.25)
    assert first_leave_fractions({"new_best": 0, "third": 0, "total": 0}) is None


# ---------------------------------------------------------------- figure


def test_bars_show_means_with_se_and_value_labels(capture):
    fig, ax = draw(capture)
    assert tuple(fig.get_size_inches()) == (6.5, 5.5) and len(fig.axes) == 1
    assert [b.get_height() for b in ax.patches] == [0.7, 0.3]
    assert all(b.get_facecolor() == pytest.approx((0.6, 0.6, 0.6, 0.55)) for b in ax.patches)
    assert all(b.get_linewidth() == 1.5 for b in ax.patches)
    (errorbar,) = [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
    segments = errorbar.lines[2][0].get_segments()
    assert segments[0][:, 1].tolist() == pytest.approx([0.6, 0.8])
    assert segments[1][:, 1].tolist() == pytest.approx([0.25, 0.35])
    assert [(t.get_text(), t.get_position()) for t in ax.texts] == [
        ("0.70", (0, 1.05)),
        ("0.30", (1, 1.05)),
    ]
    assert all(t.get_fontsize() == 12 and t.get_va() == "bottom" for t in ax.texts)


def test_subject_lines_run_from_new_best_to_third_with_jitter(capture):
    np.random.seed(0)
    jitter = np.random.RandomState(0)
    _, ax = draw(capture)
    s1, s2 = subject_lines(ax)
    for line, (new_best, third), color in [
        (s1, (0.75, 0.25), MOUSE_COLORS[0]),
        (s2, (0.625, 0.25), MOUSE_COLORS[2]),  # "empty" holds index 1 in the palette
    ]:
        x0 = 0 + jitter.uniform(-0.03, 0.03)
        x1 = 1 + jitter.uniform(-0.03, 0.03)
        np.testing.assert_allclose(line.get_xdata(), [x0, x1])
        assert line.get_ydata().tolist() == [new_best, third]
        assert line.get_color() == color
        assert (line.get_linewidth(), line.get_alpha(), line.get_marker()) == (2.5, 0.9, "o")
        assert line.get_markersize() == 6


def test_legend_gives_each_subject_its_reversal_count(capture):
    _, ax = draw(capture)
    legend = ax.get_legend()
    assert [t.get_text() for t in legend.get_texts()] == [
        "s1 (n=4 reversals)",
        "s2 (n=8 reversals)",
    ]
    assert legend._loc == 1 and legend.get_texts()[0].get_fontsize() == 10


def test_axis_text_limits_and_title(capture):
    _, ax = draw(capture)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["New Best", "Third Arm"]
    assert all(t.get_fontsize() == 12 for t in ax.get_xticklabels())
    assert ax.get_ylabel() == "Fraction of Good Reversals" and ax.yaxis.label.get_fontsize() == 12
    assert ax.get_ylim() == (0.0, 1.07)
    assert ax.get_title() == (
        "First Choice After Leaving Previous Best\n"
        "(mean ± se across subjects | n=3 subjects and n=20 reversals)"
    )
    assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()


def test_p_value_annotation_has_three_decimals_and_is_optional(capture):
    _, ax = draw(capture, p_value=0.01234)
    texts = [(t.get_text(), t.get_position()) for t in ax.texts]
    assert ("p-value: 0.012", (0.5, 1.02)) in texts and len(texts) == 3
    _, ax = draw(capture)
    assert len(ax.texts) == 2
    _, ax = draw(capture, p_value=0.0)
    assert "p-value: 0.000" in [t.get_text() for t in ax.texts]


def test_show_without_save_path_closes_the_figure(capture):
    fig, _ = draw(capture)
    assert len(capture.shown) == 1 and capture.saved == []
    assert not plt.fignum_exists(fig.number)


def test_save_path_creates_parent_and_writes_pdf_then_png(capture, tmp_path):
    base = tmp_path / "nested" / "First Leave"
    plot_first_leave_after_good_reversals(MEAN, SE, COUNTS, 0.5, save_path=base)
    pdf, png = capture.saved
    assert pdf.path == str(base) + ".pdf" and pdf.kwargs == {"bbox_inches": "tight"}
    assert png.path == str(base) + ".png" and png.kwargs == {"dpi": 300, "bbox_inches": "tight"}
    assert capture.shown == [] and base.parent.is_dir()


def test_files_are_really_written(monkeypatch, tmp_path):
    capture_figures(monkeypatch, write_files=True)
    base = tmp_path / "out" / "First Leave"
    plot_first_leave_after_good_reversals(MEAN, SE, COUNTS, save_path=base)
    assert (tmp_path / "out" / "First Leave.pdf").stat().st_size > 0
    assert (tmp_path / "out" / "First Leave.png").stat().st_size > 0


def test_no_subjects_draws_bars_only(capture):
    _, ax = draw(capture, counts={})
    assert subject_lines(ax) == [] and "n=0 subjects" in ax.get_title()
