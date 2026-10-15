import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.behavior_analysis.average_choice_probs import average_across_problem_curves
from src.behavior_visualization import plot_averaged_choice_probs as module
from src.behavior_visualization.plot_averaged_choice_probs import (
    fraction_removed_by_problem,
    plot_choice_probs_with_fraction_removed_axis,
    plot_choice_probs_with_fraction_removed_panel,
)
from src.behavior_visualization.plot_style import CHOICE_PROB_COLOR_MAP
from tests.plot_capture import (
    capture_figures,
    lines_by_label,
    title_pad_points,
    was_tight_laid_out,
)

X = np.array([-1.0, 0.0, 1.0])
NAN = np.nan


def problem_curve(prev, nxt, third, subjects, reversals):
    return dict(
        x=X.copy(),
        across=dict(mean={"prev_best": prev, "next_best": nxt, "third": third}),
        subjects=set(subjects),
        num_reversals=reversals,
    )


@pytest.fixture
def three_problems():
    return {
        3: problem_curve([0.6, 0.5, 0.4], [0.3, 0.3, 0.3], [0.1, 0.2, 0.3], {"a", "b"}, 5),
        1: problem_curve([0.4, 0.3, NAN], [0.4, 0.4, NAN], [0.2, 0.3, NAN], {"b", "c"}, 4),
        2: problem_curve([0.5, 0.4, 0.2], [0.3, 0.3, 0.4], [0.2, 0.3, 0.4], {"a"}, None),
    }


# ---------------------------------------------------------------- average_across_problem_curves


def test_average_is_nanmean_over_problems_sorted_by_problem(three_problems):
    x, avg = average_across_problem_curves(three_problems)
    np.testing.assert_array_equal(x, X)
    np.testing.assert_allclose(avg["mean"]["prev_best"], [0.5, 0.4, 0.3])
    assert avg["problems_used"] == [1, 2, 3]
    assert avg["num_problems"] == 3


def test_average_counts_union_of_subjects_and_sums_reversals(three_problems):
    _, avg = average_across_problem_curves(three_problems)
    assert avg["num_subjects"] == 3  # a, b, c
    assert avg["num_reversals"] == 9  # 4 + 0 (None) + 5


def test_average_se_divides_by_the_number_of_problems_with_data_at_each_offset(three_problems):
    # At the last offset problem 1 is NaN, so only two problems enter the SE there.
    _, avg = average_across_problem_curves(three_problems)
    se = avg["se"]["prev_best"]
    full = np.std([0.4, 0.5, 0.6], ddof=1) / np.sqrt(3)
    np.testing.assert_allclose(se[:2], [full, np.std([0.3, 0.4, 0.5], ddof=1) / np.sqrt(3)])
    assert se[2] == pytest.approx(np.std([0.2, 0.4], ddof=1) / np.sqrt(2))


def test_average_se_is_nan_where_fewer_than_two_problems_have_data(three_problems):
    three_problems[2]["across"]["mean"]["prev_best"] = [0.5, 0.4, NAN]
    _, avg = average_across_problem_curves(three_problems)
    assert avg["mean"]["prev_best"][2] == 0.4  # only problem 3 has a value there
    assert np.isnan(avg["se"]["prev_best"][2])
    assert np.isfinite(avg["se"]["prev_best"][:2]).all()


def test_num_problems_counts_only_problems_that_contribute_a_curve(three_problems):
    three_problems[2]["across"]["mean"] = {}  # problem 2 yields no curve at all
    _, avg = average_across_problem_curves(three_problems)
    assert avg["num_problems"] == 2
    assert avg["problems_used"] == [1, 3]
    np.testing.assert_allclose(avg["mean"]["prev_best"], [0.5, 0.4, 0.4])


def test_num_problems_is_zero_when_no_problem_has_a_curve():
    curves = {1: problem_curve([0.5] * 3, [0.3] * 3, [0.2] * 3, {"a"}, 1)}
    curves[1]["across"]["mean"] = {}
    _, avg = average_across_problem_curves(curves)
    assert avg["num_problems"] == 0 and avg["problems_used"] == []


def test_average_single_problem_has_nan_se():
    _, avg = average_across_problem_curves(
        {7: problem_curve([0.6, 0.5, 0.4], [0.3, 0.3, 0.3], [0.1, 0.2, 0.3], {"a"}, 2)}
    )
    assert np.isnan(avg["se"]["third"]).all()
    np.testing.assert_allclose(avg["mean"]["third"], [0.1, 0.2, 0.3])


def test_average_key_missing_from_every_problem_is_none():
    curves = {1: problem_curve([0.5] * 3, [0.3] * 3, [0.2] * 3, {"a"}, 1)}
    del curves[1]["across"]["mean"]["third"]
    _, avg = average_across_problem_curves(curves)
    assert avg["mean"]["third"] is None and avg["se"]["third"] is None
    assert avg["mean"]["prev_best"] is not None


def test_average_rejects_empty_input_and_mismatched_x(three_problems):
    with pytest.raises(ValueError, match="No problems to average."):
        average_across_problem_curves({})
    three_problems[2]["x"] = X + 1
    with pytest.raises(ValueError, match="x-array mismatch between problems 1 and 2"):
        average_across_problem_curves(three_problems)


# ---------------------------------------------------------------- the figure

AX = np.arange(-2, 3, dtype=float)
ACROSS = dict(
    mean=dict(
        prev_best=np.array([0.6, 0.5, 0.2, 0.1, 0.1]),
        next_best=np.array([0.2, 0.3, 0.5, 0.7, 0.8]),
        third=np.array([0.2, 0.2, 0.3, 0.2, 0.1]),
    ),
    se=dict(prev_best=np.full(5, 0.05), next_best=np.full(5, 0.04), third=np.full(5, 0.03)),
    num_subjects=4,
    num_reversals=21,
    num_problems=3,
)
FRAC = np.array([NAN, NAN, 0.0, 0.25, 0.5])


@pytest.fixture
def capture(monkeypatch):
    return capture_figures(monkeypatch)


def draw(capture, **kwargs):
    plot_choice_probs_with_fraction_removed_axis(AX, ACROSS, FRAC, **kwargs)
    assert len(capture.shown) + len(capture.saved) > 0
    figure = (capture.shown[0] if capture.shown else [capture.saved[0].figure])[0]
    left, right = figure.axes
    return figure, left, right


def test_show_when_no_save_path_and_figure_is_closed(capture):
    fig, _, _ = draw(capture)
    assert len(capture.shown) == 1 and capture.saved == []
    assert not plt.fignum_exists(fig.number)


def test_figure_layout_and_labels(capture):
    fig, left, right = draw(capture)
    assert tuple(fig.get_size_inches()) == (10, 6)
    assert len(fig.axes) == 2 and right.get_shared_x_axes().joined(left, right)
    assert left.get_xlabel() == "Trials from Good Reversal"
    assert left.get_ylabel() == "Choice Probability"
    assert right.get_ylabel() == "Fraction of Data Removed"
    assert right.yaxis.label.get_rotation() == 270  # rotation=-90
    assert left.get_ylim() == (0, 1) and right.get_ylim() == (0, 1)
    assert left.get_xlim() == (-2, 2)
    assert not left.spines["top"].get_visible() and not left.spines["right"].get_visible()
    assert not right.spines["top"].get_visible()
    assert left.get_title() == (
        "Reversal-Aligned Choices\n(mean ± se across problems | "
        "n=4 subjects, n=21 reversals, n=3 problems)\n"
    )


def test_choice_curves_use_arm_colours_and_labels(capture):
    _, left, _ = draw(capture)
    expected = {
        "Previous Best": ("prev_best", ACROSS["mean"]["prev_best"]),
        "New Best": ("next_best", ACROSS["mean"]["next_best"]),
        "Third Arm": ("third", ACROSS["mean"]["third"]),
    }
    for label, (key, y) in expected.items():
        (line,) = lines_by_label(left, label)
        assert line.get_color() == CHOICE_PROB_COLOR_MAP[key]
        assert line.get_linewidth() == 2
        np.testing.assert_array_equal(line.get_xdata(), AX)
        np.testing.assert_allclose(line.get_ydata(), y)
    bands = left.collections
    assert len(bands) == 3 and all(b.get_alpha() == 0.2 for b in bands)
    prev_band = bands[0].get_paths()[0].vertices
    assert prev_band[:, 1].max() == pytest.approx(0.65) and prev_band[:, 1].min() == pytest.approx(
        0.05
    )


def test_reference_lines_and_fraction_step(capture):
    _, left, right = draw(capture)
    vline = [ln for ln in left.lines if list(ln.get_xdata()) == [0, 0]][0]
    assert (vline.get_color(), vline.get_linestyle(), vline.get_linewidth()) == ("black", "--", 1)
    hline = [ln for ln in left.lines if list(ln.get_ydata()) == [1 / 3, 1 / 3]][0]
    assert (hline.get_color(), hline.get_linestyle(), hline.get_linewidth()) == ("gray", ":", 1)
    (step,) = right.lines
    assert step.get_color() == "#FF9100" and step.get_alpha() == 0.7
    assert step.get_linewidth() == 2.0 and step.get_drawstyle() == "steps-post"
    assert step.get_label() == "Total Rev"
    np.testing.assert_array_equal(step.get_xdata(), AX[2:])
    np.testing.assert_array_equal(step.get_ydata(), FRAC[2:])


def test_legend_merges_both_axes(capture):
    _, left, _ = draw(capture)
    legend = left.get_legend()
    assert [t.get_text() for t in legend.get_texts()] == [
        "Previous Best",
        "New Best",
        "Third Arm",
        "Total Rev",
    ]
    assert legend._loc == 2  # upper left
    assert legend.get_texts()[0].get_fontsize() == 10


def test_title_extra_is_appended(capture):
    _, left, _ = draw(capture, title_extra="Early reversals")
    assert left.get_title().endswith("n=3 problems)\nEarly reversals")


def test_all_nan_curve_is_not_drawn_and_missing_fraction_has_no_step(capture):
    across = {**ACROSS, "mean": {**ACROSS["mean"], "third": np.full(5, NAN)}}
    plot_choice_probs_with_fraction_removed_axis(AX, across, np.full(5, NAN))
    left, right = capture.shown[0][0].axes
    assert [ln.get_label() for ln in left.lines if ln.get_label()[0] != "_"] == [
        "Previous Best",
        "New Best",
    ]
    assert len(right.lines) == 0


def test_save_path_writes_pdf_then_png_and_creates_parent(capture, tmp_path):
    base = tmp_path / "new_dir" / "Choice Probabilities"
    plot_choice_probs_with_fraction_removed_axis(AX, ACROSS, FRAC, save_path=base)
    assert capture.shown == []
    pdf, png = capture.saved
    assert pdf.path == str(base) + ".pdf" and pdf.kwargs == {"bbox_inches": "tight"}
    assert png.path == str(base) + ".png" and png.kwargs == {"dpi": 300, "bbox_inches": "tight"}
    assert base.parent.is_dir()


def test_average_is_reexported_from_the_plotting_module():
    assert module.average_across_problem_curves is average_across_problem_curves


def test_layout_is_tightened_and_title_and_label_padding_are_kept(capture):
    fig, left, right = draw(capture)
    assert was_tight_laid_out(fig)
    assert title_pad_points(left) == pytest.approx(20)
    assert right.yaxis.labelpad == 15


def test_panel_draws_into_the_given_axes_without_opening_a_figure(capture):
    fig, axes = plt.subplots(1, 2)
    left = plot_choice_probs_with_fraction_removed_panel(AX, ACROSS, FRAC, ax=axes[0])
    assert left is axes[0]
    assert plt.get_fignums() == [fig.number] and len(fig.axes) == 3  # two panels + the twin axis
    assert len(axes[1].lines) == 0  # the neighbouring panel is untouched
    assert [t.get_text() for t in left.get_legend().get_texts()][-1] == "Total Rev"
    assert capture.shown == [] and capture.saved == []


def test_panel_creates_a_10_by_6_figure_when_no_axes_are_given(capture):
    left = plot_choice_probs_with_fraction_removed_panel(AX, ACROSS, FRAC)
    assert plt.get_fignums() == [left.figure.number]
    assert tuple(left.figure.get_size_inches()) == (10, 6)


# ---------------------------------------------------------------- fraction_removed_by_problem

FX = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])


# Offsets before the reversal are NaN in every problem, so the cross-problem mean warns.
ignore_nan_offsets = pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")


def windows(*reversal_indices):
    """Windows of one subject whose reversals happen at the given trial indices."""
    return {"m": [{"reversal_idx": g} for g in reversal_indices]}


def reversals_of(windows_dict):
    """Per problem, every anchor is a good reversal and there are no others."""
    return {
        p: {
            "good": {s: [r["reversal_idx"] for r in revs] for s, revs in w.items()},
            "bad": {s: [] for s in w},
            "unclassified": {s: [] for s in w},
        }
        for p, w in windows_dict.items()
    }


def removed_with(windows_dict, curves):
    return fraction_removed_by_problem(
        windows_dict, curves, FX, reversals_by_problem=reversals_of(windows_dict)
    )


def removed(windows_dict, **kwargs):
    curves = {p: {"across": {}} for p in windows_dict}
    return fraction_removed_by_problem(
        windows_dict, curves, FX, reversals_by_problem=reversals_of(windows_dict), **kwargs
    )


@ignore_nan_offsets
def test_fraction_removed_is_the_mean_of_per_problem_step_curves():
    # Problem 1: next reversal after 10 is 2 trials away, after 12 it is 8 (beyond FX);
    # problem 2: 4 trials away for 1 of 2 anchors; problem 3: never within FX.
    frac, used = removed({3: windows(10, 30), 1: windows(10, 12, 20), 2: windows(10, 14)})
    assert used == [1, 2, 3]  # problems are visited in sorted order
    expected = [NAN, 0, 0, 1 / 9, 1 / 9, (1 / 3 + 1 / 2) / 3]
    np.testing.assert_allclose(frac, expected)  # a median would give 0 at offsets 2 and 3


@ignore_nan_offsets
def test_fraction_removed_weights_each_problem_equally_whatever_its_anchor_count():
    # Offset 2: problem 1 removes 1 of 2 anchors, problem 2 removes 3 of 5. The mean of the
    # two fractions is 0.55; pooling the anchors would give 4/7.
    frac, _ = removed({1: windows(10, 12), 2: windows(10, 12, 100, 101, 102)})
    np.testing.assert_allclose(frac[3], (1 / 2 + 3 / 5) / 2)


@ignore_nan_offsets
def test_fraction_removed_exclude_anchor_flag_is_forwarded():
    # Counting the anchor itself as its own next reversal removes every anchor at offset 0.
    frac, _ = removed({1: windows(10, 12)}, exclude_anchor=False)
    np.testing.assert_allclose(frac[1:], [1, 1, 1, 1, 1])
    frac, _ = removed({1: windows(10, 12)})
    np.testing.assert_allclose(frac[1:], [0, 0, 1 / 2, 1 / 2, 1 / 2])


@ignore_nan_offsets
def test_fraction_removed_skips_problems_without_curves_or_windows():
    curves = {1: {"across": {}}, 3: {"across": {}}}
    windows_dict = {1: windows(10, 12), 2: windows(10, 12), 3: {}}
    frac, used = removed_with(windows_dict, curves)
    assert used == [1]
    np.testing.assert_allclose(frac[3], 1 / 2)


def test_fraction_removed_is_all_nan_when_no_problem_contributes():
    frac, used = fraction_removed_by_problem(
        {1: {}}, {1: {"across": {}}}, FX, reversals_by_problem={}
    )
    assert used == [] and np.isnan(frac).all() and frac.shape == FX.shape


def raise_while_drawing(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "_plot_choice_curves", boom)


def test_a_figure_created_by_the_panel_is_closed_when_drawing_raises(capture, monkeypatch):
    raise_while_drawing(monkeypatch)
    with pytest.raises(RuntimeError, match="boom"):
        plot_choice_probs_with_fraction_removed_panel(AX, ACROSS, FRAC)
    assert plt.get_fignums() == []


def test_a_figure_given_to_the_panel_stays_open_when_drawing_raises(capture, monkeypatch):
    raise_while_drawing(monkeypatch)
    fig, ax = plt.subplots()
    with pytest.raises(RuntimeError, match="boom"):
        plot_choice_probs_with_fraction_removed_panel(AX, ACROSS, FRAC, ax=ax)
    assert plt.get_fignums() == [fig.number]


def test_the_public_function_closes_its_figure_when_drawing_or_saving_raises(
    capture, monkeypatch, tmp_path
):
    raise_while_drawing(monkeypatch)
    with pytest.raises(RuntimeError, match="boom"):
        plot_choice_probs_with_fraction_removed_axis(AX, ACROSS, FRAC)
    monkeypatch.undo()

    def cannot_save(self, fname, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(plt.Figure, "savefig", cannot_save)
    with pytest.raises(OSError, match="disk full"):
        plot_choice_probs_with_fraction_removed_axis(AX, ACROSS, FRAC, save_path=tmp_path / "x")
    assert plt.get_fignums() == []


@ignore_nan_offsets
def test_fraction_removed_counts_bad_reversals_between_anchors():
    kinds = {"good": {"m": [10]}, "bad": {"m": [12]}, "unclassified": {"m": []}}
    frac, _ = fraction_removed_by_problem(
        {1: windows(10)}, {1: {"across": {}}}, FX, reversals_by_problem={1: kinds}
    )
    np.testing.assert_allclose(frac[1:], [0, 0, 1, 1, 1])


@ignore_nan_offsets
def test_fraction_removed_counts_blocks_only_block_changes():
    # a blocks-only problem: anchors are block changes, none is good or bad
    kinds = {"good": {"m": []}, "bad": {"m": []}, "unclassified": {"m": [10, 12]}}
    frac, _ = fraction_removed_by_problem(
        {1: windows(10, 12)}, {1: {"across": {}}}, FX, reversals_by_problem={1: kinds}
    )
    np.testing.assert_allclose(frac[1:], [0, 0, 1 / 2, 1 / 2, 1 / 2])


@ignore_nan_offsets
def test_fraction_removed_does_not_count_a_problem_without_anchors():
    windows_dict = {1: windows(10, 12), 2: {"m": []}}
    frac, used = removed(windows_dict)
    assert used == [1]
    np.testing.assert_allclose(frac[3], 1 / 2)


def test_fraction_removed_names_a_problem_missing_from_the_reversal_indices():
    windows_dict = {7: windows(10, 12)}
    with pytest.raises(KeyError, match="Problem 7.*reversals_by_problem"):
        fraction_removed_by_problem(windows_dict, {7: {"across": {}}}, FX, reversals_by_problem={})


# ---------------------------------------------------------------- problems that contribute


def test_totals_and_problem_count_cover_only_the_contributing_problems(three_problems):
    three_problems[2]["across"]["mean"] = {}  # problem 2 (subject "a", no reversals) has no curve
    three_problems[3]["across"]["mean"] = {"prev_best": [0.6, 0.5, 0.4]}
    del three_problems[3]["subjects"]
    three_problems[3]["subjects"] = {"d"}
    _, avg = average_across_problem_curves(three_problems)
    assert avg["problems_used"] == [1, 3]
    assert avg["num_subjects"] == 3  # b, c from problem 1 and d from problem 3; "a" is not counted
    assert avg["num_reversals"] == 9  # 4 + 5


def test_a_problem_whose_curves_are_entirely_nan_is_not_used(three_problems):
    three_problems[2]["across"]["mean"] = {
        k: [NAN] * 3 for k in ("prev_best", "next_best", "third")
    }
    _, avg = average_across_problem_curves(three_problems)
    assert avg["problems_used"] == [1, 3]
    assert avg["num_problems"] == 2
    assert avg["num_reversals"] == 9
    assert avg["num_subjects"] == 3  # a, b, c from problems 1 and 3


def test_a_problem_supplying_only_next_best_counts_as_used():
    curves = {
        1: problem_curve([0.5] * 3, [0.3] * 3, [0.2] * 3, {"a"}, 1),
        2: problem_curve([0.5] * 3, [0.3] * 3, [0.2] * 3, {"b"}, 1),
    }
    curves[2]["across"]["mean"] = {"next_best": [0.1, 0.2, 0.3]}
    _, avg = average_across_problem_curves(curves)
    assert avg["problems_used"] == [1, 2] and avg["num_problems"] == 2
    curves[2]["across"]["mean"] = {"third": [0.1, 0.2, 0.3]}
    _, avg = average_across_problem_curves(curves)
    assert avg["problems_used"] == [1, 2]
    np.testing.assert_allclose(avg["mean"]["third"], [0.15, 0.2, 0.25])


# ---------------------------------------------------------------- mean without an SE band


def test_mean_line_is_drawn_where_the_se_is_nan_and_only_the_band_is_left_out():
    se = {k: np.array([NAN, 0.05, 0.05, NAN, NAN]) for k in ACROSS["se"]}
    across = {**ACROSS, "se": se}
    fig, ax = plt.subplots()
    module._plot_choice_curves(ax, AX, across)
    (line,) = lines_by_label(ax, "Previous Best")
    np.testing.assert_array_equal(line.get_xdata(), AX)  # every finite mean is drawn
    band_x = ax.collections[0].get_paths()[0].vertices[:, 0]
    assert band_x.min() == -1 and band_x.max() == 0  # only the two bins with a finite SE
    plt.close(fig)
