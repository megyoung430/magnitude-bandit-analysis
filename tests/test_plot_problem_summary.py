import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.behavior_visualization import plot_problem_summary as pps

nan = float("nan")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def summary():
    return {
        "blocks": [1, 2, 3],
        "means": [2.0, nan, 4.5],
        "ses": [0.5, nan, 0.0],
        "ns": [2, 0, 1],
        "per_mouse_blocklens": {"A": {1: 2.5, 3: 4.5}, "B": {1: 1.5}, "Z": {9: 1.0}},
        "mice": ["A", "B", "C", "Z"],
    }


def block_stats():
    return {
        1: {"block_stats": {"per_mouse_blocklens": {"A": {1: 10.0, 2: nan}, "B": {1: 20.0}}}},
        2: {"block_stats": {"per_mouse_blocklens": {"A": {1: 30.0, 2: 40.0}}}},
    }


# ---- pure helpers ----


def test_mouse_color_map_cycles_through_palette():
    mice = [f"m{i}" for i in range(len(pps.MOUSE_COLORS) + 1)]
    colors = pps.mouse_color_map(mice)
    assert colors[mice[0]] == colors[mice[-1]] == pps.MOUSE_COLORS[0]


def test_concat_nonempty_skips_empty_and_handles_all_empty():
    out = pps.concat_nonempty([np.array([1.0]), np.array([]), np.array([2.0, 3.0])])
    np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])
    assert pps.concat_nonempty([np.array([]), np.array([])]).size == 0


def test_value_range():
    assert pps.value_range(np.array([3.0, 1.0, 2.0])) == (1.0, 3.0)
    assert pps.value_range(np.array([])) is None


def test_annotation_ceiling_default_is_1p5x_bar_top_with_floor():
    means, ses = np.array([4.0, 2.0]), np.array([1.0, 0.0])
    assert pps.annotation_ceiling(means, ses, None) == (5.0, 7.5)
    # tiny bars: the label height is floored at 1 * 1.5
    assert pps.annotation_ceiling(np.array([0.2]), np.array([0.1]), None)[1] == pytest.approx(1.5)
    assert pps.annotation_ceiling(means, ses, 9.0) == (5.0, 9.0)


def test_annotation_ceiling_all_nan_falls_back_to_unit_top():
    nans = np.array([nan, nan])
    assert pps.annotation_ceiling(nans, nans, None) == (1.0, 1.5)


def test_mouse_points_keeps_only_blocks_with_data_at_their_index():
    xs, ys = pps.mouse_points([1, 2, 3], {3: 7.0, 1: 5.0, 9: 1.0})
    np.testing.assert_array_equal(xs, [0, 2])
    np.testing.assert_array_equal(ys, [5.0, 7.0])
    xs, ys = pps.mouse_points([1, 2], {})
    assert xs.size == ys.size == 0


def test_jittered_is_bounded_and_seeded_and_off_at_zero():
    xs = np.arange(5, dtype=float)
    np.testing.assert_array_equal(pps.jittered(xs, 0), xs)
    np.random.seed(0)
    a = pps.jittered(xs, 0.1)
    np.random.seed(0)
    b = pps.jittered(xs, 0.1)
    np.testing.assert_array_equal(a, b)
    assert np.all(np.abs(a - xs) <= 0.1) and not np.array_equal(a, xs)


def test_default_length_title():
    assert pps.default_length_title("Sessions", 3) == (
        "Sessions per Problem\n(mean ± se across mice | n=3 subjects)"
    )


def test_pooled_blocklens_and_title():
    data = {1: {"A": np.array([1.0, 2.0]), "B": np.array([]), "C": np.array([3.0])}}
    pooled = pps.pooled_blocklens(data, 1, ["A", "B", "C"])
    np.testing.assert_array_equal(pooled, [1.0, 2.0, 3.0])
    assert pps.pooled_title(1, pooled, 2) == "Problem 1 (blocks=3, mice=2)"
    assert pps.pooled_title(1, np.array([]), 0) == "Problem 1"


def test_mean_rank_proportions_by_subject_skips_empty_and_nan():
    rows = {
        "A": [
            {"best_prop": 0.5, "second_prop": 0.3, "third_prop": 0.2},
            {"best_prop": 0.7, "second_prop": nan, "third_prop": 0.0},
        ],
        "B": [],
        "C": [{"best_prop": 1.0}],
    }
    out = pps.mean_rank_proportions_by_subject(rows)
    assert list(out) == ["A", "C"]
    assert out["A"] == pytest.approx({"best": 0.6, "second": 0.3, "third": 0.1})
    assert out["C"]["best"] == 1.0 and np.isnan(out["C"]["second"])


def test_rank_mean_and_se_across_mice():
    per_mouse = {
        "A": {"best": 0.6, "second": 0.3, "third": 0.1},
        "B": {"best": 0.8, "second": 0.1, "third": 0.1},
    }
    means, ses = pps.rank_mean_and_se(per_mouse)
    assert means == pytest.approx([0.7, 0.2, 0.1])
    # sd(0.6, 0.8) = 0.1414..., divided by sqrt(2)
    assert ses == pytest.approx([0.1, 0.1, 0.0])


def test_pvalue_annotations_anchor_on_finite_bars_and_skip_missing():
    out = pps.pvalue_annotations(
        {"best_vs_second": 0.01234, "second_vs_third": 0.5}, finite_x=[0, 1, 2]
    )
    assert out == [
        (0, 0.75, "p(Best vs Second):\n0.012"),
        (1, 0.5, "p(Second vs Third):\n0.500"),
    ]
    assert pps.pvalue_annotations(None, [0, 1, 2]) == []
    # best-vs-third needs three finite bars
    assert pps.pvalue_annotations({"best_vs_third": 0.1}, [0, 1]) == []


# ---- panels draw into the axes they are given ----


def test_problem_bars_draws_into_given_axes_with_legend_counts():
    fig, (left, right) = plt.subplots(1, 2)
    out = pps.plot_problem_bars(summary(), left, ylabel="Y", title="T", jitter=0)
    assert out is left
    assert len(fig.axes) == 2 and not right.lines
    labels = [t.get_text() for t in left.get_legend().get_texts()]
    assert labels == ["A (n=2)", "B (n=1)"]
    assert [t.get_text() for t in left.texts] == ["2.00", "4.50"]
    assert left.get_title() == "T" and left.get_ylabel() == "Y"
    mouse_lines = [ln for ln in left.lines if ln.get_marker() == "o"]
    np.testing.assert_array_equal(mouse_lines[0].get_xdata(), [0, 2])


def test_problem_bars_lengths_style_changes_format_and_legend():
    _, ax = plt.subplots()
    pps.plot_problem_bars(summary(), ax, ylabel="Y", title="T", jitter=0, style=pps.LENGTHS_STYLE)
    assert [t.get_text() for t in ax.texts] == ["2.0", "4.5"]
    assert ax.get_legend().get_texts()[0].get_text() == "A (n=2 problems)"


def test_problem_bars_creates_figure_only_without_ax():
    plt.close("all")
    ax = pps.plot_problem_bars(summary(), ylabel="Y", title="T")
    assert len(plt.get_fignums()) == 1 and ax.figure.number in plt.get_fignums()


def test_blocklen_histogram_empty_says_no_data_and_ignores_xlim():
    _, ax = plt.subplots()
    pps.plot_blocklen_histogram(np.array([]), ax, title="Problem 1", xlim=(0, 5))
    assert [t.get_text() for t in ax.texts] == ["no data"]
    assert ax.get_xlim() != (0, 5)


def test_blocklen_histogram_draws_stats_and_xlim():
    _, ax = plt.subplots()
    pps.plot_blocklen_histogram(np.array([10.0, 20.0, 30.0]), ax, title="P", xlim=(0, 50))
    assert ax.get_xlim() == (0, 50) and ax.get_title() == "P"
    assert ax.texts[0].get_text().startswith("n = 3")
    assert len(ax.lines) == 1  # mean line


def test_rank_panel_draws_bars_lines_and_pvalues():
    per_mouse = {
        "A": {"best": 0.6, "second": 0.3, "third": 0.1},
        "B": {"best": 0.8, "second": 0.1, "third": 0.1},
    }
    means, ses = pps.rank_mean_and_se(per_mouse)
    _, ax = plt.subplots()
    out = pps.plot_rank_proportions_average_panel(
        per_mouse, means, ses, ax, pvalues={"best_vs_second": 0.02}, jitter=0
    )
    assert out is ax
    assert len(ax.patches) == 3
    assert sum(ln.get_marker() == "o" for ln in ax.lines) == 2  # one line per mouse
    assert "p(Best vs Second):\n0.020" in [t.get_text() for t in ax.texts]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["A", "B"]


# ---- figure functions ----


def test_distribution_figures_make_one_panel_per_problem(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    problems, mice, data = pps.collect_blocklens(block_stats())
    fig = pps._pooled_histogram_figure(data, problems, mice, ncols=4, sharex=False, sharey=False)
    on = [ax for ax in fig.axes if ax.axison]
    assert [ax.get_title() for ax in on] == [
        "Problem 1 (blocks=2, mice=2)",
        "Problem 2 (blocks=2, mice=1)",
    ]
    assert len(fig.axes) == 4 and not fig.axes[3].axison
    fig = pps._mouse_histogram_figure(data, "B", problems, ncols=2, sharex=True, sharey=False)
    assert [t.get_text() for t in fig.axes[1].texts] == ["no data"]


def test_figure_functions_save_pdf_and_png(tmp_path):
    out = tmp_path / "nested" / "fig"
    pps.plot_metric_by_problem(summary(), title="T", ylabel="Y", save_path=out)
    pps.plot_problem_lengths(summary(), save_path=tmp_path / "len")
    pps.plot_distributions_per_problem_all_mice(block_stats(), save_path=tmp_path / "pooled")
    pps.plot_distributions_per_mouse(block_stats(), save_dir=tmp_path / "mice")
    pps.plot_rank_proportions_average_without_blocks(
        {"A": [{"best_prop": 0.5, "second_prop": 0.3, "third_prop": 0.2}]},
        save_path=tmp_path / "rank",
    )
    for name in ["nested/fig", "len", "pooled", "rank"]:
        assert (tmp_path / f"{name}.pdf").exists() and (tmp_path / f"{name}.png").exists()
    assert (tmp_path / "mice" / "blocklen_distributions_mouse_A.png").exists()
    assert (tmp_path / "mice" / "blocklen_distributions_mouse_B.pdf").exists()
    assert plt.get_fignums() == []


def test_pooled_distribution_with_nothing_left_after_dropping_raises():
    one_problem = {1: {"block_stats": {"per_mouse_blocklens": {"A": {1: 5.0}}}}}
    with pytest.raises(ValueError):
        pps.plot_distributions_per_problem_all_mice(one_problem, drop_last_problem=True)
