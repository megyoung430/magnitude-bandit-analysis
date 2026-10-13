import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.behavior_visualization.plot_rank_proportions import (
    plot_average_across_blocks,
    plot_average_across_mice,
    plot_block_panel,
    plot_mouse_panel,
    plot_rank_proportions,
    plot_rank_proportions_average,
    plot_rank_proportions_by_block,
    plot_rank_proportions_by_mouse,
)


def make_row(best, second, third):
    total = best + second + third
    return {
        "best": best,
        "second": second,
        "third": third,
        "total": total,
        "best_prop": best / total,
        "second_prop": second / total,
        "third_prop": third / total,
    }


@pytest.fixture
def counts():
    return {
        "m1": [make_row(6, 3, 1), make_row(2, 2, 6), make_row(1, 1, 1)],
        "m2": [make_row(4, 4, 2), make_row(1, 2, 7), make_row(9, 1, 1), make_row(1, 1, 1)],
        "m3": [make_row(5, 5, 5), make_row(3, 3, 4), make_row(1, 1, 1)],
    }


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def data_lines(ax):
    # error-bar caps and stems are lines too; the data lines are the marked ones
    return [line for line in ax.lines if line.get_marker() == "o"]


def bar_heights(ax):
    return [p.get_height() for p in ax.patches]


def test_panels_draw_into_given_ax_without_new_figure(counts):
    fig, axes = plt.subplots(1, 4)
    plot_average_across_mice(counts, ax=axes[0])
    plot_average_across_blocks(counts, ax=axes[1])
    plot_mouse_panel("m1", counts["m1"][:-1], ax=axes[2])
    plot_block_panel(counts, 0, ax=axes[3])
    assert plt.get_fignums() == [fig.number]
    assert all(len(ax.patches) == 3 for ax in axes)


def test_panels_create_a_figure_when_no_ax_is_given(counts):
    ax = plot_average_across_mice(counts)
    assert ax.figure.number in plt.get_fignums()


def test_average_across_mice_bars_are_mouse_means(counts):
    ax = plot_average_across_mice(counts)
    expected_best = np.mean(
        [
            np.mean([0.6, 0.2, 1 / 3]),
            np.mean([0.4, 0.1, 9 / 11, 1 / 3]),
            np.mean([1 / 3, 0.3, 1 / 3]),
        ]
    )
    assert bar_heights(ax)[0] == pytest.approx(expected_best)
    assert [line.get_label() for line in ax.get_legend().get_lines()] == ["m1", "m2", "m3"]


def test_average_across_mice_annotates_pvalues_only_when_given(counts):
    pvalues = {"best_vs_second": 0.012, "best_vs_third": 0.0004, "second_vs_third": 0.5}
    texts = [t.get_text() for t in plot_average_across_mice(counts, pvalues).texts]
    assert "p(Best vs Second):\n0.012" in texts
    assert "p(Best vs Third):\n0.000" in texts
    assert "p(Second vs Third):\n0.500" in texts
    assert len(plot_average_across_mice(counts).texts) == 3


def test_average_across_mice_title_names_the_aggregation(counts):
    assert plot_average_across_mice(counts).get_title() == "Average Across Mice"
    pooled = plot_average_across_mice(counts, pool_trials_within_subject=True)
    assert pooled.get_title() == "Pooled Trials Within Mouse"


def test_average_across_blocks_has_one_line_per_block(counts):
    ax = plot_average_across_blocks(counts)
    assert len(data_lines(ax)) == 3
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Block 1", "Block 2", "Block 3"]


def test_mouse_panel_draws_one_line_per_row_and_labels_blocks(counts):
    ax = plot_mouse_panel("m2", counts["m2"][:-1])
    assert len(data_lines(ax)) == 3
    assert ax.get_title() == "m2"
    assert ax.get_ylabel() == "Proportion of Choices"
    assert plot_mouse_panel("m2", counts["m2"][:-1], show_ylabel=False).get_ylabel() == ""


def test_block_panel_legend_reports_trial_counts(counts):
    ax = plot_block_panel(counts, 1)
    assert ax.get_title() == "Block 2"
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        "m1 (n=10 trials)",
        "m2 (n=10 trials)",
        "m3 (n=10 trials)",
    ]
    # bars average the subjects' rows for this block, excluding each subject's last row
    assert bar_heights(ax)[0] == pytest.approx(np.mean([0.2, 0.1, 0.3]))


def test_save_path_writes_pdf_and_png_and_creates_directories(counts, tmp_path):
    base = tmp_path / "nested" / "RP"
    plot_rank_proportions(counts, save_path=str(base))
    for view in ("by Mouse", "by Block", "by Average"):
        assert (tmp_path / "nested" / f"RP {view}.pdf").exists()
        assert (tmp_path / "nested" / f"RP {view}.png").exists()
    assert plt.get_fignums() == []


def test_public_plots_show_and_close_when_no_save_path(counts, monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(len(plt.get_fignums())))
    plot_rank_proportions_average(counts)
    plot_rank_proportions_by_mouse(counts)
    plot_rank_proportions_by_block(counts)
    assert shown == [1, 1, 1]
    assert plt.get_fignums() == []


def test_by_block_without_blocks_warns_and_draws_nothing(capsys):
    plot_rank_proportions_by_block({"m1": [make_row(1, 1, 1)]})
    assert "[WARN] No blocks found to plot." in capsys.readouterr().out
    assert plt.get_fignums() == []


def test_by_mouse_turns_off_unused_axes(counts, monkeypatch):
    figs = []
    monkeypatch.setattr(plt, "show", lambda: figs.append(plt.gcf()))
    plot_rank_proportions_by_mouse(counts)
    axes = figs[0].axes
    assert len(axes) == 4  # 2 rows x ceil(3 / 2) columns
    assert [ax.axison for ax in axes] == [True, True, True, False]
