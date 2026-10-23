"""Legends are tidied when a figure is saved: unframed, outside the data, drawn once, title case."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.behavior_visualization.plot_helpers import save_or_show
from src.behavior_visualization.plot_legends import HANDLE_LENGTH, tidy_legends, title_case


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def all_legends(fig):
    return [*fig.legends, *(ax.get_legend() for ax in fig.axes if ax.get_legend())]


def boxes(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    return renderer, [leg.get_window_extent(renderer) for leg in all_legends(fig)]


def panel(ax, labels, n_points=20):
    x = np.arange(n_points)
    for i, label in enumerate(labels):
        ax.plot(x, np.sin(x / 3 + i), label=label)
    ax.legend()


# --- title case ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Prev Best magnitude", "Prev Best Magnitude"),
        ("New Third magnitude", "New Third Magnitude"),
        ("Total (mean of observed mice)", "Total (Mean of Observed Mice)"),
        ("pre window (counted)", "Pre Window (Counted)"),
        ("Still running", "Still Running"),
        ("Mean ± se across Mice", "Mean ± se across Mice"),
        ("P(best)", "P(best)"),
        ("MY_175_L (n=10 trials)", "MY_175_L (n=10 Trials)"),
        ("t=0", "t=0"),
        ("non-match", "Non-Match"),
        ("choice: Best", "Choice: Best"),
    ],
)
def test_labels_are_title_case_with_small_words_and_statistics_lower_case(label, expected):
    assert title_case(label) == expected


# --- style and placement ------------------------------------------------------------------------


def test_legends_have_no_frame_short_handles_and_title_case_labels():
    fig, ax = plt.subplots()
    panel(ax, ["prev best", "prev second"])
    tidy_legends(fig)
    (legend,) = all_legends(fig)
    assert not legend.get_frame_on()
    assert legend.handlelength == HANDLE_LENGTH
    assert [t.get_text() for t in legend.get_texts()] == ["Prev Best", "Prev Second"]


def test_a_short_legend_inside_the_axes_moves_below_them():
    fig, ax = plt.subplots()
    ax.set_xlabel("Trials from Block Change")
    panel(ax, ["a", "b"])
    tidy_legends(fig)
    renderer, (box,) = boxes(fig)
    assert box.y1 <= ax.get_tightbbox(renderer).y0 + 1


def test_a_long_mouse_list_goes_beside_the_axes():
    fig, ax = plt.subplots()
    panel(ax, [f"MY_{i}_A" for i in range(13)])
    tidy_legends(fig)
    renderer, (box,) = boxes(fig)
    assert box.x0 >= ax.get_tightbbox(renderer).x1 - 1


def test_a_legend_already_outside_the_axes_stays_where_it_is():
    fig, ax = plt.subplots()
    x = np.arange(5)
    ax.plot(x, x, label="a")
    ax.plot(x, -x, label="b")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.canvas.draw()
    before = ax.get_legend().get_window_extent()
    tidy_legends(fig)
    _, (after,) = boxes(fig)
    assert after.x0 == pytest.approx(before.x0, abs=1)
    assert after.y1 == pytest.approx(before.y1, abs=1)


def test_no_legend_covers_a_panel_of_a_grid():
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for i, ax in enumerate(axes.flat):
        panel(ax, [f"block {j}" for j in range(i + 1)] + [f"MY_{k}_A" for k in range(2)])
    fig.tight_layout()
    tidy_legends(fig)
    renderer, legend_boxes = boxes(fig)
    assert legend_boxes
    for box in legend_boxes:
        for ax in axes.flat:
            assert not box.overlaps(ax.bbox), ax


def test_a_legend_repeated_on_several_panels_is_drawn_once():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax in axes:
        panel(ax, ["P(best)", "P(second)"])
    fig.tight_layout()
    tidy_legends(fig)
    legends = all_legends(fig)
    assert len(legends) == 1
    assert [t.get_text() for t in legends[0].get_texts()] == ["P(best)", "P(second)"]


def test_a_single_mouse_entry_goes_to_the_caption():
    fig, ax = plt.subplots()
    panel(ax, ["MY_175_L", "mean"])
    notes = tidy_legends(fig)
    (legend,) = all_legends(fig)
    assert [t.get_text() for t in legend.get_texts()] == ["Mean"]
    assert notes == ["line: MY_175_L"]


def test_two_mice_keep_their_entries():
    fig, ax = plt.subplots()
    panel(ax, ["MY_1_A", "MY_2_A"])
    assert tidy_legends(fig) == []
    (legend,) = all_legends(fig)
    assert [t.get_text() for t in legend.get_texts()] == ["MY_1_A", "MY_2_A"]


def test_saving_tidies_the_legends_and_writes_the_single_mouse_note(tmp_path):
    fig, ax = plt.subplots()
    panel(ax, ["MY_175_L", "mean"])
    save_or_show(fig, tmp_path / "f")
    assert (tmp_path / "f.caption.txt").read_text() == "line: MY_175_L\n"
    assert not all_legends(fig)[0].get_frame_on()
