import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.behavior_visualization.plot_helpers import (
    FigureOutput,
    TableOutput,
    palette_color,
    write_outputs,
)
from src.behavior_visualization.plot_style import MOUSE_COLORS
from tests.plot_capture import capture_figures, was_tight_laid_out


@pytest.fixture
def capture(monkeypatch):
    return capture_figures(monkeypatch, write_files=True)


def small_figure():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    return fig


def test_palette_color_cycles_past_the_end_of_the_palette():
    assert palette_color(0) == MOUSE_COLORS[0]
    assert palette_color(len(MOUSE_COLORS) + 1) == MOUSE_COLORS[1]


def test_tables_and_figures_are_written_in_order_and_figure_paths_returned(capture, tmp_path):
    frame = pd.DataFrame({"a": [1, 2]}, index=pd.Index([5, 6], name="k"))
    outputs = [
        TableOutput("plain.csv", frame),
        FigureOutput("Fig one", small_figure()),
        TableOutput("indexed.csv", frame, index=True),
    ]
    paths = write_outputs(outputs, tmp_path, show=False)
    assert paths == [tmp_path / "Fig one.png", tmp_path / "Fig one.pdf"]
    assert all(p.is_file() for p in paths)
    assert (tmp_path / "plain.csv").read_text() == "a\n1\n2\n"
    assert (tmp_path / "indexed.csv").read_text() == "k,a\n5,1\n6,2\n"
    assert [s.kwargs for s in capture.saved] == [{"dpi": 180, "bbox_inches": "tight"}] * 2
    assert capture.shown == [] and plt.get_fignums() == []


def test_each_figure_is_tight_laid_out_shown_after_saving_and_closed(capture, tmp_path):
    figures = [small_figure(), small_figure()]
    outputs = [FigureOutput("one", figures[0]), FigureOutput("two", figures[1])]
    write_outputs(outputs, tmp_path, show=True)
    assert all(was_tight_laid_out(fig) for fig in figures)
    assert len(capture.shown) == 2
    # a figure is shown after it is saved but before it is closed, so the first show sees both
    assert [[fig.number for fig in group] for group in capture.shown] == [
        [figures[0].number, figures[1].number],
        [figures[1].number],
    ]
    assert plt.get_fignums() == []


def test_outputs_are_built_only_when_the_previous_one_has_been_written(capture, tmp_path):
    seen = []

    def outputs():
        yield TableOutput("first.csv", pd.DataFrame({"a": [1]}))
        seen.append((tmp_path / "first.csv").is_file())  # written before the next is built
        yield FigureOutput("fig", small_figure())
        seen.append(len(plt.get_fignums()))  # the figure was saved and closed first

    write_outputs(outputs(), tmp_path, show=False)
    assert seen == [True, 0]


def test_a_failing_builder_keeps_earlier_files_and_leaves_its_figure_open(capture, tmp_path):
    def outputs():
        yield TableOutput("first.csv", pd.DataFrame({"a": [1]}))
        yield FigureOutput("done", small_figure())
        small_figure()  # a figure started by a builder that then raises stays open
        raise KeyError("boom")

    with pytest.raises(KeyError, match="boom"):
        write_outputs(outputs(), tmp_path, show=False)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["done.pdf", "done.png", "first.csv"]
    assert len(plt.get_fignums()) == 1
