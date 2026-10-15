from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.behavior_visualization.plot_helpers import (
    FigureOutput,
    TableOutput,
    axes_or_new,
    despine,
    jittered,
    mouse_color_map,
    palette_color,
    save_or_show,
    write_outputs,
)
from src.behavior_visualization.plot_style import MOUSE_COLORS
from tests.plot_capture import capture_figures, was_tight_laid_out


@pytest.fixture(autouse=True)
def close_figures():
    plt.close("all")
    yield
    plt.close("all")


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


def test_a_failing_builder_keeps_earlier_files_and_closes_the_figures_it_opened(capture, tmp_path):
    def outputs():
        yield TableOutput("first.csv", pd.DataFrame({"a": [1]}))
        yield FigureOutput("done", small_figure())
        small_figure()  # a figure started by a builder that then raises
        raise KeyError("boom")

    with pytest.raises(KeyError, match="boom"):
        write_outputs(outputs(), tmp_path, show=False)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["done.pdf", "done.png", "first.csv"]
    assert plt.get_fignums() == []


def test_a_figure_that_cannot_be_saved_is_closed_with_the_ones_still_to_be_built(
    monkeypatch, tmp_path
):
    capture_figures(monkeypatch)

    def cannot_save(self, fname, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(plt.Figure, "savefig", cannot_save)
    pending = {}

    def outputs():
        pending["second"] = small_figure()  # built before the first is consumed
        yield FigureOutput("first", small_figure())
        yield FigureOutput("second", pending["second"])

    with pytest.raises(OSError, match="disk full"):
        write_outputs(outputs(), tmp_path, show=False)
    assert plt.get_fignums() == []


def test_figures_already_open_before_write_outputs_are_left_alone(capture, tmp_path):
    bystander = small_figure()

    def outputs():
        yield FigureOutput("first", small_figure())
        raise KeyError("boom")

    with pytest.raises(KeyError):
        write_outputs(outputs(), tmp_path, show=False)
    assert plt.get_fignums() == [bystander.number]


def test_an_unknown_output_raises_type_error(tmp_path):
    with pytest.raises(TypeError, match="TableOutput or FigureOutput"):
        write_outputs(["not an output"], tmp_path, show=False)


# ---- save_or_show ------------------------------------------------------------------


def test_save_or_show_writes_pdf_then_png_at_300_dpi_and_closes(capture, tmp_path):
    fig = small_figure()
    save_or_show(fig, tmp_path / "out" / "Fig")
    pdf, png = capture.saved
    assert pdf.path == str(tmp_path / "out" / "Fig") + ".pdf"
    assert pdf.kwargs == {"bbox_inches": "tight"}
    assert png.path == str(tmp_path / "out" / "Fig") + ".png"
    assert png.kwargs == {"dpi": 300, "bbox_inches": "tight"}
    assert (tmp_path / "out" / "Fig.pdf").is_file() and (tmp_path / "out" / "Fig.png").is_file()
    assert capture.shown == [] and plt.get_fignums() == []


def test_save_or_show_shows_without_a_path_and_closes(capture):
    fig = small_figure()
    save_or_show(fig, None)
    assert [[f.number for f in group] for group in capture.shown] == [[fig.number]]
    assert capture.saved == [] and plt.get_fignums() == []


def test_save_or_show_formats_and_dpi_are_chosen_by_the_caller(capture, tmp_path):
    save_or_show(
        small_figure(), tmp_path / "Fig", formats=("png", "pdf"), dpi={"png": 180, "pdf": 180}
    )
    assert [Path(s.path).suffix for s in capture.saved] == [".png", ".pdf"]
    assert [s.kwargs for s in capture.saved] == [{"dpi": 180, "bbox_inches": "tight"}] * 2


def test_save_or_show_without_make_parent_needs_an_existing_folder(capture, tmp_path):
    fig = small_figure()
    with pytest.raises(FileNotFoundError):
        save_or_show(fig, str(tmp_path / "missing" / "Fig"), make_parent=False)
    assert not (tmp_path / "missing").exists() and plt.get_fignums() == []


def test_save_or_show_without_make_parent_uses_the_path_as_given(capture, tmp_path):
    save_or_show(small_figure(), tmp_path / "Fig", make_parent=False)
    assert capture.saved_names() == ["Fig.pdf", "Fig.png"]


def test_palette_color_uses_the_palette_it_is_given():
    assert palette_color(3, ["#111111", "#222222"]) == "#222222"


def test_a_pre_existing_figure_closed_during_consumption_does_not_hide_a_builders_figure(
    capture, tmp_path
):
    bystander = small_figure()

    def outputs():
        yield from ()  # makes this a generator, consumed lazily by write_outputs
        plt.close(bystander)  # frees the bystander's figure number for reuse
        built = small_figure()  # a builder's figure that reuses that number, then it raises
        reused["same_number"] = built.number == bystander.number
        reused["built"] = built
        raise KeyError("boom")

    reused = {}
    with pytest.raises(KeyError):
        write_outputs(outputs(), tmp_path, show=False)
    assert reused["same_number"]  # the premise: the number really was reused
    assert not plt.fignum_exists(reused["built"].number)
    assert plt.get_fignums() == []


def test_save_or_show_closes_the_figure_when_saving_raises(monkeypatch, tmp_path):
    capture_figures(monkeypatch)

    def cannot_save(self, fname, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(plt.Figure, "savefig", cannot_save)
    with pytest.raises(OSError, match="disk full"):
        save_or_show(small_figure(), tmp_path / "Fig")
    assert plt.get_fignums() == []


def test_save_or_show_saves_each_named_figure_and_shows_them_together_once(capture, tmp_path):
    first, second = small_figure(), small_figure()
    save_or_show({" A": first, " B": second}, tmp_path / "Fig")
    assert capture.saved_names() == ["Fig A.pdf", "Fig A.png", "Fig B.pdf", "Fig B.png"]
    assert plt.get_fignums() == []
    third, fourth = small_figure(), small_figure()
    save_or_show({" A": third, " B": fourth}, None)
    assert [[f.number for f in group] for group in capture.shown] == [[third.number, fourth.number]]
    assert plt.get_fignums() == []


# ---- axes_or_new -------------------------------------------------------------------


def test_axes_or_new_gives_back_the_axes_it_was_given():
    fig, ax = plt.subplots()
    with axes_or_new(ax, figsize=(3, 3)) as drawn:
        assert drawn is ax
    assert plt.get_fignums() == [fig.number] and tuple(fig.get_size_inches()) != (3, 3)


def test_axes_or_new_creates_a_figure_that_survives_when_drawing_succeeds():
    with axes_or_new(None, figsize=(3, 2)) as ax:
        pass
    assert plt.get_fignums() == [ax.figure.number]
    assert tuple(ax.figure.get_size_inches()) == (3, 2)


def test_axes_or_new_closes_its_own_figure_but_not_the_callers_when_drawing_raises():
    with pytest.raises(RuntimeError, match="boom"):
        with axes_or_new(None) as ax:
            raise RuntimeError("boom")
    assert plt.get_fignums() == []
    fig, ax = plt.subplots()
    with pytest.raises(RuntimeError, match="boom"):
        with axes_or_new(ax):
            raise RuntimeError("boom")
    assert plt.get_fignums() == [fig.number]


# ---- colours, spines and jitter ----------------------------------------------------


def test_mouse_color_map_cycles_the_palette_in_order():
    mice = [f"m{i}" for i in range(len(MOUSE_COLORS) + 2)]
    colors = mouse_color_map(mice)
    assert colors["m0"] == MOUSE_COLORS[0]
    assert colors[mice[-1]] == MOUSE_COLORS[1]


def test_despine_hides_only_the_top_and_right_spines():
    _, ax = plt.subplots()
    despine(ax)
    assert {k: s.get_visible() for k, s in ax.spines.items()} == {
        "left": True,
        "right": False,
        "bottom": True,
        "top": False,
    }


def test_jittered_draws_from_the_global_stream_and_skips_it_when_zero():
    xs = np.arange(4.0)
    np.random.seed(3)
    expected = xs + np.random.uniform(-0.1, 0.1, size=4)
    np.random.seed(3)
    np.testing.assert_array_equal(jittered(xs, 0.1), expected)
    np.random.seed(3)
    np.testing.assert_array_equal(jittered(xs, 0), xs)
    assert np.random.uniform() == np.random.RandomState(3).uniform()  # stream untouched
