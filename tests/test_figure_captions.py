"""Figure titles keep the heading and statistics; everything else goes to ``<name>.caption.txt``."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from src.behavior_visualization.plot_helpers import (
    CAPTION_SUFFIX,
    FIGURE_FORMATS,
    FigureOutput,
    Title,
    add_caption,
    caption_path,
    count_text,
    figure_banner,
    figure_caption,
    figure_paths,
    save_or_show,
    set_suptitle,
    set_title,
    stats_line,
    write_outputs,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def small_figure():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    return fig, ax


# --- the parts of a title -----------------------------------------------------------------------


@pytest.mark.parametrize(("n", "text"), [(1, "1 mouse"), (2, "2 mice"), (0, "0 mice")])
def test_counts_agree_in_number(n, text):
    assert count_text(n, "mouse", "mice") == text


def test_a_stats_line_names_the_statistic_the_unit_and_each_count():
    line = stats_line("mean ± se", "mice", [(13, "mouse", "mice")])
    assert line == "mean ± se across mice, n=13 mice"
    line = stats_line("mean ± se", "events", [(120, "event", "events"), (1, "mouse", "mice")])
    assert line == "mean ± se across events, n=120 events, n=1 mouse"


def test_a_title_shows_its_heading_and_stats_and_keeps_the_rest_for_the_caption():
    title = Title("Average Across Mice", "mean ± se across mice, n=3 mice", ("pooled",))
    assert title.shown == "Average Across Mice\nmean ± se across mice, n=3 mice"
    assert Title("Value of Best").shown == "Value of Best"


def test_set_title_draws_the_shown_part_and_adds_the_rest_to_the_caption():
    fig, ax = small_figure()
    set_title(ax, Title("Heading", "mean ± se across mice, n=2 mice", ("note one", "note two")))
    assert ax.get_title() == "Heading\nmean ± se across mice, n=2 mice"
    assert figure_caption(fig) == "note one\nnote two"


def test_set_suptitle_does_the_same_for_the_figure():
    fig, _ = small_figure()
    set_suptitle(fig, Title("Heading", None, ("scope",)))
    assert fig.get_suptitle() == "Heading"
    assert figure_caption(fig) == "scope"


def test_caption_lines_are_never_drawn_and_repeated_lines_are_kept_once(tmp_path):
    fig, _ = small_figure()
    add_caption(fig, "a", "b")
    add_caption(fig, "a")
    assert all(not t.get_visible() for t in fig.texts)
    assert figure_caption(fig) == "a\nb"


# --- the caption file ---------------------------------------------------------------------------


def test_every_save_writes_all_three_formats_and_a_caption_file(tmp_path):
    fig, ax = small_figure()
    set_title(ax, Title("Heading", None, ("detail",)))
    save_or_show(fig, tmp_path / "Fig")
    assert FIGURE_FORMATS == ("pdf", "png", "svg")
    for fmt in FIGURE_FORMATS:
        assert (tmp_path / f"Fig.{fmt}").is_file()
    assert caption_path(tmp_path / "Fig") == tmp_path / f"Fig{CAPTION_SUFFIX}"
    assert (tmp_path / "Fig.caption.txt").read_text() == "detail\n"


def test_a_figure_with_nothing_to_say_still_gets_an_empty_caption_file(tmp_path):
    save_or_show(small_figure()[0], tmp_path / "Fig")
    assert (tmp_path / "Fig.caption.txt").read_text() == ""


def test_the_banner_goes_to_the_caption_not_onto_the_figure(tmp_path, monkeypatch):
    fig, ax = small_figure()
    set_title(ax, Title("Heading", None, ("detail",)))
    drawn = []
    monkeypatch.setattr(
        fig,
        "savefig",
        lambda *a, **k: drawn.append([t.get_text() for t in fig.texts if t.get_visible()]),
    )
    with figure_banner("without MY_1_A"):
        save_or_show(fig, tmp_path / "Fig")
    assert drawn == [[], [], []]
    assert (tmp_path / "Fig.caption.txt").read_text() == "without MY_1_A\ndetail\n"


def test_several_figures_under_one_base_each_get_their_caption(tmp_path):
    (a, ax_a), (b, ax_b) = small_figure(), small_figure()
    set_title(ax_a, Title("A", None, ("about a",)))
    set_title(ax_b, Title("B", None, ("about b",)))
    save_or_show({" one": a, " two": b}, tmp_path / "Fig")
    assert (tmp_path / "Fig one.caption.txt").read_text() == "about a\n"
    assert (tmp_path / "Fig two.caption.txt").read_text() == "about b\n"


def test_write_outputs_writes_the_three_formats_and_the_caption(tmp_path):
    fig, ax = small_figure()
    set_title(ax, Title("A", None, ("about a",)))
    paths = write_outputs([FigureOutput("f", fig)], tmp_path, show=False)
    assert paths == [tmp_path / "f.png", tmp_path / "f.pdf", tmp_path / "f.svg"]
    assert (tmp_path / "f.caption.txt").read_text() == "about a\n"


def test_figure_paths_lists_the_files_a_save_writes(tmp_path):
    assert figure_paths(tmp_path / "f") == [
        tmp_path / "f.pdf",
        tmp_path / "f.png",
        tmp_path / "f.svg",
    ]
