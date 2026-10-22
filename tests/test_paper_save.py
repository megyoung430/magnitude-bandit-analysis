"""Saving inside ``paper_figures``: PDF + SVG + PNG at 600 dpi, polished, with a caption file."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from PIL import Image

from src.behavior_visualization.paper_style import MM_PER_INCH, paper_figures
from src.behavior_visualization.plot_helpers import figure_banner, save_or_show


def long_title_figure():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1], label="Best")
    ax.set_xlabel("Trials From Change")
    ax.set_title("Choice probabilities around best-value decreases (n = 3 mice, 120 events)")
    ax.legend()
    return fig


def test_paper_mode_writes_pdf_svg_png_and_a_caption(tmp_path):
    with paper_figures():
        save_or_show(long_title_figure(), tmp_path / "fig")
    assert {p.name for p in tmp_path.iterdir()} == {
        "fig.pdf",
        "fig.svg",
        "fig.png",
        "fig.caption.txt",
    }
    assert "n = 3 mice, 120 events" in (tmp_path / "fig.caption.txt").read_text()


def test_the_png_is_600_dpi_at_column_width(tmp_path):
    with paper_figures():
        save_or_show(long_title_figure(), tmp_path / "fig")
    with Image.open(tmp_path / "fig.png") as image:
        assert image.info["dpi"][0] == pytest.approx(600, abs=1)
        # the tight bounding box is slightly narrower than the column
        assert image.width / 600 * MM_PER_INCH == pytest.approx(89, abs=8)


def test_the_text_is_editable_in_svg_and_pdf(tmp_path):
    with paper_figures():
        save_or_show(long_title_figure(), tmp_path / "fig")
    assert "<text" in (tmp_path / "fig.svg").read_text()
    pdf = (tmp_path / "fig.pdf").read_bytes()
    # fonttype 42 embeds the TrueType font file; the default (3) would embed glyph outlines
    assert b"/FontFile2" in pdf and b"/Type3" not in pdf


def test_the_figure_banner_goes_to_the_caption_not_the_figure(tmp_path):
    with paper_figures(), figure_banner("without MY_1_A"):
        save_or_show(long_title_figure(), tmp_path / "fig")
    assert "without MY_1_A" in (tmp_path / "fig.caption.txt").read_text()
    assert "without MY_1_A" not in (tmp_path / "fig.svg").read_text()


def test_a_figure_with_nothing_to_say_still_gets_an_empty_caption_file(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    with paper_figures():
        save_or_show(fig, tmp_path / "plain")
    assert (tmp_path / "plain.caption.txt").read_text() == ""


def test_outside_paper_mode_the_formats_are_the_callers(tmp_path):
    save_or_show(long_title_figure(), tmp_path / "fig")
    assert {p.name for p in tmp_path.iterdir()} == {"fig.pdf", "fig.png"}
