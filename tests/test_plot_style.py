"""Shared style: default fonts, and the opt-in paper-ready font settings."""

import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.t2CharStringPen import T2CharStringPen

from src.behavior_visualization.plot_style import (
    SANS_SERIF_FONTS,
    FontFallbackWarning,
    is_cff_font,
    paper_rc,
    resolved_sans_serif_font,
    use_paper_fonts,
)

DEJAVU_TTF = Path(mpl.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"


@pytest.fixture(autouse=True)
def restore_rc_params():
    with mpl.rc_context():
        yield
    plt.close("all")


@pytest.fixture
def cff_otf(tmp_path) -> Path:
    """A minimal OpenType font with CFF outlines, the kind most ``.otf`` files are."""
    builder = FontBuilder(1000, isTTF=False)
    builder.setupGlyphOrder([".notdef"])
    builder.setupCharacterMap({})
    pen = T2CharStringPen(500, None)
    builder.setupCFF("TestCFF", {"FullName": "Test CFF"}, {".notdef": pen.getCharString()}, {})
    builder.setupHorizontalMetrics({".notdef": (500, 0)})
    builder.setupHorizontalHeader(ascent=800, descent=-200)
    builder.setupNameTable({"familyName": "Test CFF", "styleName": "Regular"})
    builder.setupOS2()
    builder.setupPost()
    path = tmp_path / "test-cff.otf"
    builder.save(str(path))
    return path


def test_default_style_keeps_matplotlib_font_embedding():
    # The paper settings are opt-in: importing the style must not change how files are written.
    assert mpl.rcParams["font.sans-serif"] == SANS_SERIF_FONTS
    assert mpl.rcParams["pdf.fonttype"] == mpl.rcParamsDefault["pdf.fonttype"]
    assert mpl.rcParams["svg.fonttype"] == mpl.rcParamsDefault["svg.fonttype"]


def test_paper_rc_keeps_text_editable_without_touching_the_current_style():
    before = dict(mpl.rcParams)
    rc = paper_rc()
    assert rc["pdf.fonttype"] == 42
    assert rc["ps.fonttype"] == 42
    assert rc["svg.fonttype"] == "none"
    assert "font.sans-serif" not in rc  # without a font file the fonts stay as they are
    assert dict(mpl.rcParams) == before


def test_a_registered_font_file_goes_first():
    rc = paper_rc(DEJAVU_TTF)
    assert rc["font.sans-serif"] == ["DejaVu Sans", *SANS_SERIF_FONTS]


def test_a_missing_font_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.ttf"):
        paper_rc(tmp_path / "nope.ttf")


def test_cff_fonts_are_told_apart_from_truetype_ones(cff_otf):
    assert is_cff_font(cff_otf)
    assert not is_cff_font(DEJAVU_TTF)


def test_a_cff_otf_warns_that_pdf_text_may_not_be_editable(cff_otf):
    with pytest.warns(UserWarning, match=r"CFF.*\.ttf"):
        paper_rc(cff_otf)


def test_resolved_font_is_the_one_matplotlib_actually_uses():
    with mpl.rc_context({"font.sans-serif": ["No Such Font", "DejaVu Sans"]}):
        assert resolved_sans_serif_font() == "DejaVu Sans"


def test_use_paper_fonts_warns_when_the_preferred_font_is_missing():
    with mpl.rc_context({"font.sans-serif": ["No Such Font", "DejaVu Sans"]}):
        with pytest.warns(FontFallbackWarning, match="No Such Font.*DejaVu Sans"):
            assert use_paper_fonts() == "DejaVu Sans"


def test_use_paper_fonts_is_quiet_when_the_preferred_font_is_found():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FontFallbackWarning)
        assert use_paper_fonts(DEJAVU_TTF) == "DejaVu Sans"
    assert mpl.rcParams["pdf.fonttype"] == 42


def test_a_paper_pdf_embeds_truetype_text(tmp_path):
    with mpl.rc_context(paper_rc(DEJAVU_TTF)):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "B2")
        fig.savefig(tmp_path / "text.pdf")
    subtypes = set(re.findall(rb"/Subtype\s*/(\w+)", (tmp_path / "text.pdf").read_bytes()))
    assert b"CIDFontType2" in subtypes  # TrueType outlines: text stays editable
    assert b"Type3" not in subtypes
