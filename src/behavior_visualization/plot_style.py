"""Shared matplotlib style settings and color palettes for behavior visualizations."""

import warnings
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager

# ---------------------------------------------------------------------------
# Default font and axis sizes – applied once at module import.
# ---------------------------------------------------------------------------
SANS_SERIF_FONTS: list[str] = ["Helvetica Neue", "Helvetica", "Arial"]

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = SANS_SERIF_FONTS
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["axes.titlesize"] = 14


class FontFallbackWarning(UserWarning):
    """Matplotlib will draw text in another font than the first one asked for."""


def is_cff_font(font_path: str | Path) -> bool:
    """True for an OpenType font with CFF (PostScript) outlines, as most ``.otf`` files are.

    Args:
        font_path (str | Path): A font file.

    Returns:
        bool: ``True`` if the file starts with the ``OTTO`` tag of CFF-flavoured OpenType.
    """
    with open(font_path, "rb") as file:
        return file.read(4) == b"OTTO"


def _register_font(font_path: str | Path) -> str:
    """Add a font file to matplotlib's font list and return its family name."""
    font_path = Path(font_path)
    if not font_path.is_file():
        raise FileNotFoundError(f"font file not found: {font_path}")
    if is_cff_font(font_path):
        warnings.warn(
            f"{font_path.name} has CFF outlines; matplotlib embeds them in PDFs under a TrueType "
            "label, which Illustrator may garble or reject. Use a .ttf version of the font for "
            "paper PDFs (SVG text is not affected).",
            UserWarning,
            stacklevel=3,
        )
    font_manager.fontManager.addfont(str(font_path))
    return font_manager.FontProperties(fname=str(font_path)).get_name()


def paper_rc(font_path: str | Path | None = None) -> dict:
    """rcParams that keep text editable in saved PDF, PS and SVG files, for paper figures.

    TrueType (Type 42) fonts are embedded in PDF/PS instead of Type 3 outlines, and SVG text is
    kept as ``<text>`` elements that name the font rather than embed it. Font family and sizes
    are left as they are. Nothing is changed until the result is applied, e.g.
    ``with mpl.rc_context(paper_rc()):`` for a block of figures, or :func:`use_paper_fonts`.

    Args:
        font_path (str | Path | None): A font file to put first in the sans-serif list, for a
            machine where Helvetica Neue is not installed (e.g. Colab). Prefer a ``.ttf``: a
            CFF ``.otf`` warns, as its PDF text may not open as editable. Registering a file
            makes it available to every later figure. None uses the installed fonts. Default
            None.

    Returns:
        dict: rcParams to apply.

    Raises:
        FileNotFoundError: If *font_path* is given but does not exist.
    """
    rc = {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}
    if font_path is not None:
        rc["font.sans-serif"] = list(dict.fromkeys([_register_font(font_path), *SANS_SERIF_FONTS]))
    return rc


def resolved_sans_serif_font() -> str:
    """Name of the font matplotlib actually draws sans-serif text in, under the current rcParams.

    Returns:
        str: The family name of the first installed font of ``font.sans-serif``, or of
        matplotlib's fallback if none is installed.
    """
    path = font_manager.findfont(font_manager.FontProperties(family=["sans-serif"]))
    return font_manager.FontProperties(fname=path).get_name()


def use_paper_fonts(font_path: str | Path | None = None) -> str:
    """Apply :func:`paper_rc` for every figure from now on, and say which font is used.

    For a notebook that makes only paper figures; to change only some figures, use
    ``mpl.rc_context(paper_rc(...))`` instead.

    Args:
        font_path (str | Path | None): As for :func:`paper_rc`. Default None.

    Returns:
        str: The name of the font text will actually be drawn in.

    Raises:
        FileNotFoundError: If *font_path* is given but does not exist.

    Warns:
        FontFallbackWarning: If that is not the first font asked for (e.g. Helvetica Neue is not
            installed), so the figures would quietly come out in another font.
    """
    rc = paper_rc(font_path)
    mpl.rcParams.update(rc)
    wanted, used = mpl.rcParams["font.sans-serif"][0], resolved_sans_serif_font()
    if used != wanted:
        warnings.warn(
            f"{wanted} is not installed, so text will be drawn in {used}; pass the font file "
            "as font_path to use it.",
            FontFallbackWarning,
            stacklevel=2,
        )
    return used


# ---------------------------------------------------------------------------
# Shared color palettes
# ---------------------------------------------------------------------------

# One color per subject/mouse, cycling when there are more subjects than colors.
MOUSE_COLORS: list[str] = [
    "#4C72B0",  # blue
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B2",  # purple
    "#CCB974",  # muted yellow
    "#64B5CD",  # cyan
    "#8C8C8C",  # gray
    "#DD8452",  # orange
    "#937860",  # brown
    "#DA8BC3",  # pink
    "#8C6D31",  # dark brown
    "#1F77B4",  # strong blue
    "#2A9D8F",  # teal
    "#E76F51",  # coral
    "#6A994E",  # olive green
    "#B565A7",  # magenta-purple
]

# Colors for reversal-type lines/bars.
GOOD_COLOR: str = "#3A982E"
BAD_COLOR: str = "#F97979"
TOTAL_COLOR: str = "#808080"

# Colors for choice-probability curves around good reversals.
CHOICE_PROB_COLOR_MAP: dict[str, str] = {
    "prev_best": "#5DA5DA",
    "next_best": "#60BD68",
    "third": "#7f7f7f",
}
