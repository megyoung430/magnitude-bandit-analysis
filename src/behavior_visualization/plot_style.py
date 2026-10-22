"""Shared matplotlib style settings and color palettes for behavior visualizations."""

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


def use_paper_fonts(font_path: str | Path | None = None) -> str:
    """Make saved figures paper-ready: text stays editable text in PDF, PS and SVG files.

    Embeds TrueType (Type 42) fonts in PDF/PS instead of Type 3 outlines, and keeps SVG text as
    ``<text>`` elements, so labels can be edited in Illustrator or Inkscape. Call once, at the
    top of a notebook, before drawing. Font family and sizes are left as set above.

    Args:
        font_path (str | Path | None): A font file (e.g. ``HelveticaNeueRoman.otf``) to register
            and put first in the sans-serif list, for machines where Helvetica Neue is not
            installed system-wide. None uses the installed fonts. Default None.

    Returns:
        str: The name of the first sans-serif font matplotlib will try.

    Raises:
        FileNotFoundError: If *font_path* is given but does not exist.
    """
    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    if font_path is None:
        return mpl.rcParams["font.sans-serif"][0]
    font_path = Path(font_path)
    if not font_path.is_file():
        raise FileNotFoundError(f"font file not found: {font_path}")
    font_manager.fontManager.addfont(str(font_path))
    name = font_manager.FontProperties(fname=str(font_path)).get_name()
    mpl.rcParams["font.sans-serif"] = [name, *SANS_SERIF_FONTS]
    return name

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
