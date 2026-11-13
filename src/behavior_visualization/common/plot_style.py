"""Shared matplotlib style settings and color palettes for behavior visualizations."""

import matplotlib as mpl

# ---------------------------------------------------------------------------
# Font – applied once, when the plotting package is first imported.
# ---------------------------------------------------------------------------

# Arial where installed, then Helvetica; Liberation Sans has Arial's letter widths on Linux and
# DejaVu Sans ships with matplotlib, so a font is always found. The list sits behind the generic
# "sans-serif" family, which matplotlib resolves to the first installed name without warning
# about the missing ones. Text stays editable in vector files: TrueType in PDF, live <text> in
# SVG.
FONT_RC: dict[str, object] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
}

mpl.rcParams.update(FONT_RC)
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["axes.titlesize"] = 14

# ---------------------------------------------------------------------------
# Shared color palettes
# ---------------------------------------------------------------------------

# One colour per mouse, cycling past the end: the Okabe-Ito colours (without black), then a darker
# shade of each in the same order, so the first 14 mice of a pooled figure never share a colour.
# The darker shades were chosen to stay apart from all the others, also under simulated
# deuteranopia, protanopia and tritanopia (CAM02-UCS distance at least 10 between any two).
MOUSE_COLORS: list[str] = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#009E73",  # bluish green
    "#7D570C",  # dark orange (brown)
    "#005C85",  # dark sky blue
    "#7B2400",  # dark vermillion
    "#712B56",  # dark purple
    "#C2BD75",  # dark yellow (khaki)
    "#00348F",  # dark blue
    "#00522F",  # dark green
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
