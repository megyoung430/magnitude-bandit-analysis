"""Shared matplotlib style settings and color palettes for behavior visualizations."""

import matplotlib as mpl


# ---------------------------------------------------------------------------
# Default font and axis sizes – applied once at module import.
# ---------------------------------------------------------------------------
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial"]
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["axes.titlesize"] = 14

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
