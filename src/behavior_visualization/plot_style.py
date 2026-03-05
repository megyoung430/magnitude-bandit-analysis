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
    "#4C72B0", "#55A868", "#C44E52", "#8172B2",
    "#CCB974", "#64B5CD", "#8C8C8C", "#DD8452",
    "#937860", "#DA8BC3", "#8C6D31", "#1F77B4",
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
