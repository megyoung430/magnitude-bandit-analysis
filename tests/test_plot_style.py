"""The shared font and mouse colours that every behaviour figure uses."""

import logging
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgb

from src.behavior_visualization.plot_helpers import mouse_color_map, save_or_show
from src.behavior_visualization.plot_rank_proportions import plot_average_across_mice
from src.behavior_visualization.plot_style import FONT_RC, MOUSE_COLORS
from tests.rank_rows import make_row

OKABE_ITO_MICE = ["#E69F00", "#56B4E9", "#D55E00", "#CC79A7", "#F0E442", "#0072B2", "#009E73"]
FONT_LIST = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def rank_counts(n_mice: int = 3) -> dict:
    rng = np.random.default_rng(0)
    return {f"m{i}": [make_row(*rng.integers(1, 10, 3))] for i in range(n_mice)}


def test_the_font_is_the_sans_serif_list_with_editable_text_in_vector_files():
    assert FONT_RC == {
        "font.family": "sans-serif",
        "font.sans-serif": FONT_LIST,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }


def test_importing_the_plotting_package_applies_the_font():
    for key, value in FONT_RC.items():
        expected = [value] if key == "font.family" else value
        assert matplotlib.rcParams[key] == expected


def test_a_normal_figure_is_drawn_in_the_font_and_saved_with_editable_text(tmp_path):
    ax = plot_average_across_mice(rank_counts())
    texts = [ax.title, ax.yaxis.label, *ax.get_xticklabels()]
    assert all(t.get_fontfamily() == ["sans-serif"] for t in texts)
    save_or_show(ax.figure, tmp_path / "rank")
    pdf = (tmp_path / "rank.pdf").read_bytes()
    assert b"Type3" not in set(re.findall(rb"/Subtype\s*/(\w+)", pdf))
    assert "<text" in (tmp_path / "rank.svg").read_text()


def test_drawing_logs_no_missing_font_warnings(tmp_path, caplog):
    # Arial and Helvetica are missing on Linux CI; the generic family falls back quietly.
    with caplog.at_level(logging.WARNING, logger="matplotlib.font_manager"):
        for name in ("a", "b"):
            save_or_show(plot_average_across_mice(rank_counts()).figure, tmp_path / name)
    assert not [r for r in caplog.records if "findfont" in r.getMessage()]


# --- mouse colours ------------------------------------------------------------------------------


def test_the_first_seven_mouse_colours_are_okabe_ito_in_order():
    assert MOUSE_COLORS[:7] == OKABE_ITO_MICE


def test_the_next_seven_are_darker_shades_of_the_same_hues_in_the_same_order():
    assert len(MOUSE_COLORS) == 14
    for light, dark in zip(MOUSE_COLORS[:7], MOUSE_COLORS[7:], strict=True):
        h_light, _, v_light = matplotlib.colors.rgb_to_hsv(to_rgb(light))
        h_dark, _, v_dark = matplotlib.colors.rgb_to_hsv(to_rgb(dark))
        assert v_dark < v_light, (light, dark)
        hue_gap = abs(h_light - h_dark)
        assert min(hue_gap, 1 - hue_gap) < 0.08, (light, dark)


def test_fourteen_mice_get_fourteen_different_colours():
    colours = mouse_color_map([f"m{i}" for i in range(14)])
    assert len({c.lower() for c in colours.values()}) == 14
    assert list(colours.values()) == MOUSE_COLORS
