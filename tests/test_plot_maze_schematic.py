import matplotlib

matplotlib.use("Agg")

import logging
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis.poke_events import ARMS9
from src.behavior_visualization.plot_maze_heatmaps import (
    NODE_RADIUS,
    octagon_vertices,
    port_center,
)
from src.behavior_visualization.plot_maze_schematic import (
    CHOICE_FILL,
    IDLE_FILL,
    INITIATION_FILL,
    draw_maze_schematic,
    first_installed_font,
    platform_outline,
    plot_maze_schematic,
    port_fills,
    schematic_font,
)

CHOICES = ["A2", "B1", "C3"]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ------------------------------------------------------------------ fills


def test_each_port_gets_the_fill_of_its_role():
    fills = port_fills(CHOICES, "B2")
    assert fills["B2"] == INITIATION_FILL
    assert {fills[p] for p in CHOICES} == {CHOICE_FILL}
    assert {fills[p] for p in ("A1", "A3", "B3", "C1", "C2")} == {IDLE_FILL}
    assert set(fills) == set(ARMS9)


def test_the_initiation_tower_defaults_to_b2():
    assert port_fills(["A1", "C1", "C3"]) == port_fills(["A1", "C1", "C3"], "B2")


def test_another_initiation_tower_can_be_given():
    fills = port_fills(["A1", "C1", "C3"], "A2")
    assert fills["A2"] == INITIATION_FILL
    assert fills["B2"] == IDLE_FILL


def test_unknown_port_is_rejected():
    with pytest.raises(ValueError, match="B4"):
        port_fills(["A2", "B4"])


def test_port_cannot_be_both_initiation_and_choice():
    with pytest.raises(ValueError, match="B2"):
        port_fills(["B2", "A1"], "B2")


def test_choice_towers_can_be_a_pandas_column():
    # Membership on a Series tests its index, so the towers must be read as values.
    assert port_fills(pd.Series(CHOICES)) == port_fills(CHOICES)
    with pytest.raises(ValueError, match="B2"):
        port_fills(pd.Series(["B2", "A1"]))


@pytest.mark.parametrize(
    "choices, message",
    [("A2", "list of towers"), ([], "at least one"), (["A2", "A2", "C3"], "more than once")],
)
def test_malformed_choice_towers_are_rejected(choices, message):
    with pytest.raises(ValueError, match=message):
        port_fills(choices)


# --------------------------------------------------------------- platform


def test_platform_edge_is_flush_with_the_outer_sides_of_the_edge_towers():
    outline = platform_outline()
    half_width = NODE_RADIUS * np.cos(np.pi / 8)  # centre to flat side
    xs = [port_center(p)[0] for p in ARMS9]
    ys = [port_center(p)[1] for p in ARMS9]
    assert outline[:, 0].min() == pytest.approx(min(xs) - half_width)
    assert outline[:, 0].max() == pytest.approx(max(xs) + half_width)
    assert outline[:, 1].min() == pytest.approx(min(ys) - half_width)
    assert outline[:, 1].max() == pytest.approx(max(ys) + half_width)


def test_platform_corners_follow_the_corner_towers_angled_faces():
    outline = platform_outline()
    corner_tower = octagon_vertices(port_center("A1"), NODE_RADIUS)
    # The two corners of A1's bottom-left face are outline vertices; nothing pokes out past them.
    lower_left = corner_tower[np.argsort(corner_tower.sum(axis=1))[:2]]
    for vertex in lower_left:
        assert np.isclose(outline, vertex).all(axis=1).any()
    assert len(outline) == 8


# ---------------------------------------------------------------- drawing


def test_draws_into_the_axes_it_is_given():
    fig, axes = plt.subplots(1, 2)
    returned = draw_maze_schematic(CHOICES, ax=axes[1])
    assert returned is axes[1]
    assert len(axes[0].patches) == 0
    assert len(axes[1].patches) == 1 + len(ARMS9)  # platform plus one octagon per port


def test_port_labels_are_not_bold():
    ax = draw_maze_schematic(CHOICES)
    labels = {t.get_text(): t for t in ax.texts}
    assert set(labels) == set(ARMS9)
    assert all(t.get_fontweight() == "normal" for t in labels.values())


def test_tower_outlines_are_not_clipped_at_the_axes_edge():
    # On a small panel the stroke of the outer towers reaches past the axes limits.
    ax = draw_maze_schematic(CHOICES)
    towers = [patch for patch in ax.patches if patch.get_edgecolor()[3] > 0]
    assert len(towers) == len(ARMS9)
    assert not any(patch.get_clip_on() for patch in towers)


def test_saves_svg_pdf_and_png(tmp_path):
    base = tmp_path / "problem-05"
    plot_maze_schematic(CHOICES, save_path=base)
    assert {p.name for p in tmp_path.iterdir()} == {
        "problem-05.svg",
        "problem-05.pdf",
        "problem-05.png",
    }


# ------------------------------------------------------------------ fonts


def test_the_first_installed_font_is_picked():
    assert first_installed_font(("No Such Font", "DejaVu Sans", "Arial")) == "DejaVu Sans"


def test_no_installed_font_is_an_error():
    with pytest.raises(ValueError, match="No Such Font"):
        first_installed_font(("No Such Font",))


def test_schematic_font_prefers_arial():
    try:
        first_installed_font(("Arial",))
    except ValueError:
        pytest.skip("Arial is not installed here")
    assert schematic_font() == "Arial"


def test_all_text_uses_the_schematic_font():
    ax = draw_maze_schematic(CHOICES, title="Choice towers")
    texts = [*ax.texts, ax.title, *ax.get_legend().get_texts()]
    assert all(t.get_fontfamily() == [schematic_font()] for t in texts)


def test_text_uses_the_schematic_font_in_an_axes_the_caller_made():
    # The caller's axes and its title exist before the schematic is drawn.
    _, ax = plt.subplots()
    draw_maze_schematic(CHOICES, ax=ax, title="Choice towers")
    texts = [*ax.texts, ax.title, *ax.get_legend().get_texts()]
    assert all(t.get_fontfamily() == [schematic_font()] for t in texts)


def test_saving_does_not_warn_about_missing_fonts(tmp_path, caplog):
    # Schematic text names one installed font, so drawing it never logs a missing-font warning.
    with caplog.at_level(logging.WARNING, logger="matplotlib.font_manager"):
        plot_maze_schematic(CHOICES, title="Choice towers", save_path=tmp_path / "towers")
    assert not [r for r in caplog.records if "not found" in r.getMessage()]


def test_the_font_settings_do_not_leak_into_other_figures(tmp_path):
    before = dict(mpl.rcParams)
    draw_maze_schematic(CHOICES)
    plot_maze_schematic(CHOICES, save_path=tmp_path / "towers")
    assert dict(mpl.rcParams) == before


def test_saved_text_stays_editable(tmp_path):
    base = tmp_path / "towers"
    plot_maze_schematic(CHOICES, save_path=base)
    pdf_fonts = set(re.findall(rb"/Subtype\s*/(\w+)", base.with_suffix(".pdf").read_bytes()))
    assert b"CIDFontType2" in pdf_fonts  # TrueType text, not Type 3 outlines
    assert b"Type3" not in pdf_fonts
    assert "<text" in base.with_suffix(".svg").read_text()
