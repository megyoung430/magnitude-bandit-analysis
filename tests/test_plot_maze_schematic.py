import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis.poke_events import ARMS9
from src.behavior_visualization.paper_style import paper_figures
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
    platform_outline,
    plot_maze_schematic,
    port_fills,
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


def test_paper_mode_writes_the_paper_formats_and_a_caption(tmp_path):
    with paper_figures():
        plot_maze_schematic(CHOICES, save_path=tmp_path / "towers")
    assert {p.name for p in tmp_path.iterdir()} == {
        "towers.pdf",
        "towers.svg",
        "towers.png",
        "towers.caption.txt",
    }
