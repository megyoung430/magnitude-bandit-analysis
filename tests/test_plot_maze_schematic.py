import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
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
    platform_outline,
    plot_maze_schematic,
    port_fills,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ------------------------------------------------------------------ fills


def test_each_port_gets_the_fill_of_its_role():
    fills = port_fills("B2", ["A2", "B1", "C3"])
    assert fills["B2"] == INITIATION_FILL
    assert {fills[p] for p in ("A2", "B1", "C3")} == {CHOICE_FILL}
    assert {fills[p] for p in ("A1", "A3", "B3", "C1", "C2")} == {IDLE_FILL}
    assert set(fills) == set(ARMS9)


def test_unknown_port_is_rejected():
    with pytest.raises(ValueError, match="B4"):
        port_fills("B2", ["A2", "B4"])


def test_port_cannot_be_both_initiation_and_choice():
    with pytest.raises(ValueError, match="B2"):
        port_fills("B2", ["B2", "A1"])


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
    returned = draw_maze_schematic("B2", ["A2", "B1", "C3"], ax=axes[1])
    assert returned is axes[1]
    assert len(axes[0].patches) == 0
    assert len(axes[1].patches) == 1 + len(ARMS9)  # platform plus one octagon per port


def test_port_labels_are_not_bold():
    ax = draw_maze_schematic("B2", ["A2", "B1", "C3"])
    labels = {t.get_text(): t for t in ax.texts}
    assert set(labels) == set(ARMS9)
    assert all(t.get_fontweight() == "normal" for t in labels.values())


def test_saves_svg_pdf_and_png(tmp_path):
    base = tmp_path / "problem-05"
    plot_maze_schematic("B2", ["A2", "B1", "C3"], save_path=base)
    assert {p.name for p in tmp_path.iterdir()} == {
        "problem-05.svg",
        "problem-05.pdf",
        "problem-05.png",
    }
