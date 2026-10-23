import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.behavior_analysis.poke_events import ARMS9
from src.behavior_visualization import plot_maze_heatmaps as module
from src.behavior_visualization.plot_maze_heatmaps import (
    CURRENT_COLOR,
    INITIATION_COLOR,
    MAZE_POSITIONS,
    PREVIOUS_COLOR,
    count_norm,
    draw_maze_heatmap,
    plot_maze_heatmaps_across_mice,
    ring_specs,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def counts_with(**overrides):
    counts = dict.fromkeys(ARMS9, 0)
    counts.update(overrides)
    return counts


def rings(ax):
    """``{(port, role): PathPatch}`` for the outline rings drawn on *ax*."""
    found = {}
    for patch in ax.patches:
        gid = patch.get_gid() or ""
        if gid.startswith("ring:"):
            _, port, role = gid.split(":")
            found[(port, role)] = patch
    return found


# ------------------------------------------------------------------ layout


def test_maze_layout_has_letters_left_to_right_and_number_one_at_the_bottom():
    assert MAZE_POSITIONS == {
        "A1": (0, 0), "B1": (1, 0), "C1": (2, 0),
        "A2": (0, 1), "B2": (1, 1), "C2": (2, 1),
        "A3": (0, 2), "B3": (1, 2), "C3": (2, 2),
    }  # fmt: skip


# ------------------------------------------------------------- normalisation


def test_scale_starts_at_zero_even_when_every_port_was_poked():
    norm = count_norm(counts_with(**{p: 5 + i for i, p in enumerate(ARMS9)}))
    assert (norm.vmin, norm.vmax) == (0, 13)


def test_scale_of_an_all_zero_map_is_not_degenerate():
    norm = count_norm(counts_with())
    assert norm.vmin == 0 and norm.vmax > 0


def test_colorbar_ticks_are_zero_and_the_actual_maximum_with_the_given_label():
    ax = draw_maze_heatmap(counts_with(B1=40, A1=7), colorbar_label="Pokes (count)")
    colorbar_axes = [a for a in ax.figure.axes if a is not ax]
    assert len(colorbar_axes) == 1
    cbar_ax = colorbar_axes[0]
    assert cbar_ax.get_ylabel() == "Pokes (count)"
    assert [t.get_text() for t in cbar_ax.get_yticklabels()] == ["0", "40"]
    assert list(cbar_ax.get_yticks()) == [0, 40]


def test_colorbar_can_be_drawn_into_a_given_axes_leaving_the_map_where_it_is():
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_axes([0.2, 0.2, 0.5, 0.5])
    cax = fig.add_axes([0.75, 0.25, 0.03, 0.4])
    before = ax.get_position().bounds
    draw_maze_heatmap(counts_with(B1=40), colorbar_label="Pokes (count)", ax=ax, colorbar_ax=cax)
    assert fig.axes == [ax, cax]  # no further axes were taken from the map
    assert ax.get_position().bounds == pytest.approx(before)
    assert cax.get_ylabel() == "Pokes (count)"
    assert list(cax.get_yticks()) == [0, 40]


def test_every_port_shows_its_name_and_count():
    ax = draw_maze_heatmap(counts_with(B1=40, A1=7), colorbar_label="x")
    texts = [t.get_text() for t in ax.texts]
    assert sorted(t for t in texts if t in ARMS9) == sorted(ARMS9)
    assert "40" in texts and "7" in texts and texts.count("0") == 7


def test_a_port_missing_from_the_counts_raises():
    counts = counts_with()
    del counts["C3"]
    with pytest.raises(KeyError, match="C3"):
        draw_maze_heatmap(counts, colorbar_label="x")


# ----------------------------------------------------------------- outlines


def test_rings_have_equal_width_and_do_not_overlap():
    specs = ring_specs(["g", "r"], base_radius=0.18, width=0.04)
    widths = [s.outer - s.inner for s in specs]
    assert widths[0] == pytest.approx(widths[1])
    assert specs[0].outer == pytest.approx(specs[1].inner)
    assert specs[0].inner == pytest.approx(0.18)


def test_no_colours_means_no_rings():
    assert ring_specs([], base_radius=0.18, width=0.04) == []


def test_port_in_both_problems_gets_a_green_and_a_red_ring_of_equal_width():
    ax = draw_maze_heatmap(
        counts_with(),
        colorbar_label="x",
        current_ports={"A1", "B1"},
        previous_ports={"B1", "C3"},
    )
    found = rings(ax)
    assert set(found) == {
        ("A1", "current"), ("B1", "current"), ("B1", "previous"), ("C3", "previous"),
    }  # fmt: skip
    assert found[("B1", "current")].get_facecolor()[:3] == matplotlib.colors.to_rgb(CURRENT_COLOR)
    assert found[("B1", "previous")].get_facecolor()[:3] == matplotlib.colors.to_rgb(PREVIOUS_COLOR)

    def radii(patch):
        """Distinct vertex distances from the centre of B1 (grid column 1, row 0)."""
        distance = np.hypot(*(patch.get_path().vertices - np.array([0.65, 0.0])).T)
        return np.unique(distance.round(9))

    inner_ring, outer_ring = radii(found[("B1", "current")]), radii(found[("B1", "previous")])
    assert len(inner_ring) == len(outer_ring) == 2
    # Equal radial width, the red ring starting where the green one ends.
    assert np.diff(inner_ring) == pytest.approx(np.diff(outer_ring))
    assert inner_ring[1] == pytest.approx(outer_ring[0])


def test_no_previous_problem_means_no_red_rings():
    ax = draw_maze_heatmap(
        counts_with(), colorbar_label="x", current_ports={"A1"}, previous_ports=None
    )
    assert {role for _, role in rings(ax)} == {"current"}


def test_initiation_port_is_outlined_in_blue():
    ax = draw_maze_heatmap(counts_with(), colorbar_label="x", initiation_port="B2")
    found = rings(ax)
    assert set(found) == {("B2", "initiation")}
    assert found[("B2", "initiation")].get_facecolor()[:3] == matplotlib.colors.to_rgb(
        INITIATION_COLOR
    )


def test_unknown_outline_port_raises():
    with pytest.raises(ValueError, match="Z9"):
        draw_maze_heatmap(counts_with(), colorbar_label="x", current_ports={"Z9"})


# ------------------------------------------------------- one panel per mouse


def per_mouse():
    return {
        "m1": counts_with(A1=10, B1=3),
        "m2": counts_with(C1=250),
        "m3": counts_with(A3=1),
    }


def save_and_close(tmp_path, **kwargs):
    base = tmp_path / "maps"
    plot_maze_heatmaps_across_mice(save_path=base, **kwargs)
    return base


def test_figure_is_saved_as_png_and_pdf(tmp_path):
    base = save_and_close(tmp_path, per_mouse_counts=per_mouse(), colorbar_label="Pokes (count)")
    assert base.with_suffix(".png").exists() and base.with_suffix(".pdf").exists()


def test_one_panel_and_one_colorbar_per_mouse_each_scaled_to_its_own_max(monkeypatch):
    shown = []
    monkeypatch.setattr(module, "save_or_show", lambda fig, path: shown.append(fig))
    plot_maze_heatmaps_across_mice(per_mouse(), colorbar_label="Pokes (count)", save_path=None)
    (fig,) = shown
    labelled = [a for a in fig.axes if a.get_ylabel() == "Pokes (count)"]
    assert len(labelled) == 3
    tops = sorted(float(a.get_yticks()[-1]) for a in labelled)
    assert tops == [1, 10, 250]
    assert sorted(a.get_title() for a in fig.axes if a.get_title()) == ["m1", "m2", "m3"]


def test_empty_mouse_slot_is_hidden(monkeypatch):
    shown = []
    monkeypatch.setattr(module, "save_or_show", lambda fig, path: shown.append(fig))
    plot_maze_heatmaps_across_mice(per_mouse(), colorbar_label="x", save_path=None)
    (fig,) = shown
    # 3 mice on a 2 x 2 grid: 3 map panels, 3 colorbars, and one hidden empty slot.
    assert len(fig.axes) == 3 + 3 + 1
    assert sum(not a.axison for a in fig.axes) == 4  # 3 maps and the empty slot have no frame


def test_legend_names_only_the_outlines_that_are_drawn(monkeypatch):
    shown = []
    monkeypatch.setattr(module, "save_or_show", lambda fig, path: shown.append(fig))
    plot_maze_heatmaps_across_mice(
        {"m1": counts_with()},
        colorbar_label="x",
        current_ports={"m1": {"A1"}},
        previous_ports={"m1": None},
        save_path=None,
    )
    (legend,) = shown[0].legends
    assert [t.get_text() for t in legend.get_texts()] == ["Current problem choice ports"]


def test_no_mice_raises():
    with pytest.raises(ValueError, match="no mice"):
        plot_maze_heatmaps_across_mice({}, colorbar_label="x", save_path=None)


def test_per_mouse_outline_mapping_must_cover_every_mouse():
    with pytest.raises(ValueError, match="m3"):
        plot_maze_heatmaps_across_mice(
            per_mouse(),
            colorbar_label="x",
            current_ports={"m1": {"A1"}, "m2": {"A1"}},
            save_path=None,
        )


def test_the_first_suptitle_line_is_drawn_and_the_rest_is_the_caption(tmp_path):
    base = save_and_close(
        tmp_path,
        per_mouse_counts=per_mouse(),
        colorbar_label="Pokes (count)",
        suptitle="Pokes after choice\nColour scale: 0 to each mouse's own maximum count.",
    )
    caption = base.with_name(f"{base.name}.caption.txt").read_text()
    assert caption == "Colour scale: 0 to each mouse's own maximum count.\n"
