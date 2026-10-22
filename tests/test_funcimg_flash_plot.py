"""Flash-check figures: per-cell event images and traces, and the animated clip."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.animation import PillowWriter  # noqa: E402

from src.funcimg_analysis.flash_check import clip_dff, flash_check  # noqa: E402
from src.funcimg_visualization.plot_funcimg_flash import (  # noqa: E402
    animate_clip,
    clip_gif_name,
    flash_pdf_name,
    plot_flash_cell,
    plot_flash_cells,
)
from tests.test_funcimg_flash_check import (  # noqa: E402
    CELL,
    N_FRAMES,
    PERIOD,
    SETTINGS,
    SHAPE,
    _movie_with_a_flashing_cell,
)

PEAKS = [200, 500, 800]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def cells():
    green = _movie_with_a_flashing_cell(PEAKS)
    red = _movie_with_a_flashing_cell(PEAKS, red=True)
    z = np.zeros((2, N_FRAMES))
    z[0, PEAKS] = 10.0
    stat = [CELL, {"ypix": np.array([5]), "xpix": np.array([5])}]
    return flash_check(
        {"green": green, "red": red},
        stat,
        {"green": z, "red": np.zeros_like(z)},
        np.array([3.0, 2.0]),
        PERIOD,
        **{**SETTINGS, "n_cells": 2},
    )


def test_a_cell_row_has_four_panels_with_the_event_images_on_one_scale(cells):
    fig = plot_flash_cell(cells[0])
    assert len(fig.axes) == 4
    green, red = (ax.get_images()[0] for ax in fig.axes[1:3])
    assert green.get_clim() == red.get_clim()
    assert green.get_clim()[1] == pytest.approx(0.5)


def test_a_cell_without_events_says_so_instead_of_drawing_images(cells):
    fig = plot_flash_cell(cells[1])
    texts = [t.get_text() for ax in fig.axes[1:3] for t in ax.texts]
    assert texts == ["no event above the threshold"] * 2


def test_given_axes_are_used_and_the_wrong_number_fails_loudly(cells):
    fig, axes = plt.subplots(1, 4)
    assert plot_flash_cell(cells[0], axes=axes) is fig
    _, three = plt.subplots(1, 3)
    with pytest.raises(ValueError, match="4 axes"):
        plot_flash_cell(cells[0], axes=three)


def test_all_cells_are_rows_of_one_figure(cells):
    assert len(plot_flash_cells(cells, title="x").axes) == 8


def test_the_clip_animates_and_saves_as_a_gif(cells, tmp_path):
    movie = _movie_with_a_flashing_cell(PEAKS)[190:220:3]
    clip = {"green": movie, "red": np.full_like(movie, 100.0), "origin": (0, 0)}
    dff = {channel: clip_dff(clip[channel]) for channel in ("green", "red")}
    animation = animate_clip(clip, dff, cells[:1], 0.3, start_s=19.0, title="t")
    path = tmp_path / "clip.gif"
    animation.save(path, writer=PillowWriter(fps=5))
    assert path.stat().st_size > 0
    assert movie.shape[1:] == SHAPE


def test_file_names_carry_what_shapes_them():
    assert flash_pdf_name("ses-1-1", "v", 212.4) == "flash_check_ses-1-1_v_cut212s.pdf"
    assert clip_gif_name("ses-1-1", "v", 212.4, 60) == "clip_ses-1-1_v_212s-272s.gif"
