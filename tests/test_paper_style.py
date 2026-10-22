"""The Nature/Cell figure style: rc settings, per-axes polish, sizes and the caption split."""

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_hex

from src.behavior_visualization.paper_style import (
    MM_PER_INCH,
    OKABE_ITO,
    PAPER_DPI,
    figure_size,
    in_paper_mode,
    paper_figures,
    paper_style,
    polish,
    sentence_case,
)
from src.behavior_visualization.plot_style import CHOICE_PROB_COLOR_MAP


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def demo_figure(title="A short title", **subplots_kwargs):
    fig, ax = plt.subplots(figsize=(6, 4), **subplots_kwargs)
    ax.plot([0, 1, 2, 3], [0.1, 0.5, 0.2, 0.9], lw=2.2, marker="o", ms=6, label="Best")
    ax.set_xlim(-0.4, 3.4)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Trials From Value Change", fontsize=12)
    ax.set_ylabel("Choice Probability", fontsize=12)
    ax.legend(frameon=True, fontsize=10)
    return fig, ax


# --- the rc context ---------------------------------------------------------------------------


def test_paper_style_sets_fonts_and_editable_text_and_restores_afterwards():
    before = dict(mpl.rcParams)
    with paper_style():
        assert mpl.rcParams["font.size"] == 7
        assert mpl.rcParams["axes.linewidth"] == 0.5
        assert mpl.rcParams["pdf.fonttype"] == 42
        assert mpl.rcParams["svg.fonttype"] == "none"
        assert mpl.rcParams["axes.spines.top"] is False
        assert mpl.rcParams["xtick.direction"] == "out"
        assert mpl.rcParams["xtick.major.size"] == 2.5
    assert dict(mpl.rcParams) == before


def test_paper_style_font_list_ends_in_a_font_that_exists_everywhere():
    with paper_style():
        fonts = mpl.rcParams["font.sans-serif"]
    assert fonts[0] == "Arial" and "DejaVu Sans" in fonts


def test_paper_mode_is_on_only_inside_paper_figures():
    assert not in_paper_mode()
    with paper_figures():
        assert in_paper_mode()
        assert mpl.rcParams["font.size"] == 7
    assert not in_paper_mode()
    assert mpl.rcParams["font.size"] != 7


def test_paper_dpi_is_600():
    assert PAPER_DPI == 600


# --- sizes ------------------------------------------------------------------------------------


@pytest.mark.parametrize(("kind", "mm"), [("single", 89), ("one_half", 120), ("double", 183)])
def test_figure_size_is_the_column_width_in_inches(kind, mm):
    width, height = figure_size(kind, aspect=0.5)
    assert width == pytest.approx(mm / MM_PER_INCH)
    assert height == pytest.approx(0.5 * mm / MM_PER_INCH)


def test_figure_size_caps_the_height_at_the_page():
    _, height = figure_size("single", aspect=10)
    assert height * MM_PER_INCH == pytest.approx(230)


def test_figure_size_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="triple"):
        figure_size("triple", aspect=1)


# --- polish -----------------------------------------------------------------------------------


def test_polish_removes_top_and_right_spines_and_trims_to_the_outer_ticks():
    fig, ax = demo_figure()
    with paper_style():
        polish(fig)
    assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()
    xticks = [t for t in ax.get_xticks() if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
    yticks = [t for t in ax.get_yticks() if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]]
    assert ax.spines["bottom"].get_bounds() == (min(xticks), max(xticks))
    assert ax.spines["left"].get_bounds() == (min(yticks), max(yticks))


def test_polish_sets_thin_spines_outward_ticks_and_7_pt_text():
    fig, ax = demo_figure()
    polish(fig)
    assert ax.spines["left"].get_linewidth() == 0.5
    assert ax.spines["left"].get_position() == ("outward", 2.5)
    assert ax.xaxis.get_major_ticks()[0].tick1line.get_markersize() == 2.5
    assert ax.xaxis.label.get_fontsize() == 7
    assert all(t.get_fontsize() == 7 for t in ax.get_xticklabels())


def test_polish_unboxes_legends():
    fig, ax = demo_figure()
    polish(fig)
    assert not ax.get_legend().get_frame_on()
    assert ax.get_legend().get_texts()[0].get_fontsize() == 7


def test_polish_thins_lines_and_caps_marker_sizes():
    fig, ax = demo_figure()
    polish(fig)
    line = ax.lines[0]
    assert line.get_linewidth() == 1.0
    assert line.get_markersize() <= 3


def test_polish_keeps_dashed_reference_lines_thinner_than_data():
    fig, ax = demo_figure()
    ref = ax.axvline(1, color="black", ls="--", lw=1)
    polish(fig)
    assert ref.get_linewidth() == 0.5 < ax.lines[0].get_linewidth()


def test_polish_moves_the_suptitle_and_long_titles_to_the_caption():
    fig, ax = demo_figure(title="A very long descriptive title (n = 12 mice, 300 events)")
    fig.suptitle("Whole-figure description")
    caption = polish(fig)
    assert ax.get_title() == ""
    assert fig._suptitle is None or fig._suptitle.get_text() == ""
    assert "Whole-figure description" in caption
    assert "A very long descriptive title (n = 12 mice, 300 events)" in caption


def test_polish_keeps_a_short_title_and_a_multiline_title_goes_to_the_caption():
    fig, ax = demo_figure(title="Value of best")
    assert polish(fig) == ""
    assert ax.get_title() == "Value of best"
    fig2, ax2 = demo_figure(title="Line one\n(line two)")
    assert "Line one" in polish(fig2)
    assert ax2.get_title() == ""


def test_polish_moves_figure_level_notes_and_the_banner_to_the_caption():
    fig, _ = demo_figure()
    fig.text(0.5, 0.01, "Ties: 0")
    caption = polish(fig, banner="without MY_1_A")
    assert "Ties: 0" in caption and "without MY_1_A" in caption
    assert not fig.texts


def test_polish_uses_sentence_case_axis_labels():
    fig, ax = demo_figure()
    polish(fig)
    assert ax.get_xlabel() == "Trials from value change"
    assert ax.get_ylabel() == "Choice probability"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Proportion of Choices", "Proportion of choices"),
        ("Value of Best - Second", "Value of best - second"),
        ("P(best)", "P(best)"),
        ("Port Preference (Log2 Ratio)", "Port preference (log2 ratio)"),
        ("MY_175_L", "MY_175_L"),
        ("", ""),
    ],
)
def test_sentence_case(text, expected):
    assert sentence_case(text) == expected


def test_polish_resizes_to_a_column_width_and_uses_constrained_layout():
    fig, _ = demo_figure()
    polish(fig)
    assert fig.get_figwidth() * MM_PER_INCH == pytest.approx(89)
    assert fig.get_layout_engine().__class__.__name__ == "ConstrainedLayoutEngine"


def test_a_three_panel_figure_gets_the_double_column_width():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax in axes:
        ax.plot([0, 1], [0, 1])
    polish(fig)
    assert fig.get_figwidth() * MM_PER_INCH == pytest.approx(183)


def test_an_outside_legend_widens_a_single_panel_to_one_and_a_half_columns():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([0, 1], [0, 1], label="Best")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    polish(fig)
    assert fig.get_figwidth() * MM_PER_INCH == pytest.approx(120)


def test_polish_leaves_image_axes_with_all_four_thin_spines():
    fig, ax = plt.subplots()
    ax.imshow([[0, 1], [1, 0]])
    polish(fig)
    assert ax.spines["top"].get_visible()
    assert ax.spines["top"].get_linewidth() == 0.5


def test_polish_recolours_to_the_colourblind_safe_set_keeping_roles_distinct():
    fig, ax = plt.subplots()
    best, second = CHOICE_PROB_COLOR_MAP["prev_best"], CHOICE_PROB_COLOR_MAP["next_best"]
    ax.plot([0, 1], [0, 1], color=best)
    ax.plot([0, 1], [1, 0], color=second)
    ax.fill_between([0, 1], 0, 1, color=best, alpha=0.2)
    polish(fig)
    new_best, new_second = (to_hex(ln.get_color()) for ln in ax.lines)
    assert new_best != to_hex(best) and new_second != to_hex(second)
    assert new_best in OKABE_ITO and new_second in OKABE_ITO and new_best != new_second
    assert ax.collections[0].get_alpha() == 0.25
    assert ax.collections[0].get_linewidth()[0] == 0
    assert to_hex(ax.collections[0].get_facecolor()[0]) == new_best


def test_polish_is_idempotent_enough_to_run_twice():
    fig, ax = demo_figure(title="A very long descriptive title (n = 12 mice, 300 events)")
    first = polish(fig)
    assert polish(fig) == "" and first
