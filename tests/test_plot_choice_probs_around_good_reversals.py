import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.behavior_visualization.plot_choice_probs_around_good_reversals import (
    add_cumulative_rev_axis,
    cumulative_reversal_events_over_post,
    cumulative_total_events_over_post,
    plot_choice_probs,
    plot_choice_probs_around_good_reversals,
    plot_choice_probs_broken_axis,
    plot_choice_probs_with_removal,
    plot_cumulative_removed,
)

X = np.arange(-3, 12)
WINDOWS = {"A": [{"reversal_idx": 10}, {"reversal_idx": 20}], "B": [{"reversal_idx": 4}]}
GOOD = {"A": [10], "B": [4]}
BAD = {"A": [20]}
SERIES = {"good": np.where(X >= 0, 0.2, np.nan), "bad": np.where(X >= 0, 0.1, np.nan)}
GOODBAD_MESSAGE = "For mode='goodbad', provide all_good_idx and all_bad_idx."
TITLE = "Good Reversal-Aligned Choices\n(mean ± se across subjects | n=2 subjects, n=3 reversals)\n"


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def shown(monkeypatch):
    """Figures at the moment ``plt.show`` is called, in call order (nothing is displayed)."""
    figs = []
    monkeypatch.setattr(plt, "show", lambda: figs.append(plt.gcf()))
    return figs


@pytest.fixture
def across():
    probs = {"prev_best": np.full(X.size, 0.5), "next_best": np.full(X.size, 0.3)}
    probs["third"] = 1 - probs["prev_best"] - probs["next_best"]
    return {
        "mean": probs,
        "se": {k: np.full(X.size, 0.05) for k in probs},
        "num_subjects": 2,
        "num_reversals": 3,
    }


def cumulative_figure(shown, across, **kwargs):
    """Run the public plot with a cumulative axis; return (figure, host axis, twin axis)."""
    plot_choice_probs_around_good_reversals(
        X,
        across,
        windows_for_cumulative_axis=WINDOWS,
        all_good_idx=GOOD,
        all_bad_idx=BAD,
        **kwargs,
    )
    fig = shown[-1]
    host, twin = fig.axes
    return fig, host, twin


# --- panels -----------------------------------------------------------------


def test_choice_prob_panels_draw_into_given_axes(across):
    fig, (a, b, c) = plt.subplots(1, 3)
    assert plot_choice_probs(X, across, ax=a) is a
    assert len(a.lines) == 3 + 2  # three arms plus the two reference lines
    ax_main, ax_twin = plot_choice_probs_with_removal(X, across, SERIES, fraction=True, ax=b)
    assert ax_main is b and ax_twin.get_ylim() == (0, 1.05)
    twin_c = plot_cumulative_removed(X, SERIES, fraction=True, hidden=("bad",), ax=c)
    # the returned axis is a twin of c (shared x) carrying only the un-hidden series
    assert twin_c is not c and c.get_shared_x_axes().joined(c, twin_c)
    assert [ln.get_label() for ln in twin_c.lines] == ["Good Rev"]
    assert len(fig.axes) == 3 + 2  # a twin on each of the second and third panels


def test_panels_create_a_figure_only_without_ax(across):
    assert not plt.get_fignums()
    plot_choice_probs(X, across)
    assert len(plt.get_fignums()) == 1


def test_plain_panel_appearance(across):
    ax = plot_choice_probs(X, across)
    assert ax.get_ylim() == pytest.approx((-0.05, 1.05))
    assert ax.get_legend()._loc == 1  # upper right
    assert ax.get_title() == TITLE
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Trials from Good Reversal", "Choice Probability")


def test_removal_panel_appearance(across):
    ax, ax2 = plot_choice_probs_with_removal(X, across, SERIES, fraction=True)
    assert ax.get_ylim() == (0, 1)  # the choice-probability scale is the unit interval
    legend = ax.get_legend()
    assert legend._loc == 2  # upper left
    assert [t.get_text() for t in legend.get_texts()] == [
        "Previous Best",
        "New Best",
        "Third Arm",
        "Good Rev",
        "Bad Rev",
    ]
    assert ax.get_title() == TITLE
    assert ax2.get_ylabel() == "Fraction of Data Removed"


def test_removal_series_style_and_step_shape():
    series = {k: np.array([0.0, 0.2]) for k in ("total", "good", "bad")}
    ax2 = plot_cumulative_removed(np.arange(2), series, fraction=True)
    styles = {
        ln.get_label(): (ln.get_color(), ln.get_alpha(), ln.get_linewidth(), ln.get_drawstyle())
        for ln in ax2.lines
    }
    assert styles == {
        "Total Rev": ("#FF9100", 0.7, 2.0, "steps-post"),
        "Good Rev": ("#FF9100", 0.7, 2.0, "steps-post"),
        "Bad Rev": ("#CD0000", 0.7, 2.0, "steps-post"),
    }


@pytest.mark.parametrize(
    ("fraction", "label", "ylim"),
    [
        (True, "Fraction of Data Removed", (0, 1.05)),
        (False, "Cumulative Reversals", (0, 5 * 1.05 + 1)),
    ],
)
def test_twin_axis_label_and_limits(fraction, label, ylim):
    ax2 = plot_cumulative_removed(
        np.arange(3), {"good": np.array([0.0, 1.0, 5.0])}, fraction=fraction
    )
    assert ax2.get_ylabel() == label
    assert ax2.yaxis.label.get_rotation() == 270  # matplotlib normalises -90
    assert ax2.yaxis.labelpad == 15
    assert ax2.get_ylim() == pytest.approx(ylim)
    assert not ax2.spines["top"].get_visible()


def test_removal_axis_sits_behind_a_transparent_host_axis():
    _, ax = plt.subplots()
    series = {"good": np.array([0.0, 1.0, 2.0])}
    ax2 = plot_cumulative_removed(np.arange(3), series, fraction=True, ax=ax)
    # otherwise the host's background hides the twin (matplotlib already hides it once the
    # host is raised, so this pins the outcome rather than the explicit call)
    assert not ax.patch.get_visible()
    assert ax.get_zorder() > ax2.get_zorder()


def test_removal_axis_hides_labelled_series_but_keeps_them_for_the_axis_limits():
    series = {"good": np.array([0.0, 1.0, 2.0]), "bad": np.array([0.0, 4.0, 4.0])}
    _, ax = plt.subplots()
    ax2 = plot_cumulative_removed(np.arange(3), series, fraction=False, hidden=("bad",), ax=ax)
    assert [ln.get_label() for ln in ax2.lines] == ["Good Rev"]
    assert ax2.get_ylim()[1] == pytest.approx(4 * 1.05 + 1)


def test_broken_axis_panel_draws_into_given_axes(across):
    skip = 4
    fig, axes = plt.subplots(1, 2)
    left, right = plot_choice_probs_broken_axis(X, across, skip, axes=axes)
    assert (left, right) == tuple(axes) and len(fig.axes) == 2
    assert left.get_xlim() == (X.min(), 0)
    assert right.get_xlim()[0] == skip + 1
    # post-reversal offsets are shifted right by the skipped trials; the input is not touched
    arm = right.lines[0]
    assert arm.get_xdata().min() == 1 + skip and arm.get_xdata().max() == X.max() + skip
    assert X.tolist() == list(range(-3, 12))


def test_broken_axis_adds_the_start_of_the_right_segment_as_a_tick(across):
    # skip=4: the integer locator alone would start the right axis at 6, hiding where it begins
    _, right = plot_choice_probs_broken_axis(X, across, 4)
    ticks = right.get_xticks().tolist()
    assert ticks[0] == 5 and ticks[1] > 5


def test_broken_axis_legend_is_on_the_right_segment(across):
    left, right = plot_choice_probs_broken_axis(X, across, 4)
    assert left.get_legend() is None
    assert right.get_legend()._loc == 1
    assert [t.get_text() for t in right.get_legend().get_texts()] == [
        "Previous Best",
        "New Best",
        "Third Arm",
    ]


# --- public figures ----------------------------------------------------------


@pytest.mark.parametrize("skip", [0, 4])
def test_public_plot_saves_pdf_and_png_and_closes_figure(across, tmp_path, skip):
    save_path = tmp_path / "out" / "choice probs"
    plot_choice_probs_around_good_reversals(
        X, across, add_cumulative_axis=False, skip_n_trials_after_reversal=skip, save_path=save_path
    )
    assert sorted(p.name for p in save_path.parent.iterdir()) == [
        "choice probs.pdf",
        "choice probs.png",
    ]
    assert not plt.get_fignums()


def test_public_broken_axis_figure_layout(across, shown):
    plot_choice_probs_around_good_reversals(
        X, across, add_cumulative_axis=False, skip_n_trials_after_reversal=4
    )
    (fig,) = shown
    assert tuple(fig.get_size_inches()) == (10, 4.5)
    assert fig._suptitle.get_text() == TITLE + "(Skipping first 4 trials post-reversal)"
    assert fig._suptitle.get_position()[1] == 1.03
    assert fig._supxlabel.get_text() == "Trials from Good Reversal"
    left, right = fig.axes
    # the skipped trials are applied to the raw offsets exactly once
    assert left.lines[0].get_xdata().tolist() == [-3, -2, -1, 0]
    assert right.lines[0].get_xdata().tolist() == list(range(1 + 4, 11 + 4 + 1))


@pytest.mark.parametrize("use_total", [True, False])
def test_public_plot_with_cumulative_axis_saves_the_removal_figure(across, tmp_path, use_total):
    save_path = tmp_path / "cum"
    plot_choice_probs_around_good_reversals(
        X,
        across,
        use_total=use_total,
        windows_for_cumulative_axis=WINDOWS,
        all_good_idx=GOOD,
        all_bad_idx=BAD,
        save_path=save_path,
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cum.pdf", "cum.png"]
    assert plt.get_fignums() == []  # no stray plain figure is left open


def test_public_plot_with_cumulative_axis_shows_only_the_removal_figure(across, shown, monkeypatch):
    open_at_show = []
    monkeypatch.setattr(plt, "show", lambda: open_at_show.append(len(plt.get_fignums())))
    plot_choice_probs_around_good_reversals(
        X, across, windows_for_cumulative_axis=WINDOWS, all_good_idx=GOOD, all_bad_idx=BAD
    )
    # a real show displays every open figure, so exactly one must be open
    assert open_at_show == [1]
    assert plt.get_fignums() == []


def test_public_plot_with_cumulative_axis_appearance(across, shown):
    fig, host, twin = cumulative_figure(shown, across)
    assert tuple(fig.get_size_inches()) == (10, 6)
    assert host.get_ylim() == (0, 1) and host.get_legend()._loc == 2
    assert twin.get_ylim() == pytest.approx((0, 1.05))


def test_public_plot_good_bad_mode_draws_both_series(across, shown):
    _, _, twin = cumulative_figure(shown, across)
    assert [ln.get_label() for ln in twin.lines] == ["Good Rev", "Bad Rev"]


def test_public_plot_only_good_hides_the_bad_series(across, shown):
    _, host, twin = cumulative_figure(shown, across, only_good=True)
    assert [ln.get_label() for ln in twin.lines] == ["Good Rev"]
    assert [t.get_text() for t in host.get_legend().get_texts()][-1] == "Good Rev"


def test_public_plot_total_mode_draws_one_series(across, shown):
    _, _, twin = cumulative_figure(shown, across, use_total=True)
    assert [ln.get_label() for ln in twin.lines] == ["Total Rev"]


@pytest.mark.parametrize(
    ("plot_fraction_removed", "label"),
    [(True, "Fraction of Data Removed"), (False, "Cumulative Reversals")],
)
def test_add_cumulative_rev_axis_fraction_or_counts(across, shown, plot_fraction_removed, label):
    add_cumulative_rev_axis(
        X, across, WINDOWS, GOOD, BAD, plot_fraction_removed=plot_fraction_removed, use_total=False
    )
    (fig,) = shown
    twin = fig.axes[1]
    assert tuple(fig.get_size_inches()) == (10, 6)
    assert fig._suptitle is None
    assert twin.get_ylabel() == label
    assert len(shown) == 1 and not plt.get_fignums()  # only its own figure, closed after show
    # anchor 10 is cut by the bad reversal at 20, of 3 anchors
    bad_line = twin.lines[1].get_ydata()
    assert np.nanmax(bad_line) == pytest.approx(1 / 3 if plot_fraction_removed else 1)
    assert twin.get_ylim() == pytest.approx((0, 1.05 if plot_fraction_removed else 1 * 1.05 + 1))


def test_good_bad_cumulative_axis_without_indices_fails_loudly(across):
    with pytest.raises(ValueError, match=re.escape(GOODBAD_MESSAGE)):
        plot_choice_probs_around_good_reversals(X, across, windows_for_cumulative_axis=WINDOWS)


def test_cumulative_wrappers_return_offsets_first_and_do_not_alias_input():
    x = np.arange(-3, 12, dtype=float)
    x_disp, cum_good, cum_bad, frac_good, frac_bad = cumulative_reversal_events_over_post(
        WINDOWS, GOOD, BAD, x, across=None
    )
    assert x_disp.tolist() == x.tolist() and x_disp is not x
    assert cum_good.shape == cum_bad.shape == frac_good.shape == frac_bad.shape == x.shape

    x_disp, cum_total, frac_total = cumulative_total_events_over_post(WINDOWS, x, across=None)
    assert x_disp.tolist() == x.tolist() and x_disp is not x
    # only anchor 10 -> 20 (+10) is cut within the window; of 3 anchors
    assert cum_total[-1] == 1 and frac_total[-1] == pytest.approx(1 / 3)


# --- fine appearance details ------------------------------------------------


def test_chance_line_sits_at_one_third_and_se_bands_are_translucent(across):
    ax = plot_choice_probs(X, across)
    chance = [ln for ln in ax.lines if ln.get_linestyle() == ":"]
    assert len(chance) == 1
    assert list(chance[0].get_ydata()) == pytest.approx([1 / 3, 1 / 3])
    assert len(ax.collections) == 3
    assert [c.get_alpha() for c in ax.collections] == [0.2, 0.2, 0.2]


@pytest.mark.parametrize(
    "draw",
    [
        plot_choice_probs,
        lambda x, a: plot_choice_probs_with_removal(x, a, SERIES, fraction=True)[0],
    ],
)
def test_title_padding_and_legend_font_size(across, draw):
    ax = draw(X, across)
    pad_in_points = ax.titleOffsetTrans.transform((0, 0))[1] * 72 / ax.figure.dpi
    assert pad_in_points == pytest.approx(20)
    assert ax.get_legend().get_texts()[0].get_fontsize() == 10


def test_png_is_saved_at_300_dpi(shown, across, tmp_path):
    from PIL import Image

    plot_choice_probs_around_good_reversals(
        X, across, add_cumulative_axis=False, save_path=tmp_path / "fig"
    )
    with Image.open(tmp_path / "fig.png") as img:
        assert round(img.info["dpi"][0]) == 300
