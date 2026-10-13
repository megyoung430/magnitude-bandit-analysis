import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.behavior_visualization.plot_choice_probs_around_good_reversals import (
    _skipped_offsets,
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


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


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


def test_skipped_offsets_shift_only_post_reversal_trials():
    x = np.array([-2, 0, 1, 5])
    assert _skipped_offsets(x, 3).tolist() == [-2, 0, 4, 8]
    assert _skipped_offsets(x, 0).tolist() == [-2, 0, 1, 5]
    assert x.tolist() == [-2, 0, 1, 5]  # input untouched


def test_choice_prob_panels_draw_into_given_axes(across):
    fig, (a, b, c) = plt.subplots(1, 3)
    assert plot_choice_probs(X, across, ax=a) is a
    assert len(a.lines) == 3 + 2  # three arms plus the two reference lines
    series = {"Good Rev": np.where(X >= 0, 0.2, np.nan), "Bad Rev": np.where(X >= 0, 0.1, np.nan)}
    ax_main, ax_twin = plot_choice_probs_with_removal(X, across, series, fraction=True, ax=b)
    assert ax_main is b and ax_twin.get_ylim() == (0, 1.05)
    assert plot_cumulative_removed(X, series, fraction=True, hidden=("Bad Rev",), ax=c) is not c
    assert len(fig.axes) == 3 + 2  # a twin on each of the second and third panels


def test_broken_axis_panel_draws_into_given_axes(across):
    skip = 4
    x_disp = _skipped_offsets(X, skip)
    fig, axes = plt.subplots(1, 2)
    left, right = plot_choice_probs_broken_axis(x_disp, across, skip, axes=axes)
    assert (left, right) == tuple(axes) and len(fig.axes) == 2
    assert left.get_xlim() == (x_disp.min(), 0)
    assert right.get_xlim()[0] == skip + 1


def test_removal_axis_hides_labelled_series_but_keeps_them_for_the_axis_limits():
    series = {"Good Rev": np.array([0.0, 1.0, 2.0]), "Bad Rev": np.array([0.0, 4.0, 4.0])}
    fig, ax = plt.subplots()
    ax2 = plot_cumulative_removed(np.arange(3), series, fraction=False, hidden=("Bad Rev",), ax=ax)
    assert [ln.get_label() for ln in ax2.lines] == ["Good Rev"]
    assert ax2.get_ylim()[1] == pytest.approx(4 * 1.05 + 1)


def test_panels_create_a_figure_only_without_ax(across):
    assert not plt.get_fignums()
    plot_choice_probs(X, across)
    assert len(plt.get_fignums()) == 1


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


def test_good_bad_cumulative_axis_without_indices_fails_loudly(across):
    with pytest.raises(ValueError, match="all_good_idx and all_bad_idx"):
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
