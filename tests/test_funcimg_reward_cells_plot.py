"""Figures of the most reward-responsive cells: small multiples, heatmap and field of view."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

from src.funcimg_analysis.reward_cells import analyse_reward_cells  # noqa: E402
from src.funcimg_visualization.plot_funcimg_responses import PULSE_COLORS, roi_colors  # noqa: E402
from src.funcimg_visualization.plot_funcimg_reward_cells import (  # noqa: E402
    plot_reward_cells,
    reward_cells_pdf_name,
)
from tests.funcimg_reward_fixtures import (  # noqa: E402
    SETTINGS,
    frame_times,
    make_cells,
    outcomes,
    small_run,
)

N_CELLS = 60


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def make_run(rank_on="split_half", n_cells=N_CELLS, contrast="rewarded", small=False):
    if small:  # 20 trials: 2 per pulse count in each half
        traces, times, outs = small_run(0, n_events=20, n_cells=n_cells)
    else:
        outs, frames, pulses = outcomes(np.random.default_rng(1))
        traces, times = (
            make_cells(np.random.default_rng(2), frames, pulses, n_cells, 20),
            frame_times(),
        )
    result = analyse_reward_cells(
        {"dF/F0": traces},
        times,
        outs,
        rng=np.random.default_rng(0),
        **{**SETTINGS, "rank_on": rank_on, "contrast": contrast, "n_shifts": 20},
    )
    labels = np.zeros((40, 6 * n_cells), np.int32)
    for i in range(n_cells):
        labels[5:15, 6 * i : 6 * i + 4] = i + 1
    return {
        **result,
        "p_cell": np.linspace(0.99, 0.4, n_cells),
        "labels": labels,
        "mean_image": np.random.default_rng(3).normal(100, 5, labels.shape),
        "rois": np.arange(n_cells),
        "session": "ses-224",
        "run": "run-01",
    }


def _all_text(fig):
    """The figure's own texts and the text of its summary axis."""
    summary = next(ax for ax in fig.axes if ax.get_label() == "summary")
    return " ".join(t.get_text() for t in [*fig.texts, *summary.texts])


def panels(fig):
    return [ax for ax in fig.axes if ax.get_label() == "cell"]


def test_one_panel_per_top_cell_in_rank_order_with_index_and_classifier_probability_titles():
    run = make_run()
    fig = plot_reward_cells(run, "dF/F0", n_cells=20)
    shown = panels(fig)
    assert len(shown) == 20
    measure = run["measures"]["dF/F0"]
    for ax, cell in zip(shown, measure["order"][:20], strict=True):
        title = ax.get_title()
        assert title.startswith(f"ROI {cell}")
        assert f"A {measure['index_a'][cell]:.2f}" in title
        assert f"B {measure['index_b'][cell]:.2f}" in title
        assert f"p={run['p_cell'][cell]:.2f}" in title


def test_each_panel_draws_the_held_out_mean_per_pulse_count_in_the_pulse_colours():
    run = make_run()
    fig = plot_reward_cells(run, "dF/F0", n_cells=20)
    ax = panels(fig)[0]
    cell = run["measures"]["dF/F0"]["order"][0]
    lines = [line for line in ax.lines if len(line.get_xdata()) > 5]
    assert [to_hex(line.get_color()) for line in lines] == [
        PULSE_COLORS[p].lower() for p in range(5)
    ]
    np.testing.assert_allclose(lines[4].get_ydata(), run["measures"]["dF/F0"]["means"][4][cell])
    x = lines[0].get_xdata()
    assert x[0] == pytest.approx(-2.0, abs=0.14) and x[-1] == pytest.approx(5.0, abs=0.14)


def test_panels_share_the_axes_so_that_sizes_can_be_compared():
    fig = plot_reward_cells(make_run(), "dF/F0", n_cells=20)
    assert len({ax.get_ylim() for ax in panels(fig)}) == 1


def test_panel_titles_use_the_colour_of_the_cells_outline_in_the_field_of_view():
    run = make_run()
    fig = plot_reward_cells(run, "dF/F0", n_cells=20)
    colours = roi_colors(20)
    assert [to_hex(ax.title.get_color()) for ax in panels(fig)] == colours
    fov = next(ax for ax in fig.axes if ax.get_label() == "fov")
    texts = {t.get_text(): to_hex(t.get_color()) for t in fov.texts}
    order = run["measures"]["dF/F0"]["order"][:20]
    assert texts == {str(cell): colour for cell, colour in zip(order, colours, strict=True)}


def test_the_heatmap_shows_every_ranked_cell_sorted_by_the_rank_index_on_the_held_out_trials():
    run = make_run()
    fig = plot_reward_cells(run, "dF/F0", n_cells=20)
    heat = next(ax for ax in fig.axes if ax.get_label() == "heatmap")
    measure = run["measures"]["dF/F0"]
    image = heat.images[0].get_array()
    assert image.shape == (len(measure["order"]), len(run["offsets"]))
    np.testing.assert_allclose(image[0], measure["rewarded_mean"][measure["order"][0]])
    assert "half A" in heat.get_title() and "half B" in heat.get_title()


def test_ranking_on_all_trials_says_so_loudly():
    fig = plot_reward_cells(make_run(rank_on="all"), "dF/F0", n_cells=20)
    texts = _all_text(fig).lower()
    assert "selection bias" in texts and "not cross-validated" in texts
    assert all("A " not in ax.get_title() for ax in panels(fig))


def test_the_cross_validated_figure_reports_stability_and_the_null_counts():
    run = make_run()
    text = _all_text(plot_reward_cells(run, "dF/F0", 20))
    null = run["measures"]["dF/F0"]["null"]
    assert f"{null['n_exceed_a']} of {null['n_cells']}" in text
    assert f"{null['n_exceed_b_of_a']}" in text and "rank correlation" in text
    assert "selection bias" not in text


def test_fewer_ranked_cells_than_requested_draws_fewer_panels():
    fig = plot_reward_cells(make_run(n_cells=8), "dF/F0", n_cells=20)
    assert len(panels(fig)) == 8


def test_an_unknown_measure_is_refused():
    with pytest.raises(KeyError, match="nothing"):
        plot_reward_cells(make_run(), "nothing", n_cells=20)


def test_pdf_name_carries_the_settings_that_shape_the_figure():
    name = reward_cells_pdf_name(
        20,
        "bg-corrected dF/F0",
        "rewarded",
        "split_half",
        "ses-224-224",
        "without_destriping",
        (-2.0, 5.0),
    )
    assert name == (
        "reward_cells_top20_bgcorrecteddFF0_rewarded_split_half_ses-224-224"
        "_without_destriping_win-2to5s.pdf"
    )


def ribbons(ax):
    return [c for c in ax.collections if c.get_alpha() == 0.2]


def test_each_panels_legend_gives_the_trials_per_pulse_count_of_the_plotted_half():
    run = make_run()
    fig = plot_reward_cells(run, "dF/F0", n_cells=20)
    expected = [f"{p}: n={n}" for p, n in run["n_trials_by_half"]["b"].items()]
    for ax in panels(fig):
        assert [t.get_text() for t in ax.get_legend().get_texts()] == expected


def test_all_trials_ranking_gives_the_counts_of_all_trials():
    run = make_run(rank_on="all")
    ax = panels(plot_reward_cells(run, "dF/F0", n_cells=5))[0]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        f"{p}: n={n}" for p, n in run["n_trials_by_half"]["all"].items()
    ]


def test_no_sem_ribbon_is_drawn_for_a_pulse_count_with_fewer_than_three_trials_and_it_says_so():
    run = make_run(small=True, n_cells=30)
    assert set(run["n_trials_by_half"]["b"].values()) == {2}
    fig = plot_reward_cells(run, "dF/F0", n_cells=6)
    assert all(ribbons(ax) == [] for ax in panels(fig))
    assert "fewer than 3 trials" in _all_text(fig)
    full = make_run()  # 8 trials per pulse count in half B
    assert all(len(ribbons(ax)) == 5 for ax in panels(plot_reward_cells(full, "dF/F0", 6)))


def test_the_heatmap_of_the_four_vs_zero_contrast_shows_that_difference():
    run = make_run(contrast="4_vs_0")
    heat = next(
        ax for ax in plot_reward_cells(run, "dF/F0", 20).axes if ax.get_label() == "heatmap"
    )
    measure = run["measures"]["dF/F0"]
    np.testing.assert_allclose(
        heat.images[0].get_array()[0], measure["ranked_mean"][measure["order"][0]]
    )
    assert "4-pulse minus 0-pulse" in heat.get_title()


def test_the_rewarded_index_is_described_as_any_response_to_reward_not_pulse_specific():
    run = make_run()
    fig = plot_reward_cells(run, "dF/F0", 20)
    text = _all_text(fig) + " ".join(ax.get_title() for ax in fig.axes)
    assert "any response to reward" in text and "not specific to the number of pulses" in text


def test_the_summary_says_the_null_is_approximate_and_that_roi_colours_repeat():
    text = _all_text(plot_reward_cells(make_run(), "dF/F0", 20))
    assert "approximate" in text and "colours repeat" in text


def test_the_figure_can_be_drawn_into_a_given_figure():
    figure = plt.figure(figsize=(20, 12))
    assert plot_reward_cells(make_run(), "dF/F0", 20, figure=figure) is figure
    assert len(panels(figure)) == 20
