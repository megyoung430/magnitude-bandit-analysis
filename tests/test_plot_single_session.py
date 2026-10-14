import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.behavior_visualization.plot_single_session import (
    event_indices_from_cumulative,
    plot_choices_by_rank,
    plot_choices_by_tower,
    plot_ema_traces,
    plot_magnitudes_and_choices,
    plot_single_session,
)

TOWERS = ["A1", "B1", "C1"]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def session():
    n = 12
    chosen = np.arange(n) % 3
    return {
        "reward_magnitudes_by_tower": {
            "A1": [4] * n,
            "B1": [1] * n,
            "C1": [0] * n,
        },
        "choices_by_tower": {t: [int(c == j) for c in chosen] for j, t in enumerate(TOWERS)},
        "good_reversals": [0] * 6 + [1] * 6,
        "bad_reversals": [0] * 9 + [1] * 3,
        "rank_counts": [{"best": 1}, {"best": 6, "second": 4, "third": 2}],
        "ema_best_arm_choices": np.linspace(0, 1, n).tolist(),
        "ema_second_arm_choices": np.linspace(1, 0, n).tolist(),
    }


def test_event_indices_still_importable_from_plot_module():
    assert event_indices_from_cumulative([0, 1, 1, 2]) == [1, 3]


def test_panels_draw_into_given_axes_without_creating_a_figure(session):
    fig, axes = plt.subplots(2, 2)
    mags, choices = session["reward_magnitudes_by_tower"], session["choices_by_tower"]
    plot_magnitudes_and_choices(mags, choices, TOWERS, ([6], [9], []), ax=axes[0, 0])
    plot_ema_traces(session, ("ema_best_arm_choices",), ([6], [9], []), 12, ax=axes[0, 1])
    plot_choices_by_tower(choices, TOWERS, 12, ax=axes[1, 0])
    plot_choices_by_rank({"best": 6, "second": 4}, ax=axes[1, 1])
    assert plt.get_fignums() == [fig.number]
    assert len(axes[0, 0].lines) == 3 + 2  # tower traces + good and bad reversal lines
    assert [p.get_height() for p in axes[1, 0].patches] == [4, 4, 4]


def test_panels_create_their_own_figure_when_no_ax_given(session):
    ax = plot_choices_by_tower(session["choices_by_tower"], TOWERS, 12)
    assert len(plt.get_fignums()) == 1
    assert ax.figure.number in plt.get_fignums()


def test_reversal_lines_label_only_the_first_of_each_kind(session):
    mags, choices = session["reward_magnitudes_by_tower"], session["choices_by_tower"]
    ax = plot_magnitudes_and_choices(mags, choices, TOWERS, ([2, 5], [8, 9], []))
    labels = [line.get_label() for line in ax.lines]
    assert labels.count("Good Rev") == 1
    assert labels.count("Bad Rev") == 1


def test_block_lines_used_only_when_no_good_or_bad_reversals(session):
    mags, choices = session["reward_magnitudes_by_tower"], session["choices_by_tower"]
    ax = plot_magnitudes_and_choices(mags, choices, TOWERS, ([], [], [3, 7]))
    labels = [line.get_label() for line in ax.lines][3:]
    assert len(labels) == 2 and labels[0] == "Rev"


def test_rank_panel_shows_placeholder_without_rank_data():
    ax = plot_choices_by_rank(None)
    assert [t.get_text() for t in ax.texts] == ["No rank data"]
    assert len(ax.patches) == 0


def test_tower_panel_labels_percentages_and_survives_no_choices():
    ax = plot_choices_by_tower({t: [0, 0] for t in TOWERS}, TOWERS, 2)
    assert [t.get_text() for t in ax.texts] == ["0%", "0%", "0%"]


def test_plot_single_session_saves_pdf_and_png(session, tmp_path):
    plot_single_session(session, title="x", save_path=tmp_path / "out" / "Session Plot")
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "Session Plot.pdf",
        "Session Plot.png",
    ]
    assert plt.get_fignums() == []


def test_plot_single_session_without_ema_has_three_panels(session, monkeypatch):
    for key in ("ema_best_arm_choices", "ema_second_arm_choices"):
        del session[key]
    seen = []
    monkeypatch.setattr(plt, "show", lambda: seen.append(len(plt.gcf().axes)))
    plot_single_session(session)
    assert seen == [3]


def test_plot_single_session_rejects_unnormalised_probabilities(session):
    probs = {t: [0.9] * 12 for t in TOWERS}
    with pytest.raises(ValueError, match="normalized"):
        plot_single_session(session, choice_probabilities_by_tower=probs)


@pytest.mark.parametrize("not_a_session", [[1, 2, 3], None])
def test_plot_single_session_rejects_non_mapping_session_with_type_error(not_a_session):
    with pytest.raises(TypeError):
        plot_single_session(not_a_session)
