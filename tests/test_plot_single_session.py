import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from src.behavior_analysis.single_session import chosen_magnitudes
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
    plot_ema_traces(session, ("ema_best_arm_choices",), ([6], [9], []), N=12, ax=axes[0, 1])
    plot_choices_by_tower(choices, TOWERS, N=12, ax=axes[1, 0])
    plot_choices_by_rank({"best": 6, "second": 4}, ax=axes[1, 1])
    assert plt.get_fignums() == [fig.number]
    assert len(axes[0, 0].lines) == 3 + 2  # tower traces + good and bad reversal lines
    assert [p.get_height() for p in axes[1, 0].patches] == [4, 4, 4]


def test_panels_create_their_own_figure_when_no_ax_given(session):
    ax = plot_choices_by_tower(session["choices_by_tower"], TOWERS, N=12)
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
    ax = plot_choices_by_tower({t: [0, 0] for t in TOWERS}, TOWERS, N=2)
    assert [t.get_text() for t in ax.texts] == ["0%", "0%", "0%"]


def test_plot_single_session_saves_every_format_and_a_caption(session, tmp_path):
    plot_single_session(session, title="x", save_path=tmp_path / "out" / "Session Plot")
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "Session Plot.caption.txt",
        "Session Plot.pdf",
        "Session Plot.png",
        "Session Plot.svg",
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


# ---------------------------------------------------------------------------
# What the assembled figure looks like (characterization through plot_single_session)
# ---------------------------------------------------------------------------

GOOD, BAD = "#3a982e", "#f97979"


@pytest.fixture
def draw(monkeypatch):
    """Call plot_single_session and return the figure it showed (it closes it afterwards)."""

    def _draw(session, **kwargs):
        shown = []
        monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))
        plot_single_session(session, **kwargs)
        return shown[0]

    return _draw


def _lines_labelled(ax, label):
    return [line for line in ax.lines if line.get_label() == label]


def _colour(artist):
    return matplotlib.colors.to_hex(artist.get_color())


def _legend_texts(ax):
    return [t.get_text() for t in ax.get_legend().get_texts()]


def test_figure_with_ema_has_four_panels_with_titles_and_labels(session, draw):
    fig = draw(session, title="My title")
    assert tuple(fig.get_size_inches()) == (14, 11)
    magnitudes, ema, towers, ranks = fig.axes
    assert magnitudes.get_title() == "My title"
    assert [a.get_ylabel() for a in fig.axes] == [
        "Reward Magnitude",
        "EMA",
        "Count by Tower",
        "Count by Rank",
    ]
    assert ema.get_xlabel() == "Trial"


def test_default_title_is_used_when_none_given(session, draw):
    assert draw(session).axes[0].get_title() == "Reward Magnitudes & Choices"


def test_magnitude_panel_ticks_tower_colours_and_legend(session, draw):
    magnitudes = draw(session).axes[0]
    assert list(magnitudes.get_yticks()) == [0, 1, 4]
    tower_lines = magnitudes.lines[:3]
    assert [line.get_label() for line in tower_lines] == TOWERS
    assert [_colour(line) for line in tower_lines] == ["#a6cee3", "#fdbf6f", "#b2df8a"]
    assert sorted(_legend_texts(magnitudes)) == sorted([*TOWERS, "Choice", "Good Rev", "Bad Rev"])


def test_good_and_bad_reversal_lines_sit_half_a_trial_before_the_event(session, draw):
    fig = draw(session)
    for ax in fig.axes[:2]:  # magnitude and EMA panels share the reversal markers
        (good,) = _lines_labelled(ax, "Good Rev")
        (bad,) = _lines_labelled(ax, "Bad Rev")
        assert good.get_xdata()[0] == 6 - 0.5 and _colour(good) == GOOD
        assert bad.get_xdata()[0] == 9 - 0.5 and _colour(bad) == BAD


def test_block_lines_are_drawn_only_when_there_are_no_good_or_bad_reversals(session, draw):
    session["blocks"] = [0] * 4 + [1] * 4 + [2] * 4
    with_reversals = draw(session).axes[0]
    assert _lines_labelled(with_reversals, "Rev") == []

    for key in ("good_reversals", "bad_reversals"):
        del session[key]
    blocks_only = draw(session).axes[0]
    (block_line,) = _lines_labelled(blocks_only, "Rev")
    assert block_line.get_xdata()[0] == 4 - 0.5 and _colour(block_line) == GOOD
    assert len(blocks_only.lines) == 3 + 2  # tower traces + one line per block start
    assert _lines_labelled(blocks_only, "Good Rev") == []
    assert _lines_labelled(blocks_only, "Bad Rev") == []


def test_ema_panel_draws_both_traces_and_the_threshold(session, draw):
    ema = draw(session).axes[1]
    assert _legend_texts(ema) == ["Best", "Second", "Threshold", "Good Rev", "Bad Rev"]
    assert [_colour(line) for line in ema.lines[:2]] == ["#000000", "#858585"]
    (threshold,) = _lines_labelled(ema, "Threshold")
    assert list(threshold.get_ydata()) == [0.7, 0.7]
    assert tuple(ema.get_ylim()) == (0, 1)


def test_bar_panels_show_heights_and_percent_labels(session, draw):
    _, _, towers, ranks = draw(session).axes
    assert [t.get_text() for t in towers.get_xticklabels()] == TOWERS
    assert [p.get_height() for p in towers.patches] == [4, 4, 4]
    assert [t.get_text() for t in towers.texts] == ["33%", "33%", "33%"]
    assert [t.get_text() for t in ranks.get_xticklabels()] == ["best", "second", "third"]
    assert [p.get_height() for p in ranks.patches] == [6, 4, 2]
    assert [t.get_text() for t in ranks.texts] == ["50%", "33%", "17%"]


def test_figure_without_ema_has_three_panels_and_labels_the_top_x_axis(session, draw):
    for key in ("ema_best_arm_choices", "ema_second_arm_choices"):
        del session[key]
    fig = draw(session)
    assert tuple(fig.get_size_inches()) == (14, 8)
    assert [a.get_ylabel() for a in fig.axes] == [
        "Reward Magnitude",
        "Count by Tower",
        "Count by Rank",
    ]
    assert fig.axes[0].get_xlabel() == "Trial"


def test_probabilities_add_a_right_hand_twin_axis_with_its_own_legend(session, draw):
    probabilities = {t: [1 / 3] * 12 for t in TOWERS}
    fig = draw(session, choice_probabilities_by_tower=probabilities)
    assert len(fig.axes) == 5
    (twin,) = [a for a in fig.axes if a.get_ylabel() == "Vanilla RNN choice probability"]
    assert tuple(twin.get_ylim()) == (0, 1)
    assert _legend_texts(twin) == ["RNN P(A1)", "RNN P(B1)", "RNN P(C1)"]


def test_no_twin_axis_without_probabilities(session, draw):
    assert len(draw(session).axes) == 4


def test_chosen_arm_dots_are_the_chosen_magnitudes_in_black(session, draw):
    magnitudes = draw(session).axes[0]
    (dots,) = magnitudes.collections
    expected = chosen_magnitudes(
        session["reward_magnitudes_by_tower"], session["choices_by_tower"], towers=TOWERS, N=12
    )
    assert list(dots.get_offsets()[:, 0]) == list(range(12))
    assert list(dots.get_offsets()[:, 1]) == list(expected)
    assert list(expected[:3]) == [4, 1, 0]  # guards against both sides being trivially equal
    assert matplotlib.colors.to_hex(dots.get_facecolor()[0]) == "#000000"


def test_percent_labels_sit_centred_on_top_of_their_bars(session, draw):
    _, _, towers, ranks = draw(session).axes
    for ax in (towers, ranks):
        for bar, label in zip(ax.patches, ax.texts, strict=True):
            assert label.get_position() == (bar.get_x() + bar.get_width() / 2, bar.get_height())
            assert label.get_va() == "bottom" and label.get_ha() == "center"


def test_bar_colours_follow_tower_and_rank_palettes(session, draw):
    _, _, towers, ranks = draw(session).axes
    assert [matplotlib.colors.to_hex(p.get_facecolor()) for p in towers.patches] == [
        "#a6cee3",
        "#fdbf6f",
        "#b2df8a",
    ]
    assert [matplotlib.colors.to_hex(p.get_facecolor()) for p in ranks.patches] == [
        "#000000",
        "#858585",
        "#cccccc",
    ]


def test_magnitude_and_ema_panels_share_the_trial_axis(session, draw):
    magnitudes, ema, towers, _ = draw(session).axes
    assert magnitudes.get_shared_x_axes().joined(magnitudes, ema)
    assert not magnitudes.get_shared_x_axes().joined(magnitudes, towers)


def test_saved_files_use_tight_bbox_and_png_is_300_dpi(session, tmp_path, monkeypatch):
    calls = []
    original = Figure.savefig

    def recording_savefig(self, fname, **kwargs):
        calls.append((Path(fname).suffix, kwargs))
        return original(self, fname, **kwargs)

    monkeypatch.setattr(Figure, "savefig", recording_savefig)
    plot_single_session(session, save_path=tmp_path / "out")
    assert calls == [
        (".pdf", {"bbox_inches": "tight"}),
        (".png", {"dpi": 300, "bbox_inches": "tight"}),
        (".svg", {"bbox_inches": "tight"}),
    ]


def test_drawing_emits_no_layout_warning(session, draw):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        draw(session, choice_probabilities_by_tower={t: [1 / 3] * 12 for t in TOWERS})


def test_an_empty_save_path_shows_the_figure_instead_of_writing_files(
    session, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))
    plot_single_session(session, save_path="")
    assert len(shown) == 1 and list(tmp_path.iterdir()) == [] and plt.get_fignums() == []
