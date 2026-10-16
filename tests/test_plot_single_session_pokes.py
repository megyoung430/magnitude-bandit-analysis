"""The session-level maze heatmaps added to the single-session figure."""

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.behavior_analysis.poke_counts import session_poke_counts, unparsed_session_counts
from src.behavior_analysis.poke_events import PokeAlignmentError
from src.behavior_visualization import plot_single_session as module
from src.behavior_visualization.plot_single_session import plot_single_session
from tests.poke_fixtures import Recording, full_trial, session_record

TOWERS = ["A1", "B1", "C1"]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def figures(monkeypatch):
    """Figures ``plot_single_session`` hands over for saving."""
    captured = []
    monkeypatch.setattr(module, "save_or_show", lambda fig, path: captured.append(fig))
    return captured


def base_session(with_ema=True):
    n = 3
    session = {
        "reward_magnitudes_by_tower": {"A1": [4] * n, "B1": [1] * n, "C1": [0] * n},
        "choices_by_tower": {"A1": [1, 0, 0], "B1": [0, 1, 0], "C1": [0, 0, 1]},
        "good_reversals": [0, 0, 0],
        "bad_reversals": [0, 0, 0],
        "rank_counts": [{"best": 1}, {"best": 1, "second": 1}, {"best": 1, "second": 1}],
    }
    if with_ema:
        session["ema_best_arm_choices"] = [0.1, 0.2, 0.3]
        session["ema_second_arm_choices"] = [0.3, 0.2, 0.1]
    return session


def session_with_pokes(with_ema=True):
    rec = Recording()
    full_trial(rec, long_before=["A3"], choice="A1", after_choice=["A1", "C3", "C3"])
    full_trial(rec, choice="B1", after_choice=["A2"])
    full_trial(rec, choice="C1")
    return {**base_session(with_ema), **session_record([rec.to_frame()])}


def inches(fig, ax):
    box = ax.get_position()
    width, height = fig.get_size_inches()
    return np.array([box.x0 * width, box.y0 * height, box.width * width, box.height * height])


def ring_ids(fig):
    return sorted(p.get_gid() for ax in fig.axes for p in ax.patches if p.get_gid())


@pytest.mark.parametrize("with_ema", [True, False])
def test_session_without_raw_recordings_gets_the_figure_it_always_had(figures, with_ema):
    plot_single_session(base_session(with_ema))
    (fig,) = figures
    assert len(fig.axes) == (4 if with_ema else 3)
    assert fig.get_size_inches().tolist() == ([14, 11] if with_ema else [14, 8])


@pytest.mark.parametrize("with_ema", [True, False])
def test_existing_panels_keep_their_size_and_place_when_maze_panels_are_added(figures, with_ema):
    plot_single_session(base_session(with_ema))
    plot_single_session(session_with_pokes(with_ema))
    plain, with_maps = figures
    n_old = len(plain.axes)
    assert with_maps.get_size_inches()[0] == plain.get_size_inches()[0]
    for old, new in zip(plain.axes, with_maps.axes[:n_old], strict=True):
        # Same distance from the figure's top edge, same size.
        top_old = plain.get_size_inches()[1] - inches(plain, old)[1]
        top_new = with_maps.get_size_inches()[1] - inches(with_maps, new)[1]
        assert inches(plain, old)[2:] == pytest.approx(inches(with_maps, new)[2:])
        assert top_old == pytest.approx(top_new)
        assert new.get_ylabel() == old.get_ylabel()


def test_two_maze_panels_and_colorbars_are_added(figures):
    plot_single_session(session_with_pokes())
    (fig,) = figures
    titled = [ax.get_title() for ax in fig.axes[4:] if ax.get_title()]
    assert [t.split("\n")[0] for t in titled] == [
        "Long pokes after initiation",
        "Pokes after choice",
    ]
    assert all("0 to this session's maximum" in t for t in titled)
    labelled = [a.get_ylabel() for a in fig.axes[4:] if "count" in a.get_ylabel()]
    assert len(labelled) == 2


def bar_and_maze_axes(fig, with_ema):
    """The (bar chart, maze map) pairs: counts by tower over the long-pokes-after-initiation map,
    counts by rank over the pokes-after-choice map."""
    n_standard = 4 if with_ema else 3
    towers_ax, ranks_ax = fig.axes[n_standard - 2 : n_standard]
    initiation_ax, choice_ax = [ax for ax in fig.axes[n_standard:] if ax.get_title()]
    assert initiation_ax.get_title().startswith("Long pokes after initiation")
    return [(towers_ax, initiation_ax), (ranks_ax, choice_ax)]


@pytest.mark.parametrize("with_ema", [True, False])
def test_each_maze_map_is_centred_under_the_bar_chart_above_it(figures, with_ema):
    plot_single_session(session_with_pokes(with_ema))
    (fig,) = figures
    fig.canvas.draw()
    for bar, maze in bar_and_maze_axes(fig, with_ema):
        bar_box, maze_box = bar.get_position(), maze.get_position()
        assert (maze_box.x0 + maze_box.x1) / 2 == pytest.approx((bar_box.x0 + bar_box.x1) / 2)
        assert maze_box.y1 < bar_box.y0  # below the bar chart, not overlapping it
        width, height = inches(fig, maze)[2:]
        assert width == pytest.approx(height)  # drawn square, filling its box


@pytest.mark.parametrize("with_ema", [True, False])
def test_colorbars_sit_beside_their_map_within_the_bar_chart_width(figures, with_ema):
    plot_single_session(session_with_pokes(with_ema))
    (fig,) = figures
    fig.canvas.draw()
    colorbars = [ax for ax in fig.axes if "count" in ax.get_ylabel()]
    assert len(colorbars) == 2
    for colorbar_ax, (bar, maze) in zip(colorbars, bar_and_maze_axes(fig, with_ema), strict=True):
        assert colorbar_ax.get_position().x0 >= maze.get_position().x1
        assert colorbar_ax.get_position().x1 <= bar.get_position().x1 + 0.05


def test_maze_counts_are_this_sessions_own(figures):
    plot_single_session(session_with_pokes())
    (fig,) = figures
    initiation_ax, choice_ax = [ax for ax in fig.axes[4:] if ax.get_title()]

    def shown(ax):
        texts = [t.get_text() for t in ax.texts]
        # texts alternate port name then its count
        return {texts[i]: int(texts[i + 1]) for i in range(0, len(texts), 2)}

    assert {p: n for p, n in shown(initiation_ax).items() if n} == {
        "A3": 1,
        "A1": 1,
        "B1": 1,
        "C1": 1,
    }
    # After choice: trial 1 -> C3, C3 (leading A1 dropped) and the B2 that initiates trial 2;
    # trial 2 -> A2 and the B2 that initiates trial 3; trial 3's window is open.
    assert {p: n for p, n in shown(choice_ax).items() if n} == {"C3": 2, "A2": 1, "B2": 2}


def test_current_choice_ports_are_outlined_green_and_initiation_port_blue(figures):
    plot_single_session(session_with_pokes())
    (fig,) = figures
    assert ring_ids(fig) == sorted(
        [f"ring:{p}:current" for p in TOWERS] * 2 + ["ring:B2:initiation"]
    )


def test_previous_problem_ports_are_outlined_red_on_both_maps(figures):
    plot_single_session(session_with_pokes(), previous_choice_towers={"A1", "C3"})
    (fig,) = figures
    assert [g for g in ring_ids(fig) if g.endswith(":previous")] == [
        "ring:A1:previous",
        "ring:A1:previous",
        "ring:C3:previous",
        "ring:C3:previous",
    ]


def test_no_previous_ports_means_no_red(figures):
    plot_single_session(session_with_pokes())
    assert not any(g.endswith(":previous") for g in ring_ids(figures[0]))


def test_unknown_previous_port_raises():
    with pytest.raises(ValueError, match="Z9"):
        plot_single_session(session_with_pokes(), previous_choice_towers={"Z9"})


def session_with_unreadable_events():
    rec = Recording()
    full_trial(rec, choice="A1")
    full_trial(rec, choice="B1")
    full_trial(rec, choice="C1")
    rec.initiate()
    rec.poke_in("A1").state("poked_in")  # poked_in without hold_poke
    return {**base_session(), **session_record([rec.to_frame()])}


def test_unreadable_raw_events_warn_and_the_figure_is_drawn_without_maps(figures):
    with pytest.warns(UserWarning, match="x.*maze maps not drawn.*poked_in"):
        plot_single_session(session_with_unreadable_events(), title="x")
    (fig,) = figures
    assert len(fig.axes) == 4  # the standard panels only
    assert fig.get_size_inches().tolist() == [14, 11]


def test_misaligned_raw_events_stop_the_run(figures):
    session = session_with_pokes()
    session["choice"] = ["A1", "C1", "B1"]
    with pytest.raises(PokeAlignmentError, match="trial 2"):
        plot_single_session(session)
    assert figures == []


def test_other_exceptions_from_the_parser_propagate(figures, monkeypatch):
    def broken(session, *, label):
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "session_poke_counts", broken)
    with pytest.raises(RuntimeError, match="boom"):
        plot_single_session(session_with_pokes())


def test_pre_parsed_counts_are_used_without_parsing_again(figures, monkeypatch):
    session = session_with_pokes()
    counts = session_poke_counts(session)

    def must_not_parse(*args, **kwargs):
        raise AssertionError("parsed again")

    monkeypatch.setattr(module, "session_poke_counts", must_not_parse)
    plot_single_session(session, poke_counts=counts)
    (fig,) = figures
    assert len(fig.axes) > 4


def test_pre_parsed_counts_with_a_parse_error_draw_no_maps_and_do_not_warn(figures):
    session = session_with_pokes()
    failed = unparsed_session_counts("unreadable")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot_single_session(session, poke_counts=failed)
    assert len(figures[0].axes) == 4


def test_initiation_outline_follows_the_sessions_own_initiation_tower(figures):
    rec = Recording(init="B3")
    full_trial(rec, choice="A1")
    full_trial(rec, choice="B1")
    full_trial(rec, choice="C1")
    session = {**base_session(), **session_record([rec.to_frame()], initiation_tower="B3")}
    plot_single_session(session)
    assert [g for g in ring_ids(figures[0]) if g.endswith(":initiation")] == ["ring:B3:initiation"]


def test_only_parser_errors_are_caught(figures):
    session = session_with_pokes()
    session["choice_towers"] = []
    with pytest.raises(ValueError, match="choice towers"):
        plot_single_session(session)
