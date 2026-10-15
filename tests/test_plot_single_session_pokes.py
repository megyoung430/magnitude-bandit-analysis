"""The session-level maze heatmaps added to the single-session figure."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

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
    assert titled == ["Long pokes after initiation", "Pokes after choice"]
    labelled = [a.get_ylabel() for a in fig.axes[4:] if "count" in a.get_ylabel()]
    assert len(labelled) == 2
    assert all("session" in label for label in labelled)


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
    # After choice: trial 1 -> C3, C3 (leading A1 dropped); trial 2 -> A2; trial 3 is open.
    assert {p: n for p, n in shown(choice_ax).items() if n} == {"C3": 2, "A2": 1}


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


def test_misaligned_raw_events_raise_instead_of_drawing_the_wrong_trials(figures):
    session = session_with_pokes()
    session["choice"] = ["A1", "C1", "B1"]
    with pytest.raises(ValueError, match="trial 2"):
        plot_single_session(session)
