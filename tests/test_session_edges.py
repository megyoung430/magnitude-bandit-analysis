"""The first and last trial of each session, as the two sides of a boundary between sessions.

``reconcile_reversals`` stores them on every session with trials, so that two sessions of a
problem can be compared directly, whatever sessions without trials lie between them in time.
"""

from src.behavior_import.extract_trials import extract_trials
from tests.boundary_fixtures import (
    ZERO_B1,
    ZERO_B3,
    ZERO_B3_DRIFTED,
    prints,
    recording,
    session,
    snapshot,
)


def imported(sessions):
    data = {"m": {key: session(frames) for key, frames in sessions.items()}}
    return extract_trials(data)["m"]


def test_the_edges_are_the_first_trial_of_the_first_recording_and_the_last_of_the_last():
    first = recording(
        snapshot(ZERO_B3, blocks=1, tot_blocks=4),
        prints(ZERO_B3, 3, blocks=1),
        snapshot(ZERO_B3, blocks=1, tot_blocks=4),
    )
    second = recording(
        snapshot(ZERO_B1, blocks=1, tot_blocks=5),
        prints(ZERO_B1, 2, blocks=1),
        snapshot(ZERO_B1, blocks=1, tot_blocks=5),
    )
    edges = imported({"ses-1": [first, second]})["ses-1"]["session_edges"]
    assert edges["first"]["magnitudes"] == ZERO_B3
    assert edges["last"]["magnitudes"] == ZERO_B1
    assert edges["first"]["totals"]["block"] == 4
    assert edges["last"]["totals"]["block"] == 5


def test_a_recording_without_trials_is_not_an_edge():
    empty_first = recording(snapshot(ZERO_B1, blocks=1, tot_blocks=2), [], snapshot(ZERO_B1))
    with_trials = recording(
        snapshot(ZERO_B3, blocks=1, tot_blocks=3),
        prints(ZERO_B3_DRIFTED, 2, blocks=1),
        snapshot(ZERO_B3_DRIFTED, blocks=1, tot_blocks=3),
    )
    empty_last = recording(snapshot(ZERO_B1, blocks=1, tot_blocks=9), [], snapshot(ZERO_B1))
    edges = imported({"ses-1": [empty_first, with_trials, empty_last]})["ses-1"]["session_edges"]
    assert edges["first"]["magnitudes"] == ZERO_B3_DRIFTED
    assert edges["first"]["totals"]["block"] == 3
    assert edges["last"]["totals"]["block"] == 3


def test_a_session_without_any_trial_has_no_edges():
    empty = recording(snapshot(ZERO_B1, blocks=1, tot_blocks=2), [], snapshot(ZERO_B1))
    assert imported({"ses-1": [empty]})["ses-1"]["session_edges"] is None
