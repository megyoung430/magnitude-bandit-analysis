"""Unit tests for the small helpers behind ``src.behavior_import.extract_trials``."""

import pytest

from src.behavior_import.extract_trials import (
    add_reward_magnitude_features,
    available_towers,
    reward_magnitude_features,
    top_magnitudes,
)

# --- reward magnitude features ---


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        (["B1", "A1"], ["B1", "A1"]),
        (("A1", "B1"), ["A1", "B1"]),
        ({"A1": 1, "B1": 2}, ["A1", "B1"]),
        ("A1", ["A1"]),
        (None, ["C1"]),  # not logged: fall back to the magnitude keys
        (5, ["C1"]),  # unrecognised type: same fallback
        ([], []),  # an explicit empty list is respected, not replaced
    ],
)
def test_available_towers(entry, expected):
    assert available_towers(entry, {"C1": 3}) == expected


def test_top_magnitudes_sorts_descending_and_skips_missing():
    magnitudes = {"A": 1, "B": 5, "C": None, "D": 3, "E": 4}
    assert top_magnitudes(["A", "B", "C", "D", "E", "Z"], magnitudes, 3) == [5, 4, 3]


def test_top_magnitudes_returns_fewer_when_few_towers():
    assert top_magnitudes(["A"], {"A": 2}, 3) == [2]
    assert top_magnitudes([], {}, 3) == []


def test_top_magnitudes_ties_do_not_change_values():
    # Tied towers are ordered by name; the magnitudes themselves are the same either way.
    assert top_magnitudes(["B", "A", "C"], {"A": 2, "B": 2, "C": 2}, 3) == [2, 2, 2]


def test_reward_magnitude_features_basic():
    mags = [{"A1": 4, "B1": 2, "C1": 1}, {"A1": 1, "B1": 3, "C1": 2}]
    chosen, by_arm, by_rank = reward_magnitude_features(mags, ["B1", "A1"], None)
    assert chosen == [2, 1]
    assert by_arm == {"A1": [4, 1], "B1": [2, 3], "C1": [1, 2]}
    assert by_rank == {"best": [4, 3], "second": [2, 2], "third": [1, 1]}


def test_reward_magnitude_features_missing_trials_stay_aligned():
    mags = [None, {"A1": 4}, {"A1": 2, "B1": 1}]
    chosen, by_arm, by_rank = reward_magnitude_features(mags, ["A1", "A1", "B1"], None)
    assert chosen == [None, 4, 1]
    assert by_arm == {"A1": [None, 4, 2], "B1": [None, None, 1]}
    assert by_rank["best"] == [None, 4, 2]
    assert by_rank["second"] == [None, None, 1]
    assert by_rank["third"] == [None, None, None]


def test_reward_magnitude_features_mismatched_choice_length_is_all_missing():
    chosen, _, _ = reward_magnitude_features([{"A1": 1}, {"A1": 1}], ["A1"], None)
    assert chosen == [None, None]


def test_reward_magnitude_features_logged_towers_restrict_ranking():
    mags = [{"A1": 9, "B1": 2, "C1": 1}]
    _, by_arm, by_rank = reward_magnitude_features(mags, ["B1"], [["B1", "C1"]])
    assert list(by_arm) == ["B1", "C1"]
    assert by_rank["best"] == [2]
    assert by_rank["second"] == [1]


def test_reward_magnitude_features_short_tower_list_falls_back_per_trial():
    mags = [{"A1": 1, "B1": 2}, {"C1": 5}]
    _, by_arm, _ = reward_magnitude_features(mags, [None, None], [["A1"]])
    assert by_arm == {"A1": [1, None], "C1": [None, 5]}


def test_add_reward_magnitude_features_sets_all_keys_and_legacy_alias():
    session = {"reward_magnitudes": [{"A1": 2, "B1": 1}], "choice": ["A1"]}
    out = add_reward_magnitude_features(session)
    assert out is session
    assert session["chosen_magnitude"] == [2]
    assert session["reward_magnitudes_by_tower"] is session["reward_magnitude_by_arm"]


@pytest.mark.parametrize(
    "session", [{}, {"reward_magnitudes": []}, {"reward_magnitudes": {"A1": 1}}]
)
def test_add_reward_magnitude_features_leaves_session_without_magnitudes_alone(session):
    before = dict(session)
    assert add_reward_magnitude_features(session) == before
