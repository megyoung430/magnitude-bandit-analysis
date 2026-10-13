"""Unit tests for the small helpers behind ``src.behavior_import.extract_trials``."""

import pytest

from src.behavior_import.extract_trials import (
    add_reward_magnitude_features,
    assign_problem_ordinals,
    available_towers,
    fill_missing_chosen_rank_from_rank_counts,
    group_sessions_by_problem,
    infer_rank_from_counts,
    problem_signature,
    reward_magnitude_features,
    session_sort_key,
    signed_sessions,
    standardize_rank_counts,
    standardize_rank_key,
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


# --- grouping sessions into problems ---


def session(towers, *, trials=True, **extra):
    return {"trial_info": [[{"x": 1}]] if trials else [], "choice_towers": [towers], **extra}


def test_session_sort_key_orders_by_session_number_then_date():
    keys = ["ses-10_date-20260101", "ses-2_date-20260301", "other", "ses-2_date-20260201"]
    assert sorted(keys, key=session_sort_key) == [
        "ses-2_date-20260201",
        "ses-2_date-20260301",
        "ses-10_date-20260101",
        "other",
    ]


def test_problem_signature_ignores_tower_order_and_separates_has_good():
    a = problem_signature(session(["B1", "A1"], has_good=True))
    b = problem_signature(session(["A1", "B1"], has_good=True))
    c = problem_signature(session(["A1", "B1"]))
    assert a == b
    assert a != c


def test_problem_signature_is_none_without_towers():
    assert problem_signature({}) is None
    assert problem_signature({"choice_towers": [None]}) is None


@pytest.mark.parametrize(
    ("sigs", "merge", "expected"),
    [
        (["A", "B", "A"], False, [1, 2, 3]),
        (["A", "B", "A"], True, [1, 2, 1]),
        (["A", "A", "B", "B"], False, [1, 1, 2, 2]),
        (["A", None, "A"], False, [1, 1, 2]),  # None stays put, but breaks "same as previous"
        ([None, "A"], False, [1, 1]),  # ...and the first real signature is problem 1
        ([None, None], True, [1, 1]),
        (["A", "B", None, "B"], True, [1, 2, 2, 2]),
        ([], False, []),
    ],
)
def test_assign_problem_ordinals(sigs, merge, expected):
    assert assign_problem_ordinals(sigs, merge) == expected


def test_signed_sessions_skips_sessions_without_trials_and_reports_it(capsys):
    sessions = {
        "ses-2": session(["A1"]),
        "ses-1": session(["B1"]),
        "ses-3": session(["A1"], trials=False),
    }
    signed = signed_sessions("mouse", sessions)
    assert [sid for sid, _ in signed] == ["ses-1", "ses-2"]
    assert "ses-3" in capsys.readouterr().out


def by_problem(grouped):
    return {problem: list(subjects["m"]) for problem, subjects in grouped.items()}


def test_group_sessions_by_problem_merges_recurring_only_when_asked():
    data = {
        "m": {
            "ses-1": session(["A1"]),
            "ses-2": session(["B1"]),
            "ses-3": session(["A1"]),
        }
    }
    separate = group_sessions_by_problem(data)
    merged = group_sessions_by_problem(data, merge_recurring_problems=True)
    assert by_problem(separate) == {1: ["ses-1"], 2: ["ses-2"], 3: ["ses-3"]}
    assert by_problem(merged) == {1: ["ses-1", "ses-3"], 2: ["ses-2"]}


def test_group_sessions_by_problem_copies_unless_told_not_to():
    data = {"m": {"ses-1": session(["A1"])}}
    original = data["m"]["ses-1"]
    assert group_sessions_by_problem(data)[1]["m"]["ses-1"] is not original
    assert group_sessions_by_problem(data, copy_sessions=False)[1]["m"]["ses-1"] is original


def test_group_sessions_by_problem_numbers_each_subject_independently():
    data = {
        "m1": {"ses-1": session(["A1"]), "ses-2": session(["B1"])},
        "m2": {"ses-1": session(["B1"])},
    }
    out = group_sessions_by_problem(data)
    assert list(out[1]) == ["m1", "m2"]
    assert list(out[2]) == ["m1"]


# --- filling chosen_rank from rank_counts ---


def counts(best, second, third):
    return {"best": best, "second": second, "third": third}


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("second_best", "second"),
        (" Third Best ", "third"),
        ("BEST", "best"),
        ("Fourth", "fourth"),  # unknown labels are only stripped and lower-cased
        (None, None),
        (3, 3),
    ],
)
def test_standardize_rank_key(key, expected):
    assert standardize_rank_key(key) == expected


def test_standardize_rank_counts_drops_unknown_keys_and_non_dicts():
    assert standardize_rank_counts({"second best": 2, "best": 1, "fourth": 9}) == {
        "second": 2,
        "best": 1,
    }
    assert standardize_rank_counts(None) == {}


@pytest.mark.parametrize(
    ("current", "previous", "first", "expected"),
    [
        (counts(2, 1, 0), counts(1, 1, 0), False, "best"),
        (counts(1, 1, 0), counts(1, 1, 0), False, None),  # nothing incremented
        (counts(2, 2, 0), counts(1, 1, 0), False, None),  # ambiguous: two incremented
        (counts(0, 0, 3), {}, True, "third"),  # first trial: biggest count wins
        (counts(2, 2, 1), {}, True, "best"),  # ...earliest rank on ties
        (counts(0, 0, 0), {}, True, None),  # ...but only if positive
        ({"best": None, "second": 1}, {}, False, "second"),  # None counts as zero
    ],
)
def test_infer_rank_from_counts(current, previous, first, expected):
    assert infer_rank_from_counts(current, previous, first) == expected


def test_fill_missing_chosen_rank_fills_only_the_none_entries():
    session = {
        "chosen_rank": ["second", None, None, None],
        "rank_counts": [counts(0, 1, 0), counts(0, 1, 1), counts(1, 1, 1), counts(1, 1, 1)],
    }
    out = fill_missing_chosen_rank_from_rank_counts(session)
    assert out is session
    assert session["chosen_rank"] == ["second", "third", "best", None]


def test_fill_missing_chosen_rank_reads_label_variants():
    session = {
        "chosen_rank": [None, None],
        "rank_counts": [{"second_best": 1}, {"second_best": 1, "third best": 1}],
    }
    fill_missing_chosen_rank_from_rank_counts(session)
    assert session["chosen_rank"] == ["second", "third"]


@pytest.mark.parametrize(
    "session",
    [
        {"rank_counts": [counts(1, 0, 0)]},
        {"chosen_rank": [None]},
        {"chosen_rank": [], "rank_counts": []},
        {"chosen_rank": [None, None], "rank_counts": [counts(1, 0, 0)]},
    ],
)
def test_fill_missing_chosen_rank_skips_unusable_sessions(session):
    before = {k: list(v) for k, v in session.items()}
    fill_missing_chosen_rank_from_rank_counts(session)
    assert session == before
