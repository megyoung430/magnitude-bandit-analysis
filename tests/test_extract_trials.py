"""Unit tests for the small helpers behind ``src.behavior_import.extract_trials``."""

import json
import warnings
from copy import deepcopy

import pandas as pd
import pytest

from src.behavior_import.extract_trials import (
    add_reward_magnitude_features,
    available_towers,
    bandit_trial_info,
    drop_unlogged_reversal_vars,
    extract_session_trials,
    extract_trials,
    extract_trials_grouped_by_problem,
    fill_missing_chosen_rank_from_rank_counts,
    group_sessions_by_problem,
    infer_rank_from_counts,
    merge_segments,
    merge_trial_variables,
    problem_signature,
    recording_trial_info,
    reward_magnitude_features,
    session_sort_key,
    session_trial_variables,
    signed_sessions,
    standardize_rank_counts,
    standardize_rank_key,
    top_magnitudes,
)

# Facts that ``import_data`` stores on every session; a clean first session of a layout.
EVIDENCE = {
    "starts_found_ports_false": True,
    "has_port_finding_trials": True,
    "task_name": "Meg\\3x3_field_drifting_value_bandit",
}

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


def test_reward_magnitude_features_pins_existing_behaviour_suspect_mismatched_choice_length():
    # Pins current (suspect) behaviour: a length mismatch silently blanks every chosen magnitude.
    chosen, _, _ = reward_magnitude_features([{"A1": 1}, {"A1": 1}], ["A1"], None)
    assert chosen == [None, None]


def test_reward_magnitude_features_logged_towers_restrict_ranking():
    mags = [{"A1": 9, "B1": 2, "C1": 1}]
    _, by_arm, by_rank = reward_magnitude_features(mags, ["B1"], [["B1", "C1"]])
    assert list(by_arm) == ["B1", "C1"]
    assert by_rank["best"] == [2]
    assert by_rank["second"] == [1]


def test_reward_magnitude_features_pins_existing_behaviour_suspect_short_tower_list_padding():
    # Pins current (suspect) behaviour: a too-short tower list is silently padded with None.
    mags = [{"A1": 1, "B1": 2}, {"C1": 5}]
    _, by_arm, _ = reward_magnitude_features(mags, [None, None], [["A1"]])
    assert by_arm == {"A1": [1, None], "C1": [None, 5]}


def test_add_reward_magnitude_features_sets_all_keys_and_legacy_alias():
    session = {"reward_magnitudes": [{"A1": 2, "B1": 1}], "choice": ["A1"]}
    out = add_reward_magnitude_features(session)
    assert out is session
    assert session["chosen_magnitude"] == [2]
    assert session["reward_magnitude_by_arm"] == {"A1": [2], "B1": [1]}
    assert session["reward_magnitude_by_rank"] == {"best": [2], "second": [1], "third": [None]}
    assert session["reward_magnitudes_by_tower"] is session["reward_magnitude_by_arm"]


@pytest.mark.parametrize(
    "session", [{}, {"reward_magnitudes": []}, {"reward_magnitudes": {"A1": 1}}]
)
def test_add_reward_magnitude_features_leaves_session_without_magnitudes_alone(session):
    before = dict(session)
    assert add_reward_magnitude_features(session) == before


# --- grouping sessions into problems ---


def session_with_towers(towers, *, trials=True, **extra):
    return {
        "trial_info": [[{"x": 1}]] if trials else [],
        "choice_towers": [towers],
        "settings_evidence": EVIDENCE,
        **extra,
    }


def test_session_sort_key_orders_by_date_then_session_number():
    keys = ["ses-10_date-20260101", "ses-2_date-20260301", "other", "ses-2_date-20260201"]
    assert sorted(keys, key=session_sort_key) == [
        "ses-10_date-20260101",
        "ses-2_date-20260201",
        "ses-2_date-20260301",
        "other",
    ]


def test_problem_signature_ignores_tower_order_but_not_the_initiation_tower():
    a = problem_signature(session_with_towers(["B1", "A1"], initiation_tower="B2"))
    b = problem_signature(session_with_towers(["A1", "B1"], initiation_tower="B2"))
    c = problem_signature(session_with_towers(["A1", "B1"], initiation_tower="A2"))
    assert a == b
    assert a != c


def test_problem_signature_does_not_separate_has_good():
    good = problem_signature(session_with_towers(["A1", "B1"], has_good=True))
    assert good == problem_signature(session_with_towers(["A1", "B1"]))


def test_problem_signature_is_none_without_towers():
    assert problem_signature({}) is None
    assert problem_signature({"choice_towers": [None]}) is None
    assert problem_signature({"choice_towers": [[]]}) is None


def test_signed_sessions_skips_sessions_without_trials_and_reports_it(capsys):
    sessions = {
        "ses-2": session_with_towers(["A1"]),
        "ses-1": session_with_towers(["B1"]),
        "ses-3": session_with_towers(["A1"], trials=False),
    }
    signed = signed_sessions("mouse", sessions)
    assert [sid for sid, _ in signed] == ["ses-1", "ses-2"]
    assert "ses-3" in capsys.readouterr().out


def by_problem(grouped):
    return {problem: list(subjects["m"]) for problem, subjects in grouped.items()}


def test_group_sessions_by_problem_numbers_a_return_to_an_earlier_layout_as_new():
    data = {
        "m": {
            "ses-1": session_with_towers(["A1"]),
            "ses-2": session_with_towers(["B1"]),
            "ses-3": session_with_towers(["B1"]),
            "ses-4": session_with_towers(["A1"]),
        }
    }
    assert by_problem(group_sessions_by_problem(data)) == {
        1: ["ses-1"],
        2: ["ses-2", "ses-3"],
        3: ["ses-4"],
    }


def test_group_sessions_by_problem_leaves_out_a_flagged_session_and_pools_around_it():
    data = {
        "m": {
            "ses-1": session_with_towers(["A1"]),
            "ses-2": session_with_towers(["B1"]),  # lasts one session: suspected settings error
            "ses-3": session_with_towers(["A1"]),
        }
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # import_data warns; grouping does not repeat it
        grouped = group_sessions_by_problem(data)
    assert by_problem(grouped) == {1: ["ses-1", "ses-3"]}
    assert "ses-2" in data["m"]  # still available for single-session analyses


def test_group_sessions_by_problem_uses_the_initiation_tower_and_ignores_has_good():
    data = {
        "m": {
            "ses-1": session_with_towers(["A1"], initiation_tower="B2", has_good=True),
            "ses-2": session_with_towers(["A1"], initiation_tower="B2"),
            "ses-3": session_with_towers(["A1"], initiation_tower="A2"),
            "ses-4": session_with_towers(["A1"], initiation_tower="A2"),
        }
    }
    assert by_problem(group_sessions_by_problem(data)) == {
        1: ["ses-1", "ses-2"],
        2: ["ses-3", "ses-4"],
    }


def test_group_sessions_by_problem_requires_settings_evidence():
    data = {"m": {"ses-1": {"trial_info": [[{"x": 1}]], "choice_towers": [["A1"]]}}}
    with pytest.raises(ValueError, match="m ses-1.*settings_evidence"):
        group_sessions_by_problem(data)


def test_group_sessions_by_problem_copies_unless_told_not_to():
    data = {"m": {"ses-1": session_with_towers(["A1"])}}
    original = data["m"]["ses-1"]
    assert group_sessions_by_problem(data)[1]["m"]["ses-1"] is not original
    assert group_sessions_by_problem(data, copy_sessions=False)[1]["m"]["ses-1"] is original


def test_group_sessions_by_problem_numbers_each_subject_independently():
    data = {
        "m1": {"ses-1": session_with_towers(["A1"]), "ses-2": session_with_towers(["B1"])},
        "m2": {"ses-1": session_with_towers(["B1"])},
    }
    out = group_sessions_by_problem(data)
    assert list(out[1]) == ["m1", "m2"]
    assert list(out[2]) == ["m1"]


# --- filling chosen_rank from rank_counts ---


def rank_counts(best, second, third):
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
        (rank_counts(2, 1, 0), rank_counts(1, 1, 0), False, "best"),
        (rank_counts(1, 1, 0), rank_counts(1, 1, 0), False, None),  # nothing incremented
        (rank_counts(2, 2, 0), rank_counts(1, 1, 0), False, None),  # ambiguous: two incremented
        (rank_counts(0, 0, 3), {}, True, "third"),  # first trial: biggest count wins
        (rank_counts(2, 2, 1), {}, True, "best"),  # ...earliest rank on ties
        (rank_counts(0, 0, 0), {}, True, None),  # ...but only if positive
        ({"best": None, "second": 1}, {}, False, "second"),  # None counts as zero
    ],
)
def test_infer_rank_from_counts(current, previous, first, expected):
    assert infer_rank_from_counts(current, previous, first) == expected


def test_fill_missing_chosen_rank_fills_only_the_none_entries():
    session = {
        "chosen_rank": ["second", None, None, None],
        "rank_counts": [
            rank_counts(0, 1, 0),
            rank_counts(0, 1, 1),
            rank_counts(1, 1, 1),
            rank_counts(1, 1, 1),
        ],
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
        {"rank_counts": [rank_counts(1, 0, 0)]},
        {"chosen_rank": [None]},
        {"chosen_rank": [], "rank_counts": []},
        {"chosen_rank": [None, None], "rank_counts": [rank_counts(1, 0, 0)]},
    ],
)
def test_fill_missing_chosen_rank_skips_unusable_sessions(session):
    before = {k: list(v) for k, v in session.items()}
    fill_missing_chosen_rank_from_rank_counts(session)
    assert session == before


# --- extracting trials from recordings ---


def recording(trials, *, other_rows=()):
    """A recording frame with one variable-print row per trial dict."""
    rows = [{"time": -1, "type": t, "subtype": st, "content": c} for t, st, c in other_rows]
    rows += [
        {"time": t.pop("time", i), "type": "variable", "subtype": "print", "content": json.dumps(t)}
        for i, t in enumerate(dict(trial) for trial in trials)
    ]
    return pd.DataFrame(rows)


def bandit_trial(n, choice, rank, counts, **extra):
    return {
        "num_trials": n,
        "chosen_tower": choice,
        "current_reward_magnitudes": {"A1": 4, "B1": 2, "C1": 1},
        "current_choice_towers": ["A1", "B1", "C1"],
        "chosen_rank": rank,
        "rank_counts": counts,
        **extra,
    }


def test_recording_trial_info_keeps_only_sorted_print_rows_without_port_finding():
    df = recording(
        [{"num_trials": 2, "time": 5}, {"num_t_found": 1}, {"num_trials": 1, "time": 2}],
        other_rows=[("event", "poke", "poke_A"), ("variable", "run_start", "{}")],
    )
    assert recording_trial_info(df) == [{"num_trials": 1}, {"num_trials": 2}]


def test_recording_trial_info_without_subtype_column_raises():
    df = recording([{"num_trials": 1}]).drop(columns="subtype")
    with pytest.raises(AttributeError):
        recording_trial_info(df)


def test_recording_trial_info_pins_existing_behaviour_suspect_missing_type_column():
    # Pins current (suspect) behaviour: no `type` column gives no trials instead of an error.
    df = recording([{"num_trials": 1}]).drop(columns="type")
    assert recording_trial_info(df) == []


def test_bandit_trial_info_skips_pretraining_recordings(capsys):
    pretraining = recording([{"num_trials": 1, "chosen_tower": "B1"}])
    assert bandit_trial_info(pretraining, "mouse", "ses-1") == []
    assert "ses-1" in capsys.readouterr().out
    bandit = recording([bandit_trial(1, "A1", "best", {"best": 1})])
    assert len(bandit_trial_info(bandit, "mouse", "ses-1")) == 1


def test_bandit_trial_info_of_an_empty_recording_is_empty():
    empty = pd.DataFrame({"type": [], "subtype": [], "time": [], "content": []})
    assert bandit_trial_info(empty, "m", "s") == []


@pytest.mark.parametrize(
    ("var_name", "segments", "expected"),
    [
        ("trial", [[1, 2, 3], [1, 2]], [1, 2, 3, 4, 5]),
        ("blocks", [[1, 2], [1, 2]], [1, 2, 2, 3]),
        ("rank_counts", [[{"best": 1}], [{"best": 1}]], [{"best": 1}, {"best": 2}]),
        ("trials_in_block", [[1, 2], [3, 1]], [1, 2, 5, 1]),
        ("choice", [["A1"], [], ["B1", "C1"]], ["A1", "B1", "C1"]),
    ],
)
def test_merge_segments_uses_the_rule_for_each_variable(var_name, segments, expected):
    assert merge_segments(var_name, segments) == expected


def test_merge_trial_variables_treats_a_missing_variable_as_empty():
    merged = merge_trial_variables([{"choice": ["A1"], "trial": [1]}, {"choice": ["B1"]}])
    assert merged == {"choice": ["A1", "B1"], "trial": [1]}


def test_session_trial_variables_merges_only_multi_file_sessions(capsys):
    only = {"choice": ["A1"]}
    assert session_trial_variables([only], "m", "s") is only
    assert capsys.readouterr().out == ""
    assert session_trial_variables([only, {"choice": ["B1"]}], "m", "s") == {"choice": ["A1", "B1"]}
    assert "Merging multiple files" in capsys.readouterr().out


def test_drop_unlogged_reversal_vars_only_when_not_logged():
    session = {"has_good": False, "has_bad": True, "good_reversals": [None], "bad_reversals": [0]}
    drop_unlogged_reversal_vars(session)
    assert session == {"has_good": False, "has_bad": True, "bad_reversals": [0]}


def test_extract_session_trials_without_bandit_trials_gets_empty_fields(capsys):
    session = {"data": recording([{"num_trials": 1}])}
    extract_session_trials(session, "m", "ses-1")
    assert session["trial_info"] == []
    assert session["trial_variables"] == {}
    assert "No trial information" in capsys.readouterr().out


def test_extract_session_trials_merges_restarted_recordings_and_derives_fields():
    first = recording(
        [
            bandit_trial(1, "A1", "best", {"best": 1, "second": 0}, num_blocks=1),
            bandit_trial(2, "B1", None, {"best": 1, "second": 1}, num_blocks=1),
        ]
    )
    restarted = recording([bandit_trial(1, "A1", "best", {"best": 1, "second": 0}, num_blocks=1)])
    session = {"data": [first, restarted]}
    extract_session_trials(session, "m", "ses-1")
    assert session["trial"] == [1, 2, 3]
    assert session["blocks"] == [1, 1, 1]
    # The second trial's rank is only known from rank_counts, so the one-hot list shows
    # whether the back-fill ran before the rank indicators were built.
    assert session["chosen_rank"] == ["best", "second", "best"]
    assert session["choices_by_rank"]["best"] == [1, 0, 1]
    assert session["choices_by_rank"]["second"] == [0, 1, 0]
    assert session["choices_by_tower"]["A1"] == [1, 0, 1]
    assert session["chosen_magnitude"] == [4, 2, 4]
    assert session["has_good"] is False
    assert "good_reversals" not in session


def test_extract_trials_updates_every_session_in_place():
    bandit = recording([bandit_trial(1, "A1", "best", {"best": 1}, num_blocks=1)])
    data = {"m": {"ses-1": {"data": bandit}, "ses-2": {"data": bandit.copy()}}}
    out = extract_trials(data)
    assert out is data
    assert all(session["choice"] == ["A1"] for session in data["m"].values())


def test_grouped_by_problem_numbers_a_recurring_problem_as_new():
    problem_a = {"current_choice_towers": ["A1", "B1", "C1"]}
    problem_b = {
        "current_choice_towers": ["A1", "B1", "D1"],
        "current_reward_magnitudes": {"A1": 1, "B1": 3, "D1": 2},
    }
    # Session 4 returns to problem A but logged no reward magnitudes.
    no_magnitudes = {**problem_a, "current_reward_magnitudes": None}

    def one_trial_session(extra):
        return {
            "data": recording(
                [bandit_trial(1, "B1", "second", {"second": 1}, num_blocks=1, **extra)]
            ),
            "settings_evidence": EVIDENCE,
        }

    data = {
        "m": {
            "ses-1": one_trial_session(problem_a),
            "ses-2": one_trial_session(problem_b),
            "ses-3": one_trial_session(problem_b),
            "ses-4": one_trial_session(no_magnitudes),
        }
    }
    grouped = extract_trials_grouped_by_problem(deepcopy(data))

    assert by_problem(grouped) == {1: ["ses-1"], 2: ["ses-2", "ses-3"], 3: ["ses-4"]}
    assert grouped[1]["m"]["ses-1"]["chosen_magnitude"] == [2]
    assert grouped[2]["m"]["ses-2"]["chosen_magnitude"] == [3]
    assert grouped[3]["m"]["ses-4"]["chosen_magnitude"] == [None]
    assert grouped[3]["m"]["ses-4"]["choices_by_rank"]["second"] == [1]
