"""One session ordering rule: date, then session number; a disagreement raises everywhere."""

import pytest

from src.agents.cohort_fitting import prepare_cohort_sequences
from src.behavior_analysis.get_good_reversal_info import get_good_reversal_info
from src.behavior_analysis.get_total_reversals import (
    get_all_reversal_indices,
    get_reversal_indices_by_kind,
    get_unclassified_reversal_indices,
)
from src.behavior_analysis.get_variables_across_sessions import get_vars_across_all_sessions
from src.behavior_analysis.matched_trial_budget import match_trial_budget
from src.behavior_analysis.problem_utils import flatten_trials
from src.behavior_analysis.reversals_by_session import reversal_counts_by_session
from src.behavior_analysis.trial_block_summary import build_trial_block_tables
from src.behavior_import.extract_trials import group_sessions_by_problem, signed_sessions
from src.behavior_import.import_data import _sorted_session_keys
from src.behavior_import.reconcile_reversals import reconcile_reversals
from src.behavior_import.session_order import (
    check_session_order,
    compress_numbers,
    ordered_session_keys,
    session_sort_key,
)
from tests.test_extract_trials import session_with_towers

# --- the rule ---


def test_sessions_are_ordered_by_date_then_number():
    keys = ["ses-3_date-20260102", "ses-2_date-20260102", "ses-1_date-20260101"]
    assert ordered_session_keys("m", keys) == [
        "ses-1_date-20260101",
        "ses-2_date-20260102",
        "ses-3_date-20260102",
    ]


def test_a_key_without_a_date_sorts_after_the_dated_keys():
    # Documented fallback for bare keys such as "ses-5": no date sorts after every dated key;
    # ties go by number, then by the key itself (so _part-NN keeps recording order).
    keys = [
        "ses-6",
        "ses-5",
        "ses-4_date-20260102_part-02",
        "ses-4_date-20260102_part-01",
        "ses-3_date-20260101",
    ]
    expected = [
        "ses-3_date-20260101",
        "ses-4_date-20260102_part-01",
        "ses-4_date-20260102_part-02",
        "ses-5",
        "ses-6",
    ]
    assert ordered_session_keys("m", keys) == expected
    assert ordered_session_keys("m", reversed(keys)) == expected


@pytest.mark.parametrize("key", ["other", "date-20260101", "ses_date-20260101", "ses-x"])
def test_a_key_without_a_session_number_raises(key):
    with pytest.raises(ValueError, match=r"m: session key .* has no 'ses-<n>'"):
        ordered_session_keys("m", ["ses-1_date-20260101", key])


def test_agreeing_orders_pass():
    check_session_order("m", ["ses-1_date-20260101", "ses-2_date-20260101", "ses-3_date-20260105"])


def test_an_undated_key_is_not_compared_by_date():
    check_session_order("m", ["ses-9", "ses-1_date-20260102", "ses-2_date-20260103"])


def test_the_same_number_on_two_dates_raises():
    keys = ["ses-3_date-20260101", "ses-3_date-20260102"]
    with pytest.raises(ValueError, match=r"m: session number 3 is on more than one date"):
        ordered_session_keys("m", keys)


def test_the_same_number_and_date_is_allowed_for_split_segments():
    keys = ["ses-3_date-20260101_part-02", "ses-3_date-20260101_part-01"]
    assert ordered_session_keys("m", keys) == sorted(keys)


def test_a_later_date_with_a_smaller_number_raises_naming_subject_and_sessions():
    keys = ["ses-2_date-20260101", "ses-1_date-20260102"]
    with pytest.raises(ValueError, match=r"m: .*'ses-1_date-20260102'.*'ses-2_date-20260101'"):
        check_session_order("m", keys)


def test_a_conflict_between_non_neighbouring_keys_raises():
    keys = ["ses-5_date-20260101", "ses-6_date-20260102", "ses-4_date-20260103"]
    with pytest.raises(ValueError, match="ses-4_date-20260103"):
        ordered_session_keys("m", keys)


def test_session_sort_key_is_the_one_used_by_the_problem_numbering():
    from src.behavior_import import problem_numbering

    assert problem_numbering.session_sort_key is session_sort_key


# --- every entry point that orders a subject's sessions ---

SWAPPED = ("ses-2_date-20260101", "ses-1_date-20260102")  # number order says the opposite
BLOCKS = {"trial": [1, 2], "blocks": [1, 2], "choices_by_tower": {}, "choices_by_rank": {}}


def swapped_sessions(make):
    return {key: make() for key in SWAPPED}


def blocks_data():
    return {"m": swapped_sessions(lambda: {**BLOCKS, "reward_magnitudes_by_tower": {}})}


def numbering_data():
    return {"m": swapped_sessions(lambda: session_with_towers(["A1"]))}


ENTRY_POINTS = {
    "merge": lambda: get_vars_across_all_sessions(blocks_data()),
    "good reversal windows": lambda: get_good_reversal_info(blocks_data()),
    "unclassified indices": lambda: get_unclassified_reversal_indices(blocks_data()),
    "indices by kind": lambda: get_reversal_indices_by_kind(blocks_data()),
    "good and bad indices": lambda: get_all_reversal_indices(blocks_data()),
    "reconcile reversals": lambda: reconcile_reversals(blocks_data()),
    "matched trial budget": lambda: match_trial_budget(blocks_data(), {"m": 1}, trial_budget=2),
    "trial block tables": lambda: build_trial_block_tables({1: blocks_data()}, {"m": 1}),
    "counts by session": lambda: reversal_counts_by_session(blocks_data()),
    "problem grouping": lambda: group_sessions_by_problem(numbering_data()),
    "signed sessions": lambda: signed_sessions("m", numbering_data()["m"]),
    "import_data keys": lambda: _sorted_session_keys({("m", k): [] for k in SWAPPED}, "m"),
    "flatten trials": lambda: flatten_trials({k: [{}] for k in SWAPPED}, subject="m"),
    "cohort sequences": lambda: prepare_cohort_sequences({5: {"m": swapped_sessions(dict)}}, [5]),
}


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_a_disagreement_raises_at_every_entry_point(entry_point):
    with pytest.raises(ValueError, match=r"m: session number order and date order disagree"):
        ENTRY_POINTS[entry_point]()


def agreeing_blocks_data():
    return {
        "m": {
            "ses-1_date-20260101": {**BLOCKS, "reward_magnitudes_by_tower": {}},
            "ses-2_date-20260102": {**BLOCKS, "reward_magnitudes_by_tower": {}},
        }
    }


def test_agreeing_orders_work_everywhere():
    data = agreeing_blocks_data()
    get_vars_across_all_sessions(data)
    get_good_reversal_info(data)
    assert get_unclassified_reversal_indices(data) == {"m": [1, 3]}
    assert get_reversal_indices_by_kind(data)["unclassified"] == {"m": [1, 3]}
    get_all_reversal_indices(data)
    match_trial_budget(data, {"m": 1}, trial_budget=3)
    build_trial_block_tables({1: data}, {"m": 1})
    group = {"m": {k: session_with_towers(["A1"]) for k in data["m"]}}
    assert [k for k, _ in signed_sessions("m", group["m"])] == list(data["m"])
    group_sessions_by_problem(group)
    assert flatten_trials({k: [{}] for k in data["m"]}, subject="m") == [{}, {}]


@pytest.mark.parametrize(
    ("numbers", "text"),
    [
        ([], ""),
        ([5], "5"),
        ([1, 2], "1-2"),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "1-10"),
        ([3, 4, 5, 6, 7, 8, 20], "3-8, 20"),
        ([20, 3, 4, 7, 5, 6, 8, 3], "3-8, 20"),
        ([1, 3, 5], "1, 3, 5"),
        ([1, 2, 4, 5, 9], "1-2, 4-5, 9"),
    ],
)
def test_compress_numbers(numbers, text):
    assert compress_numbers(numbers) == text
