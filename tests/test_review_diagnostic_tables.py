"""The diagnostic DataFrames of the entry points, concatenated over problems for one file each."""

import pandas as pd
import pytest

from src.behavior_analysis.diagnostic_tables import DIAGNOSTIC_TABLES, stack_diagnostics

PROBLEM_MAP = pd.DataFrame(
    {
        "mouse": ["M1", "M1", "M2"],
        "task_type": ["blocked", "drifting", "drifting"],
        "problem_in_type": [1, 1, 1],
        "overall_problem": [1, 2, 3],
    }
)


def frames(**by_name):
    return by_name


def test_the_four_kinds_are_the_ones_the_entry_points_return():
    assert DIAGNOSTIC_TABLES == (
        "poke_trial_summary",
        "reversal_aligned_arms_counts",
        "value_change_events",
        "value_change_bin_counts",
    )


def test_tables_of_all_problems_are_stacked_with_task_type_and_problem_in_front():
    arms_1 = pd.DataFrame({"scope": ["pooled"], "n_reversals": [3]})
    arms_2 = pd.DataFrame({"scope": ["pooled"], "n_reversals": [5]})
    stacked = stack_diagnostics(
        {
            ("drifting", 1): {"reversal_aligned_arms_counts": arms_1},
            ("drifting", 2): {"reversal_aligned_arms_counts": arms_2},
        },
        PROBLEM_MAP,
    )
    table = stacked["reversal_aligned_arms_counts"]
    assert list(table.columns) == ["task_type", "problem_in_type", "scope", "n_reversals"]
    assert table["problem_in_type"].tolist() == [1, 2]
    assert table["n_reversals"].tolist() == [3, 5]
    assert set(table["task_type"]) == {"drifting"}


def test_a_table_with_a_mouse_column_gets_that_mouses_overall_problem():
    poke = pd.DataFrame({"mouse": ["M1", "M2"], "session": ["s1", "s1"], "n_trials": [4, 6]})
    stacked = stack_diagnostics({("drifting", 1): {"poke_trial_summary": poke}}, PROBLEM_MAP)
    table = stacked["poke_trial_summary"]
    assert list(table.columns)[:4] == ["task_type", "problem_in_type", "overall_problem", "mouse"]
    assert table["overall_problem"].tolist() == [2, 3]


def test_a_kind_no_problem_returned_has_only_its_leading_columns():
    other = pd.DataFrame({"x": [1]})
    stacked = stack_diagnostics({("blocked", 1): {"poke_trial_summary": other}}, PROBLEM_MAP)
    assert set(stacked) == set(DIAGNOSTIC_TABLES)
    assert stacked["poke_trial_summary"]["x"].tolist() == [1]
    for kind in DIAGNOSTIC_TABLES[1:]:
        assert stacked[kind].empty
        assert list(stacked[kind].columns) == ["task_type", "problem_in_type"]


def test_an_empty_table_keeps_its_columns():
    empty = pd.DataFrame({"mouse": pd.Series([], dtype=object), "n": pd.Series([], dtype=int)})
    stacked = stack_diagnostics({("drifting", 1): {"poke_trial_summary": empty}}, PROBLEM_MAP)
    table = stacked["poke_trial_summary"]
    assert table.empty
    assert list(table.columns) == ["task_type", "problem_in_type", "overall_problem", "mouse", "n"]


def test_a_mouse_without_that_problem_in_the_map_fails_loudly():
    poke = pd.DataFrame({"mouse": ["M9"], "n": [1]})
    with pytest.raises(ValueError, match="M9"):
        stack_diagnostics({("drifting", 1): {"poke_trial_summary": poke}}, PROBLEM_MAP)


def test_an_unknown_kind_fails_loudly():
    with pytest.raises(ValueError, match="mystery"):
        stack_diagnostics({("drifting", 1): {"mystery": pd.DataFrame({"a": [1]})}}, PROBLEM_MAP)


def test_nothing_to_stack_gives_every_kind_without_rows():
    stacked = stack_diagnostics({}, PROBLEM_MAP)
    assert list(stacked) == list(DIAGNOSTIC_TABLES)
    assert all(table.empty for table in stacked.values())
    assert list(stacked["poke_trial_summary"].columns) == [
        "task_type",
        "problem_in_type",
        "overall_problem",
        "mouse",
    ]
