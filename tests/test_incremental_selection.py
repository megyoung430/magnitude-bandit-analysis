"""Picking the recent sessions of a cohort, and merging a partial rerun into stacked tables."""

import pandas as pd
import pytest

from src.pipelines.incremental import (
    NoSessionsSelectedError,
    describe_plan,
    merge_diagnostic_tables,
    parse_since,
    replace_problem_rows,
    select_recent_sessions,
)

B1, D1, D2 = ("blocked", 1), ("drifting", 1), ("drifting", 2)


def grouped_cohort():
    """Sessions are only keys here; selection never looks inside them.

    M1: blocked 1 has ses-1..ses-3, drifting 1 has ses-4, ses-5, drifting 2 has ses-6.
    M2: blocked 1 has ses-1, ses-2 (two sessions on 20261012), drifting 1 has ses-3.
    M3 (new mouse): drifting 2 has ses-1 only.
    """
    return {
        B1: {
            "M1": {f"ses-{n}_date-2026100{n}": {} for n in (1, 2, 3)},
            "M2": {"ses-1_date-20261012": {}, "ses-2_date-20261012": {}},
        },
        D1: {
            "M1": {"ses-4_date-20261004": {}, "ses-5_date-20261005": {}},
            "M2": {"ses-3_date-20261013": {}},
        },
        D2: {
            "M1": {"ses-6_date-20261016": {}},
            "M3": {"ses-1_date-20261016": {}},
        },
    }


# --- last N sessions per mouse ---------------------------------------------------------


def test_last_sessions_takes_each_mouses_most_recent_sessions():
    selection = select_recent_sessions(grouped_cohort(), last_sessions=1)
    assert selection.sessions == [
        ("M1", "ses-6_date-20261016"),
        ("M2", "ses-3_date-20261013"),
        ("M3", "ses-1_date-20261016"),
    ]
    assert selection.problems == {D1, D2}


def test_last_sessions_can_span_two_problems_of_one_mouse():
    selection = select_recent_sessions(grouped_cohort(), last_sessions=3)
    assert ("M1", "ses-4_date-20261004") in selection.sessions
    assert {B1, D1, D2} == selection.problems  # M2's last 3 reach back into blocked
    assert selection.sessions_by_problem[B1] == [
        ("M2", "ses-1_date-20261012"),
        ("M2", "ses-2_date-20261012"),
    ]


def test_a_mouse_with_fewer_sessions_than_n_contributes_all_of_them():
    selection = select_recent_sessions(grouped_cohort(), last_sessions=50)
    m3 = [pair for pair in selection.sessions if pair[0] == "M3"]
    assert m3 == [("M3", "ses-1_date-20261016")]
    assert len(selection.sessions) == 6 + 3 + 1


def test_sessions_tied_on_date_are_ordered_by_session_number():
    selection = select_recent_sessions(grouped_cohort(), last_sessions=4)
    m2 = [key for mouse, key in selection.sessions if mouse == "M2"]
    assert m2 == ["ses-1_date-20261012", "ses-2_date-20261012", "ses-3_date-20261013"]
    one = select_recent_sessions({B1: {"M2": grouped_cohort()[B1]["M2"]}}, last_sessions=1)
    assert one.sessions == [("M2", "ses-2_date-20261012")]


def test_split_parts_of_one_recording_count_as_one_session():
    grouped = {
        B1: {"M1": {"ses-1_date-20261001": {}, "ses-2_date-20261002_part-01": {}}},
        D1: {"M1": {"ses-2_date-20261002_part-02": {}, "ses-3_date-20261003": {}}},
    }
    selection = select_recent_sessions(grouped, last_sessions=2)
    assert selection.sessions == [("M1", "ses-2_date-20261002"), ("M1", "ses-3_date-20261003")]
    assert selection.problems == {B1, D1}


def test_pooled_drifting_analyses_run_only_when_a_drifting_problem_is_selected():
    drifting = select_recent_sessions(grouped_cohort(), last_sessions=1)
    assert drifting.runs_pooled_drifting
    blocked_only = select_recent_sessions({B1: grouped_cohort()[B1]}, last_sessions=1)
    assert blocked_only.problems == {B1}
    assert not blocked_only.runs_pooled_drifting


# --- since date ---------------------------------------------------------------------------


def test_since_includes_the_date_itself_and_later_sessions():
    selection = select_recent_sessions(grouped_cohort(), since="2026-10-13")
    assert selection.sessions == [
        ("M1", "ses-6_date-20261016"),
        ("M2", "ses-3_date-20261013"),
        ("M3", "ses-1_date-20261016"),
    ]
    assert selection.problems == {D1, D2}


def test_since_selects_both_problems_when_sessions_straddle_them():
    selection = select_recent_sessions(grouped_cohort(), since="2026-10-05")
    assert {B1, D1, D2} == selection.problems
    assert ("M1", "ses-5_date-20261005") in selection.sessions
    assert ("M1", "ses-4_date-20261004") not in selection.sessions


def test_parse_since_returns_the_yyyymmdd_int_of_the_session_keys():
    assert parse_since("2026-10-16") == 20261016


@pytest.mark.parametrize("text", ["20261016", "2026-13-01", "16/10/2026", "yesterday", ""])
def test_an_unparseable_date_raises(text):
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        parse_since(text)


# --- failures ---------------------------------------------------------------------------


def test_exactly_one_mode_is_required():
    with pytest.raises(ValueError, match="exactly one"):
        select_recent_sessions(grouped_cohort())
    with pytest.raises(ValueError, match="exactly one"):
        select_recent_sessions(grouped_cohort(), last_sessions=2, since="2026-10-01")


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_last_sessions_below_one_or_not_an_integer_raises(n):
    with pytest.raises(ValueError, match="last_sessions"):
        select_recent_sessions(grouped_cohort(), last_sessions=n)


def test_no_matching_session_raises_and_names_the_mice_and_dates_that_exist():
    with pytest.raises(NoSessionsSelectedError) as err:
        select_recent_sessions(grouped_cohort(), since="2027-01-01")
    message = str(err.value)
    assert "2027-01-01" in message
    assert "M1" in message and "M2" in message and "M3" in message
    assert "2026-10-01" in message and "2026-10-16" in message


def test_the_no_match_error_is_a_value_error():
    assert issubclass(NoSessionsSelectedError, ValueError)


def test_sessions_that_cannot_be_ordered_raise():
    grouped = {B1: {"M1": {"ses-2_date-20261001": {}, "ses-1_date-20261002": {}}}}
    with pytest.raises(ValueError, match="M1"):
        select_recent_sessions(grouped, last_sessions=1)


def test_since_needs_a_date_in_every_session_key():
    grouped = {B1: {"M1": {"ses-1": {}}}}
    with pytest.raises(ValueError, match="no date"):
        select_recent_sessions(grouped, since="2026-10-01")


# --- the plan text ---------------------------------------------------------------------


def test_the_plan_names_problems_sessions_and_pooled_scopes():
    selection = select_recent_sessions(grouped_cohort(), last_sessions=1)
    text = describe_plan("cohort-04", selection)
    assert "cohort-04" in text and "last 1 session(s) per mouse" in text
    assert "drifting problem 2" in text and "M3: ses-1_date-20261016" in text
    assert "blocked problem 1" not in text
    assert "all-drifting-problems" in text and "will be regenerated" in text


def test_the_plan_says_when_the_pooled_drifting_analyses_are_skipped():
    selection = select_recent_sessions({B1: grouped_cohort()[B1]}, last_sessions=1)
    assert "all-drifting-problems: skipped" in describe_plan("cohort-04", selection)


# --- merging stacked tables ---------------------------------------------------------------


def table(*rows):
    return pd.DataFrame(rows, columns=["task_type", "problem_in_type", "mouse", "value"])


def test_rows_of_rerun_problems_are_replaced_and_others_kept():
    existing = table(
        ("blocked", 1, "M1", 10),
        ("drifting", 1, "M1", 11),
        ("drifting", 1, "M2", 12),
        ("drifting", 2, "M1", 13),
    )
    new = table(("drifting", 1, "M1", 99), ("drifting", 1, "M3", 98))
    merged, partial = replace_problem_rows(existing, new, {D1})
    assert not partial
    assert merged.values.tolist() == [
        ["blocked", 1, "M1", 10],
        ["drifting", 1, "M1", 99],
        ["drifting", 1, "M3", 98],
        ["drifting", 2, "M1", 13],
    ]


def test_a_rerun_problem_that_now_has_no_rows_loses_its_old_rows():
    existing = table(("drifting", 1, "M1", 11), ("drifting", 2, "M1", 13))
    merged, _ = replace_problem_rows(existing, table(), {D1})
    assert merged.values.tolist() == [["drifting", 2, "M1", 13]]


def test_a_missing_existing_table_gives_only_the_new_rows_and_is_partial():
    new = table(("drifting", 1, "M1", 99))
    merged, partial = replace_problem_rows(None, new, {D1})
    assert partial
    assert merged.equals(new)


def test_an_existing_table_without_the_problem_columns_is_partial():
    existing = pd.DataFrame({"mouse": ["M1"], "value": [1]})
    new = table(("drifting", 1, "M1", 99))
    merged, partial = replace_problem_rows(existing, new, {D1})
    assert partial
    assert merged.equals(new)


def test_a_header_only_existing_table_is_not_partial():
    merged, partial = replace_problem_rows(table(), table(("drifting", 1, "M1", 99)), {D1})
    assert not partial
    assert len(merged) == 1


def test_merge_diagnostic_tables_names_the_partial_tables():
    new = {"poke_trial_summary": table(("drifting", 1, "M1", 99)), "value_change_events": table()}
    existing = {"poke_trial_summary": table(("blocked", 1, "M1", 1)), "value_change_events": None}
    merged, partial = merge_diagnostic_tables(new, existing, {D1})
    assert partial == ["value_change_events"]
    assert len(merged["poke_trial_summary"]) == 2
    assert merged["value_change_events"].empty
