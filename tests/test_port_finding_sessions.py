"""Which problem a port-finding-only session belongs to (pure, no recordings needed)."""

import warnings

import pytest

from src.behavior_analysis.port_finding_sessions import (
    assign_port_finding_sessions,
    port_finding_sessions_for_problem,
)

A = {"A1", "B1", "C1"}
B = {"A3", "B3", "C3"}


def bandit(towers, initiation="B2"):
    return {"choice_towers": [sorted(towers)] * 3, "initiation_tower": initiation}


def only_port_finding(towers, initiation="B2"):
    return {"df": "frames", "choice_towers": frozenset(towers), "initiation_tower": initiation}


def trials(**problems):
    """``{"p1": {"m1": {session: ...}}}`` keyed ``{1: ...}``; ``p2=`` is problem 2."""
    return {int(name[1:]): sessions for name, sessions in problems.items()}


def test_a_port_finding_session_belongs_to_the_next_problem_with_the_same_towers():
    by_problem = trials(
        p1={"m1": {"ses-1_date-20260101": bandit(A), "ses-2_date-20260102": bandit(A)}},
        p2={"m1": {"ses-4_date-20260104": bandit(B)}},
    )
    found = {"m1": {"ses-3_date-20260103": only_port_finding(B)}}
    assignment = assign_port_finding_sessions(by_problem, found)
    assert assignment.unassigned == []
    assert port_finding_sessions_for_problem(assignment, by_problem[2]) == found
    assert port_finding_sessions_for_problem(assignment, by_problem[1]) == {}


def test_it_goes_to_the_next_problem_not_the_previous_one_with_the_same_towers():
    # the mouse returns to layout A after a problem on B: the port finding before the return
    # belongs to the return, not to the earlier A problem
    by_problem = trials(
        p1={"m1": {"ses-1_date-20260101": bandit(A)}},
        p2={"m1": {"ses-2_date-20260102": bandit(B)}},
        p3={"m1": {"ses-4_date-20260104": bandit(A)}},
    )
    found = {"m1": {"ses-3_date-20260103": only_port_finding(A)}}
    assignment = assign_port_finding_sessions(by_problem, found)
    assert port_finding_sessions_for_problem(assignment, by_problem[3]) == found
    assert port_finding_sessions_for_problem(assignment, by_problem[1]) == {}


def test_two_port_finding_sessions_in_a_row_both_go_to_the_problem_that_follows():
    by_problem = trials(p1={"m1": {"ses-3_date-20260103": bandit(B)}})
    found = {
        "m1": {
            "ses-1_date-20260101": only_port_finding(B),
            "ses-2_date-20260102": only_port_finding(B),
        }
    }
    assignment = assign_port_finding_sessions(by_problem, found)
    assert assignment.unassigned == []
    assert list(port_finding_sessions_for_problem(assignment, by_problem[1])["m1"]) == [
        "ses-1_date-20260101",
        "ses-2_date-20260102",
    ]


def test_different_towers_are_not_assigned_and_the_warning_names_the_session_and_towers():
    by_problem = trials(p1={"m1": {"ses-2_date-20260102": bandit(A)}})
    found = {"m1": {"ses-1_date-20260101": only_port_finding(B)}}
    with pytest.warns(UserWarning, match=r"m1 ses-1_date-20260101.*A3.*B3.*C3") as caught:
        assignment = assign_port_finding_sessions(by_problem, found)
    assert "ses-2_date-20260102" in str(caught[0].message)  # names the session it was compared to
    (unassigned,) = assignment.unassigned
    assert (unassigned.mouse, unassigned.session) == ("m1", "ses-1_date-20260101")
    assert unassigned.towers == frozenset(B)
    assert "towers" in unassigned.reason
    assert port_finding_sessions_for_problem(assignment, by_problem[1]) == {}


def test_a_different_initiation_tower_is_not_the_same_problem():
    by_problem = trials(p1={"m1": {"ses-2_date-20260102": bandit(A, initiation="B2")}})
    found = {"m1": {"ses-1_date-20260101": only_port_finding(A, initiation="A2")}}
    with pytest.warns(UserWarning, match="initiation"):
        assignment = assign_port_finding_sessions(by_problem, found)
    assert len(assignment.unassigned) == 1


def test_without_a_later_session_with_bandit_trials_it_is_not_assigned():
    by_problem = trials(p1={"m1": {"ses-1_date-20260101": bandit(A)}})
    found = {"m1": {"ses-2_date-20260102": only_port_finding(A)}}
    with pytest.warns(UserWarning, match=r"m1 ses-2_date-20260102.*no later session"):
        assignment = assign_port_finding_sessions(by_problem, found)
    assert [u.session for u in assignment.unassigned] == ["ses-2_date-20260102"]


def test_a_mouse_without_any_bandit_session_is_not_assigned():
    found = {"m9": {"ses-1_date-20260101": only_port_finding(A)}}
    with pytest.warns(UserWarning, match="m9 ses-1_date-20260101"):
        assignment = assign_port_finding_sessions({}, found)
    assert [u.mouse for u in assignment.unassigned] == ["m9"]


def test_mice_are_assigned_separately():
    by_problem = trials(
        p1={
            "m1": {"ses-2_date-20260102": bandit(A)},
            "m2": {"ses-2_date-20260102": bandit(B)},
        }
    )
    found = {"m1": {"ses-1_date-20260101": only_port_finding(A)}}
    assignment = assign_port_finding_sessions(by_problem, found)
    assert port_finding_sessions_for_problem(assignment, by_problem[1]) == found
    assert port_finding_sessions_for_problem(assignment, {"m2": by_problem[1]["m2"]}) == {}


def test_the_order_is_the_session_order_not_the_dict_order():
    by_problem = {
        2: {"m1": {"ses-3_date-20260103": bandit(B)}},
        1: {"m1": {"ses-1_date-20260101": bandit(A)}},
    }
    found = {"m1": {"ses-2_date-20260102": only_port_finding(B)}}
    assignment = assign_port_finding_sessions(by_problem, found)
    assert port_finding_sessions_for_problem(assignment, by_problem[2]) == found


def test_the_lookup_follows_the_session_not_the_problem_key():
    # the runner regroups problems by task type: the key changes, the sessions do not
    by_problem = trials(p7={"m1": {"ses-2_date-20260102": bandit(A)}})
    found = {"m1": {"ses-1_date-20260101": only_port_finding(A)}}
    assignment = assign_port_finding_sessions(by_problem, found)
    regrouped = {"m1": dict(by_problem[7]["m1"])}
    assert port_finding_sessions_for_problem(assignment, regrouped) == found


def test_another_port_finding_session_with_other_towers_in_between_blocks_the_assignment():
    by_problem = trials(p1={"m1": {"ses-3_date-20260103": bandit(B)}})
    found = {
        "m1": {
            "ses-1_date-20260101": only_port_finding(B),
            "ses-2_date-20260102": only_port_finding(A),
        }
    }
    with pytest.warns(UserWarning, match=r"m1 ses-1_date-20260101.*ses-2_date-20260102"):
        assignment = assign_port_finding_sessions(by_problem, found)
    assert [u.session for u in assignment.unassigned] == [
        "ses-1_date-20260101",
        "ses-2_date-20260102",
    ]
    assert port_finding_sessions_for_problem(assignment, by_problem[1]) == {}


def test_a_port_finding_session_in_between_with_the_same_towers_does_not_block():
    by_problem = trials(p1={"m1": {"ses-3_date-20260103": bandit(B)}})
    found = {
        "m1": {
            "ses-1_date-20260101": only_port_finding(B),
            "ses-2_date-20260102": only_port_finding(B),
        }
    }
    assignment = assign_port_finding_sessions(by_problem, found)
    assert assignment.unassigned == []


def test_sessions_on_the_same_date_are_ordered_by_session_number():
    # ses-62 (bandit) and ses-63 (port finding only) share a date; ses-64 follows on a later one
    by_problem = trials(
        p1={"m1": {"ses-62_date-20260925": bandit(A)}},
        p2={"m1": {"ses-64_date-20260928": bandit(B)}},
    )
    found = {"m1": {"ses-63_date-20260925": only_port_finding(B)}}
    assignment = assign_port_finding_sessions(by_problem, found)
    assert port_finding_sessions_for_problem(assignment, by_problem[2]) == found
    assert port_finding_sessions_for_problem(assignment, by_problem[1]) == {}


def test_the_segments_of_a_split_session_are_ordered_by_part():
    by_problem = trials(p1={"m1": {"ses-5_date-20260105_part-02": bandit(B)}})
    found = {"m1": {"ses-5_date-20260105_part-01": only_port_finding(B)}}
    assignment = assign_port_finding_sessions(by_problem, found)
    assert port_finding_sessions_for_problem(assignment, by_problem[1]) == found


def test_nothing_to_assign_gives_an_empty_assignment_without_warnings():
    by_problem = trials(p1={"m1": {"ses-1_date-20260101": bandit(A)}})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assignment = assign_port_finding_sessions(by_problem, {})
    assert assignment.unassigned == []
    assert port_finding_sessions_for_problem(assignment, by_problem[1]) == {}


def test_a_session_with_unorderable_keys_raises():
    by_problem = trials(p1={"m1": {"ses-2_date-20260102": bandit(A)}})
    with pytest.raises(ValueError, match="m1"):
        assign_port_finding_sessions(by_problem, {"m1": {"first": only_port_finding(A)}})
