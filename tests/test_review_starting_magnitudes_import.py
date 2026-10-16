"""Starting magnitudes compared across sessions, through the real importer.

Raw pyControl files are written to a temporary tree, imported and numbered like a cohort, and the
table is built from what the importer returns. The sessions between two sessions of a problem
(one without trials, one left out as a settings error) must not be what a session is compared
with.
"""

import pandas as pd
import pytest

from src.behavior_analysis import starting_magnitudes as sm
from src.behavior_analysis.review_context import ProblemIndex
from src.behavior_analysis.starting_magnitudes import starting_magnitudes_table
from src.behavior_import.load_trials import load_trials_with_log
from src.behavior_import.problem_numbering import ProblemNumberingWarning
from src.behavior_import.task_types import (
    number_within_task_type,
    problem_map_table,
    regroup_by_task_type,
)
from tests.boundary_fixtures import DRIFTING_TASK, TOWERS, prints, recording, snapshot

ZERO_B3 = {"B1": 4, "B3": 0, "C2": 2}
ZERO_B3_DRIFTED = {"B1": 3, "B3": 0, "C2": 2}
ZERO_B3_DRIFTED_AGAIN = {"B1": 2, "B3": 0, "C2": 2}
ZERO_B1 = {"B1": 0, "B3": 4, "C2": 3}
ALL_TWO = {"B1": 2, "B3": 2, "C2": 2}
OTHER_TOWERS = ("A3", "B2", "C1")
OTHER_ARMS = {"A3": 0, "B2": 3, "C1": 4}


def with_layout(frame, towers, initiation):
    """*frame* with the problem layout in its ``run_start`` (the importer numbers by it)."""
    start = frame["subtype"] == "run_start"
    frame = frame.copy()
    layout = (
        f', "curr_choice_tows": {list(towers)}, "curr_initiation_tow": "{initiation}", '
        '"found_ports": false, "num_t_found": 0}'
    ).replace("'", '"')
    frame.loc[start, "content"] = frame.loc[start, "content"].str[:-1] + layout
    return frame


def session_frames(mags, trials, tot_blocks, *, end=None, towers=TOWERS, initiation="B2"):
    """One recording of a session that starts on *mags* with the session total *tot_blocks*."""
    start = snapshot(mags, blocks=1, tot_blocks=tot_blocks)
    end = snapshot(end or mags, blocks=1, tot_blocks=tot_blocks)
    printed = [
        {**trial, "c_tows": list(towers), "chosen_tow": towers[0]}
        for trial in prints(mags, trials, blocks=1)
    ]
    return with_layout(recording(start, printed, end), towers, initiation)


def write_session(root, number, frame):
    """The recording as a pyControl ``.tsv`` of session *number* of mouse M1."""
    folder = root / "sub-01_id-M1" / f"ses-{number}_date-2026010{number}"
    folder.mkdir(parents=True)
    rows = pd.concat(
        [
            pd.DataFrame([(0.0, "info", "task_name", DRIFTING_TASK)], columns=frame.columns),
            frame,
        ]
    )
    rows.to_csv(folder / f"M1-2026-01-0{number}-100000.tsv", sep="\t", index=False)


def table_of(root):
    """The starting-magnitudes table of the tree, with the import log's skips used."""
    trials, log = load_trials_with_log(
        root, cache_dir=root / "cache", use_cache=False, classify_task_types=True
    )
    numbering = number_within_task_type(trials)
    index = ProblemIndex.from_cohort(trials, problem_map_table(trials, numbering))
    grouped = regroup_by_task_type(trials, numbering)
    return starting_magnitudes_table(grouped, log.skips, index)


def one_row(table, session="ses-4_date-20260104"):
    """The row of *session* (ses-2 drifts a step from ses-1 and is always listed too)."""
    (row,) = table[table["session"] == session].to_dict("records")
    return row


@pytest.fixture
def first_two_sessions(tmp_path):
    write_session(tmp_path, 1, session_frames(ZERO_B3, 3, 1))
    write_session(tmp_path, 2, session_frames(ZERO_B3_DRIFTED, 3, 1))
    return tmp_path


def no_trial_session(number, tmp_path):
    """A session whose recording logs a ``run_start`` on other magnitudes but no trial."""
    frame = recording(snapshot(ZERO_B1, blocks=1, tot_blocks=2), [], snapshot(ZERO_B1))
    write_session(tmp_path, number, with_layout(frame, TOWERS, "B2"))


def test_a_session_without_trials_in_between_is_not_what_the_next_session_is_compared_with(
    first_two_sessions,
):
    no_trial_session(3, first_two_sessions)
    write_session(first_two_sessions, 4, session_frames(ZERO_B3_DRIFTED_AGAIN, 3, 1))
    row = one_row(table_of(first_two_sessions))
    assert (row["session"], row["previous_session"]) == (
        "ses-4_date-20260104",
        "ses-2_date-20260102",
    )
    assert row["magnitudes_end_previous"] == "B1=3, B3=0, C2=2"
    assert row["magnitudes_start"] == "B1=2, B3=0, C2=2"
    assert row["likely_cause"] == sm.DRIFT
    assert row["block_total_change"] == 0


def test_a_change_the_empty_session_hides_is_blamed_on_it_when_nothing_else_explains_it(
    first_two_sessions,
):
    no_trial_session(3, first_two_sessions)
    write_session(first_two_sessions, 4, session_frames(ZERO_B1, 3, 1))
    row = one_row(table_of(first_two_sessions))
    assert row["previous_session"] == "ses-2_date-20260102"
    assert row["likely_cause"] == sm.PREVIOUS_SESSION_NO_TRIALS


def test_a_total_logged_after_the_empty_session_is_counted_against_the_last_trial_before_it(
    first_two_sessions,
):
    no_trial_session(3, first_two_sessions)
    write_session(first_two_sessions, 4, session_frames(ZERO_B1, 3, 2))
    row = one_row(table_of(first_two_sessions))
    assert row["block_total_change"] == 1
    assert row["likely_cause"] == sm.LOGGED_AND_CONSISTENT


def test_the_session_after_a_settings_error_is_compared_with_the_session_before_it(
    first_two_sessions,
):
    write_session(first_two_sessions, 3, session_frames(OTHER_ARMS, 3, 1, towers=OTHER_TOWERS))
    write_session(first_two_sessions, 4, session_frames(ZERO_B1, 3, 2))
    with pytest.warns(ProblemNumberingWarning):
        table = table_of(first_two_sessions)
    row = one_row(table)
    assert (row["session"], row["previous_session"]) == (
        "ses-4_date-20260104",
        "ses-2_date-20260102",
    )
    assert row["magnitudes_end_previous"] == "B1=3, B3=0, C2=2"
    assert row["likely_cause"] == sm.LOGGED_AND_CONSISTENT


def test_a_settings_error_with_the_same_arms_is_not_compared_under_the_other_sessions_name(
    first_two_sessions,
):
    # only the initiation tower differs, so the arms of the two layouts are the same
    write_session(first_two_sessions, 3, session_frames(ALL_TWO, 3, 0, initiation="A2"))
    write_session(first_two_sessions, 4, session_frames(ZERO_B3_DRIFTED_AGAIN, 3, 1))
    with pytest.warns(ProblemNumberingWarning):
        table = table_of(first_two_sessions)
    row = one_row(table)
    assert row["previous_session"] == "ses-2_date-20260102"
    assert row["magnitudes_end_previous"] == "B1=3, B3=0, C2=2"
    assert row["likely_cause"] == sm.DRIFT


def test_the_first_session_after_the_gap_of_a_problem_that_starts_there_is_not_listed(tmp_path):
    write_session(tmp_path, 1, session_frames(ZERO_B3, 3, 1))
    assert table_of(tmp_path).empty
