"""The suspect-reversal review table lists exactly the reversals get_good_reversal_info drops."""

import re
import warnings

import pandas as pd
import pytest

from src.behavior_analysis.get_good_reversal_info import ReversalWarning, get_good_reversal_info
from src.behavior_analysis.suspect_reversal_table import (
    COLUMNS,
    suspect_reversal_table,
    suspect_reversal_table_by_problem,
    write_suspect_reversal_table,
)

TOWERS = "ABC"
A_BEST, B_BEST, TIE_AB = (4, 1, 2), (1, 4, 2), (4, 4, 2)


def session(good, magnitudes, **extra):
    """A counter session; *magnitudes* is one ``(A, B, C)`` triple per trial."""
    n = len(magnitudes)
    return {
        "trial": list(range(1, n + 1)),
        "blocks": [1] * n,
        "has_good": True,
        "good_reversals": good,
        "reward_magnitudes_by_tower": {t: [m[i] for m in magnitudes] for i, t in enumerate(TOWERS)},
        "choices_by_tower": {t: [0] * n for t in TOWERS},
        "choices_by_rank": {r: [0] * n for r in ("best", "second", "third")},
        **extra,
    }


def frame(path):
    df = pd.DataFrame({"x": [0]})
    df.attrs["source_path"] = path
    return df


def mixed_subject():
    """One kept reversal and one of every dropped kind.

    ses-1: kept (A to B at trial 2), best unchanged (B to B at trial 4)
    ses-2: tie for best at trial 2, missing magnitude at trial 4
    ses-3: a between-runs event at trial 2 whose best tower looks unchanged
    ses-4: a jump of two at trial 2 with an unchanged best (two reversals)
    """
    return {
        "ses-1": session(
            [0, 0, 1, 1, 2, 2],
            [A_BEST, A_BEST, B_BEST, B_BEST, B_BEST, B_BEST],
            trial_variables=[{"trial": [1, 2, 3]}, {"trial": [1, 2, 3]}],
            data=[frame("/raw/sub/run-a.tsv"), frame("/raw/sub/run-b.tsv")],
        ),
        "ses-2": session(
            [0, 0, 1, 1, 2, 2], [A_BEST, A_BEST, TIE_AB, TIE_AB, (4, 1, None), (4, 1, 2)]
        ),
        "ses-3": session(
            [0, 0, 1, 1],
            [A_BEST] * 4,
            reversal_boundary_events=[dict(kind="good", index=2, count=1, source="between_runs")],
        ),
        "ses-4": session([0, 0, 2, 2], [A_BEST] * 4),
    }


def dropped_per_warning(data):
    """``{(subject, session, trial): reversals}`` from the text of the drop warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        get_good_reversal_info(data)
    (text,) = [str(w.message) for w in caught if "dropped" in str(w.message)]
    listing = text.split(": ", 1)[1].split("]. A good")[0]
    found = {}
    for entry in listing.split("; "):
        subj, ses, trial = re.match(r"(\S+) (\S+) trial (\d+)", entry).groups()
        count = re.search(r"(\d+) reversals on this trial", entry)
        found[(subj, ses, int(trial))] = int(count.group(1)) if count else 1
    return found


def table_per_boundary(table):
    """Row counts summed per ``(subject, session, session_trial)``."""
    grouped = table.groupby(["subject", "session", "session_trial"])["count"].sum()
    return {k: int(v) for k, v in grouped.items()}


def test_rows_correspond_exactly_to_the_reversals_get_good_reversal_info_drops():
    data = {"m": mixed_subject()}
    table = suspect_reversal_table(data)
    assert table_per_boundary(table) == dropped_per_warning(data)
    assert table["count"].sum() == 1 + 1 + 1 + 1 + 2 == 6


def test_every_reason_appears_and_kept_reversals_do_not():
    table = suspect_reversal_table({"m": mixed_subject()})
    assert sorted(set(table["reason"])) == [
        "best tower unchanged",
        "missing magnitude",
        "tie for best",
        "uncheckable between-run or terminal",
    ]
    # the kept reversal is ses-1 trial 2 (merged 2)
    assert not ((table["session"] == "ses-1") & (table["session_trial"] == 2)).any()
    assert 2 not in set(table["merged_trial"])


def test_a_between_runs_event_is_labelled_uncheckable_not_best_unchanged():
    table = suspect_reversal_table({"m": mixed_subject()})
    (row,) = table[table["session"] == "ses-3"].to_dict("records")
    assert row["reason"] == "uncheckable between-run or terminal"
    assert row["count"] == 1


def test_a_jump_of_two_is_one_row_with_count_two():
    table = suspect_reversal_table({"m": mixed_subject()})
    (row,) = table[table["session"] == "ses-4"].to_dict("records")
    assert (row["count"], row["good_reversal_number"], row["reason"]) == (
        2,
        7,  # the merged counter: 2 + 2 + 1 from the sessions before, then the jump of 2
        "best tower unchanged",
    )


def test_an_event_and_an_in_run_reversal_on_one_trial_split_into_two_rows():
    sess = session(
        [0, 0, 2, 2],
        [A_BEST] * 4,
        reversal_boundary_events=[dict(kind="good", index=2, count=1, source="between_runs")],
    )
    table = suspect_reversal_table({"m": {"ses-1": sess}})
    assert sorted(zip(table["reason"], table["count"], strict=True)) == [
        ("best tower unchanged", 1),
        ("uncheckable between-run or terminal", 1),
    ]


def test_row_carries_where_magnitudes_and_best_towers():
    table = suspect_reversal_table({"m": mixed_subject()})
    (row,) = table[(table["session"] == "ses-1")].to_dict("records")
    assert row["subject"] == "m"
    assert (row["session_trial"], row["merged_trial"]) == (4, 4)
    assert (row["run"], row["file"]) == (1, "run-b.tsv")
    assert (row["good_reversal_number"], row["reason"]) == (2, "best tower unchanged")
    assert row["magnitudes_before"] == "A=1, B=4, C=2"
    assert row["magnitudes_after"] == "A=1, B=4, C=2"
    assert (row["best_before"], row["best_after"]) == ("B", "B")


def test_missing_and_tied_magnitudes_are_readable():
    table = suspect_reversal_table({"m": mixed_subject()})
    missing = table[table["reason"] == "missing magnitude"].iloc[0]
    assert missing["magnitudes_after"] == "A=4, B=1, C=missing"
    assert pd.isna(missing["best_after"]) and missing["best_before"] == "A"
    tie = table[table["reason"] == "tie for best"].iloc[0]
    assert tie["magnitudes_after"] == "A=4, B=4, C=2"
    assert tie["best_after"] == "A"  # first listed, as the analyses resolve ties


def test_run_and_file_are_missing_when_not_recorded():
    table = suspect_reversal_table({"m": mixed_subject()})
    row = table[table["session"] == "ses-2"].iloc[0]
    assert pd.isna(row["run"]) and pd.isna(row["file"])


def test_columns_order_and_dtypes():
    table = suspect_reversal_table({"m": mixed_subject()})
    assert list(table.columns) == list(COLUMNS)
    assert list(COLUMNS) == [
        "subject",
        "session",
        "session_trial",
        "merged_trial",
        "run",
        "file",
        "reason",
        "good_reversal_number",
        "magnitudes_before",
        "magnitudes_after",
        "best_before",
        "best_after",
        "count",
    ]
    dtypes = {c: str(t) for c, t in table.dtypes.items()}
    assert dtypes["session_trial"] == "Int64" and dtypes["run"] == "Int64"
    assert dtypes["merged_trial"] == "int64" and dtypes["count"] == "int64"
    assert dtypes["good_reversal_number"] == "Int64"
    assert dtypes["subject"] == dtypes["reason"] == "object"


def test_rows_are_ordered_by_subject_then_merged_trial():
    data = {"z": mixed_subject(), "a": mixed_subject()}
    table = suspect_reversal_table(data)
    assert list(dict.fromkeys(table["subject"])) == ["z", "a"]  # the order of the input
    for _, rows in table.groupby("subject", sort=False):
        assert rows["merged_trial"].is_monotonic_increasing


def test_empty_input_gives_an_empty_table_with_the_same_columns_and_dtypes():
    full = suspect_reversal_table({"m": mixed_subject()})
    for empty in (suspect_reversal_table({}), suspect_reversal_table({"m": {}})):
        assert len(empty) == 0
        assert list(empty.columns) == list(full.columns)
        assert empty.dtypes.equals(full.dtypes)


def test_no_dropped_reversals_gives_an_empty_table():
    sess = session([0, 0, 1, 1], [A_BEST, A_BEST, B_BEST, B_BEST])
    assert len(suspect_reversal_table({"m": {"ses-1": sess}})) == 0


def test_the_table_does_not_warn_or_change_the_input():
    data = {"m": mixed_subject()}
    before = repr(data)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ReversalWarning)
        suspect_reversal_table(data)
    assert repr(data) == before


def test_max_zero_count_is_passed_through_like_get_good_reversal_info():
    zeros = (4, 0, 0)
    data = {"m": {"ses-1": session([0, 0, 1, 1], [zeros] * 4)}}
    assert len(suspect_reversal_table(data)) == 0  # skipped by the zero-reward gate
    assert len(suspect_reversal_table(data, max_zero_count=2)) == 1


def test_by_problem_adds_a_leading_problem_column():
    by_problem = {1: {"m": mixed_subject()}, 2: {"m": mixed_subject()}}
    table = suspect_reversal_table_by_problem(by_problem)
    assert list(table.columns) == ["problem", *COLUMNS]
    one = suspect_reversal_table({"m": mixed_subject()})
    assert len(table) == 2 * len(one)
    assert set(table["problem"]) == {1, 2}
    assert table[table["problem"] == 2][list(COLUMNS)].reset_index(drop=True).equals(one)


def test_by_problem_with_no_problems_is_empty_with_problem_column():
    table = suspect_reversal_table_by_problem({})
    assert list(table.columns) == ["problem", *COLUMNS] and len(table) == 0


def cells(frame):
    """Row-wise cell values with every missing value as ``None``."""
    rows = frame.astype(object).itertuples(index=False)
    return [[None if pd.isna(v) else v for v in row] for row in rows]


def test_csv_round_trip_creates_the_folder(tmp_path):
    table = suspect_reversal_table({"m": mixed_subject()})
    path = tmp_path / "new" / "folder" / "suspect.csv"
    write_suspect_reversal_table(table, path)
    back = pd.read_csv(path, dtype={"run": "Int64", "session_trial": "Int64"})
    assert list(back.columns) == list(COLUMNS)
    assert len(back) == len(table)
    back = back.astype(table.dtypes.to_dict())
    assert cells(back) == cells(table)
    assert back.dtypes.equals(table.dtypes)


def test_csv_of_an_empty_table_keeps_the_header(tmp_path):
    path = tmp_path / "empty.csv"
    write_suspect_reversal_table(suspect_reversal_table({}), path)
    assert path.read_text().strip() == ",".join(COLUMNS)


def test_write_rejects_a_table_without_the_expected_columns(tmp_path):
    with pytest.raises(ValueError, match="columns"):
        write_suspect_reversal_table(pd.DataFrame({"a": [1]}), tmp_path / "x.csv")


def test_write_accepts_the_task_type_columns_and_keeps_them_in_the_csv(tmp_path):
    table = suspect_reversal_table({"m": mixed_subject()})
    typed = table.assign(task_type="drifting", problem_in_type=1, overall_problem=3)
    typed = typed[["task_type", "problem_in_type", "overall_problem", *COLUMNS]]
    path = tmp_path / "typed.csv"
    write_suspect_reversal_table(typed, path)
    header = path.read_text().splitlines()[0]
    assert header == ",".join(["task_type", "problem_in_type", "overall_problem", *COLUMNS])
