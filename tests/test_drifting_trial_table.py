"""The drifting trial table: one row per trial, in order, from importer session dicts."""

import pytest

from src.behavior_analysis.drifting_trial_table import COLUMNS, drifting_trial_table
from tests.all_cohorts_fixtures import SESSIONS_PER_PROBLEM, TRIALS_PER_SESSION, synthetic_cohort


def _session(choices, magnitudes, blocks=None, initiation="B2"):
    return {
        "choice": choices,
        "reward_magnitudes": magnitudes,
        "blocks": blocks if blocks is not None else [1] * len(choices),
        "initiation_tower": initiation,
    }


MAGS = {"A1": 4, "B1": 2, "C1": 0}


def test_synthetic_cohort_table_has_every_trial_in_order():
    cohort = synthetic_cohort(["m1", "m2"])
    problems = {p: cohort[p] for p in (2, 3)}
    table = drifting_trial_table(problems)
    assert list(table.columns) == COLUMNS
    per_problem = TRIALS_PER_SESSION * SESSIONS_PER_PROBLEM
    assert len(table) == 2 * 2 * per_problem
    for _, rows in table.groupby(["mouse", "problem"]):
        assert rows["trial"].tolist() == list(range(per_problem))
        assert rows["session_index"].is_monotonic_increasing
    assert all(
        m[c] == r for m, c, r in zip(table.magnitudes, table.choice, table.reward, strict=True)
    )


def test_missing_trial_is_dropped_and_flags_the_next():
    session = _session(["A1", None, "B1"], [MAGS, MAGS, MAGS])
    table = drifting_trial_table({1: {"m": {"ses-1_date-20260101": session}}})
    assert table["choice"].tolist() == ["A1", "B1"]
    assert table["trial"].tolist() == [0, 1]
    assert table["trial_in_session"].tolist() == [0, 2]
    assert table["after_gap"].tolist() == [False, True]


def test_trials_number_on_across_sessions():
    first = _session(["A1", "A1"], [MAGS, MAGS])
    second = _session(["B1"], [MAGS])
    sessions = {"ses-2_date-20260102": second, "ses-1_date-20260101": first}
    table = drifting_trial_table({1: {"m": sessions}})
    assert table["session"].tolist() == ["ses-1_date-20260101"] * 2 + ["ses-2_date-20260102"]
    assert table["trial"].tolist() == [0, 1, 2]
    assert table["session_index"].tolist() == [0, 0, 1]


def test_misaligned_lists_raise():
    session = _session(["A1", "B1"], [MAGS])
    with pytest.raises(ValueError, match="must align"):
        drifting_trial_table({1: {"m": {"ses-1_date-20260101": session}}})


def test_choice_not_offered_raises():
    session = _session(["C3"], [MAGS])
    with pytest.raises(ValueError, match="not among the offered ports"):
        drifting_trial_table({1: {"m": {"ses-1_date-20260101": session}}})


def test_no_trials_gives_empty_table_with_columns():
    table = drifting_trial_table({})
    assert table.empty and list(table.columns) == COLUMNS
