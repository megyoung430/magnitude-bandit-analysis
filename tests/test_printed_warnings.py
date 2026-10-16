"""Problems the analyses used to print are Python warnings, so the warnings log has them."""

import warnings

import pytest

import src.pipelines.open_field as pipeline
from src.behavior_analysis import get_good_reversal_info as good
from src.behavior_analysis.get_num_long_pokes import get_num_long_pokes
from src.behavior_analysis.warnings_log import warnings_log_table
from src.behavior_import.warning_records import record_warnings
from tests.review_fixtures import cohort, session


def test_a_session_without_long_poke_counts_warns_instead_of_printing(capsys):
    trials = {
        "m1": {"ses-1": {"num_long_pokes": [1, 2]}},
        "m2": {"ses-1": {}, "ses-2": {"num_long_pokes": [3]}},
    }
    with pytest.warns(UserWarning, match="Missing num_long_pokes for m2 ses-1"):
        get_num_long_pokes(trials)
    assert capsys.readouterr().out == ""


def test_a_session_without_trials_is_skipped_with_a_warning(monkeypatch, tmp_path, capsys):
    drawn = []
    monkeypatch.setattr(pipeline, "plot_single_session", lambda session, **kw: drawn.append(kw))
    sessions = {"m1": {"ses-1": {"trial_info": [1]}, "ses-2": {"trial_info": []}}}
    with pytest.warns(UserWarning, match=r"Skipping m1 \| ses-2\. There are no trials\."):
        pipeline.run_single_session_analysis(sessions, tmp_path)
    assert len(drawn) == 1
    assert "Skipping" not in capsys.readouterr().out


def test_a_warning_the_analysis_raises_reaches_the_warnings_log():
    _, index, _, _ = cohort({1: {"M1": {"ses-1_date-20260101": session()}}})
    empty = {
        "trial": [1, 2],
        "has_good": False,
        "blocks": [1, 2],
        "reward_magnitudes_by_tower": {},
        "choices_by_tower": {},
        "choices_by_rank": {},
    }
    with warnings.catch_warnings(), record_warnings() as recorded:
        warnings.simplefilter("always")
        good.get_good_reversal_info({"M1": {"ses-1_date-20260101": empty}})
    table = warnings_log_table(recorded, index)
    (row,) = table.to_dict("records")
    assert row["category"] == "ReversalWarning"
    assert row["message"].startswith("M1: no towers found")
    assert row["mouse"] == "M1"
