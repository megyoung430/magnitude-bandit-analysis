"""Non-choice long pokes per trial within one problem, aligned by relative session."""

import numpy as np
import pytest

from src.behavior_analysis.long_pokes_by_session import (
    problem_mean_long_pokes,
    session_mean_long_pokes,
    summarize_pokes_by_session,
    to_relative_sessions,
)


def sess(values):
    return {"num_long_pokes": values}


def test_session_mean_is_over_the_trials_of_the_session():
    means = session_mean_long_pokes({"m": {"ses-1_date-20260101": sess([0, 1, 2, 1])}})
    assert means == {"m": {"ses-1_date-20260101": 1.0}}


def test_trials_without_a_value_do_not_count():
    means = session_mean_long_pokes({"m": {"ses-1_date-20260101": sess([0, None, 3])}})
    assert means["m"]["ses-1_date-20260101"] == 1.5


def test_session_without_values_has_nan_mean():
    means = session_mean_long_pokes({"m": {"ses-1_date-20260101": sess([])}})
    assert np.isnan(means["m"]["ses-1_date-20260101"])


def test_session_missing_num_long_pokes_raises_instead_of_being_skipped():
    with pytest.raises(ValueError, match="m.*ses-1_date-20260101.*num_long_pokes"):
        session_mean_long_pokes({"m": {"ses-1_date-20260101": {}}})


def test_sessions_are_in_date_order_not_dict_order():
    sessions = {"ses-10_date-20260110": sess([2]), "ses-9_date-20260109": sess([1])}
    assert list(session_mean_long_pokes({"m": sessions})["m"]) == [
        "ses-9_date-20260109",
        "ses-10_date-20260110",
    ]


def test_relative_sessions_start_at_one_for_every_mouse():
    per_mouse = {
        "a": {"ses-6_date-20260106": 2.0, "ses-5_date-20260105": 1.0},
        "b": {"ses-9_date-20260109": 5.0},
        "average_across_subjects": {"x": 0.0},
    }
    assert to_relative_sessions(per_mouse) == {
        "a": {"rel_ses-01": 1.0, "rel_ses-02": 2.0},
        "b": {"rel_ses-01": 5.0},
    }


def test_relative_sessions_with_unorderable_keys_raise():
    with pytest.raises(ValueError, match="m"):
        to_relative_sessions({"m": {"ses-2_date-20260101": 1.0, "ses-1_date-20260102": 2.0}})


def test_across_mouse_summary_on_relative_sessions_pools_by_relative_index():
    per_mouse = {
        "a": {"ses-5_date-20260105": 1.0, "ses-6_date-20260106": 2.0},
        "b": {"ses-9_date-20260109": 3.0},
    }
    summary = summarize_pokes_by_session(to_relative_sessions(per_mouse))
    assert summary.sessions == ["rel_ses-01", "rel_ses-02"]
    np.testing.assert_allclose(summary.mean, [2.0, 2.0])
    assert np.isnan(summary.se[1])  # one mouse: no standard error


def test_problem_mean_pools_trials_across_sessions_of_a_mouse():
    sessions = {"ses-1_date-20260101": sess([0, 0]), "ses-2_date-20260102": sess([3, 3, 3, 3])}
    assert problem_mean_long_pokes({"m": sessions}) == {"m": 2.0}
