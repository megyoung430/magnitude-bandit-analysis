import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis.reversals_by_session import (
    centered_moving_average,
    cumulative_by_session,
    curves_by_subject,
    mean_across_subjects,
    moving_average_by_session,
    reversal_counts_by_session,
    reversal_kinds,
    reversal_totals_by_subject,
    session_number,
)


def perf_session(good, bad):
    return {"has_good": True, "has_bad": True, "reversal_totals": {"good": good, "bad": bad}}


def blocks_session(n_blocks):
    return {"blocks": list(range(1, n_blocks + 1))}


@pytest.fixture
def perf_cohort():
    # session dicts deliberately out of order
    return {
        "m2": {
            "ses-2_date-20260102": perf_session(1, 0),
            "ses-1_date-20260101": perf_session(2, 1),
        },
        "m1": {"ses-10": perf_session(0, 3)},
    }


def test_session_number():
    assert session_number("ses-12_date-20260116") == 12
    with pytest.raises(ValueError, match="ses-"):
        session_number("session12")


def test_reversal_counts_by_session_sorted_per_subject(perf_cohort):
    counts = reversal_counts_by_session(perf_cohort)
    assert counts[["subject", "session", "session_label"]].values.tolist() == [
        ["m1", 10, "ses-10"],
        ["m2", 1, "ses-01"],
        ["m2", 2, "ses-02"],
    ]
    assert counts["good"].tolist() == [0, 2, 1]
    assert counts["bad"].tolist() == [3, 1, 0]
    assert counts["total"].tolist() == [3, 3, 1]


def test_reversal_counts_requires_sessions():
    with pytest.raises(ValueError):
        reversal_counts_by_session({"m1": {}})


def test_reversal_kinds(perf_cohort):
    assert reversal_kinds(reversal_counts_by_session(perf_cohort)) == ("good", "bad")
    blocks = {"m1": {"ses-1": blocks_session(4)}}
    counts = reversal_counts_by_session(blocks)
    assert reversal_kinds(counts) == ("total",)
    assert counts["total"].tolist() == [3]


def test_reversal_kinds_rejects_mixed_paradigms(perf_cohort):
    perf_cohort["m1"]["ses-11"] = blocks_session(3)
    with pytest.raises(ValueError, match="mix"):
        reversal_kinds(reversal_counts_by_session(perf_cohort))


def test_cumulative_by_session_is_per_subject(perf_cohort):
    counts = reversal_counts_by_session(perf_cohort)
    cum = cumulative_by_session(counts, ("good", "bad"))
    assert cum["good"].tolist() == [0, 2, 3]
    assert cum["bad"].tolist() == [3, 1, 1]
    assert counts["good"].tolist() == [0, 2, 1]  # input untouched


def test_centered_moving_average_shrinks_at_edges():
    np.testing.assert_allclose(centered_moving_average([1, 2, 3, 4, 5], 3), [1.5, 2, 3, 4, 4.5])


def test_centered_moving_average_even_window_widens_to_odd():
    np.testing.assert_allclose(centered_moving_average([1, 2, 3, 4, 5], 4), [2, 2.5, 3, 3.5, 4])


def test_centered_moving_average_ignores_nan_and_window_one():
    np.testing.assert_allclose(centered_moving_average([1, np.nan, 3], 3), [1, 2, 3])
    np.testing.assert_allclose(centered_moving_average([1, 5], 1), [1, 5])


def test_moving_average_by_session_is_per_subject():
    counts = pd.DataFrame({"subject": ["a", "a", "a", "b"], "total": [0.0, 3.0, 0.0, 9.0]})
    ma = moving_average_by_session(counts, ("total",), window=3)
    np.testing.assert_allclose(ma["total"], [1.5, 1.0, 1.5, 9.0])


def test_curves_and_mean_across_subjects_carry_last_value_forward():
    df = pd.DataFrame({"subject": ["b", "a", "a", "a"], "total": [10.0, 1.0, 2.0, 3.0]})
    curves = curves_by_subject(df, "total")
    assert list(curves) == ["a", "b"]
    np.testing.assert_allclose(mean_across_subjects(curves), [5.5, 6.0, 6.5])


def test_suspect_columns_are_present_and_nan_without_good_bad_counters():
    counts = reversal_counts_by_session(
        {"m": {"ses-1": blocks_session(3), "ses-2": blocks_session(2)}}
    )
    assert counts["total"].tolist() == [2, 1]
    for column in ("suspect_good_reversals", "uncheckable_good_reversals"):
        assert column in counts
        assert counts[column].isna().all()


def test_suspect_columns_are_numbers_with_good_bad_counters(perf_cohort):
    counts = reversal_counts_by_session(perf_cohort)
    assert counts["suspect_good_reversals"].notna().all()
    assert counts["uncheckable_good_reversals"].notna().all()


def test_reversal_totals_by_subject(perf_cohort):
    totals = reversal_totals_by_subject(perf_cohort)
    assert totals.index.tolist() == ["m1", "m2"]
    assert totals.loc["m2", ["good", "bad", "total"]].tolist() == [3, 1, 4]
