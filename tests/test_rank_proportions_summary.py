import numpy as np
import pytest

from src.behavior_analysis.rank_proportions_summary import (
    mean_rank_proportions_by_subject,
    rank_mean_and_se,
)

nan = float("nan")


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # nanmean of an all-NaN rank
def test_mean_rank_proportions_by_subject_skips_empty_and_nan():
    rows = {
        "A": [
            {"best_prop": 0.5, "second_prop": 0.3, "third_prop": 0.2},
            {"best_prop": 0.7, "second_prop": nan, "third_prop": 0.0},
        ],
        "B": [],
        "C": [{"best_prop": 1.0}],
    }
    out = mean_rank_proportions_by_subject(rows)
    assert list(out) == ["A", "C"]
    assert out["A"] == pytest.approx({"best": 0.6, "second": 0.3, "third": 0.1})
    assert out["C"]["best"] == 1.0 and np.isnan(out["C"]["second"])


def test_rank_mean_and_se_across_mice():
    per_mouse = {
        "A": {"best": 0.6, "second": 0.3, "third": 0.1},
        "B": {"best": 0.8, "second": 0.1, "third": 0.1},
    }
    means, ses = rank_mean_and_se(per_mouse)
    assert means == pytest.approx([0.7, 0.2, 0.1])
    # sd(0.6, 0.8) = 0.1414..., divided by sqrt(2)
    assert ses == pytest.approx([0.1, 0.1, 0.0])


def test_rank_se_counts_only_non_nan_mice():
    per_mouse = {
        "A": {"best": 0.6, "second": 0.3, "third": 0.1},
        "B": {"best": 0.8, "second": 0.1, "third": 0.1},
        "C": {"best": nan, "second": 0.2, "third": 0.1},
    }
    _, ses = rank_mean_and_se(per_mouse)
    assert ses[0] == pytest.approx(np.std([0.6, 0.8], ddof=1) / np.sqrt(2))
    assert ses[1] == pytest.approx(np.std([0.3, 0.1, 0.2], ddof=1) / np.sqrt(3))


def test_rank_se_is_nan_for_a_single_mouse():
    per_mouse = {"A": {"best": 0.6, "second": 0.3, "third": 0.1}}
    with pytest.warns(RuntimeWarning):
        means, ses = rank_mean_and_se(per_mouse)
    assert means == pytest.approx([0.6, 0.3, 0.1])
    assert all(np.isnan(se) for se in ses)


def test_rank_se_is_nan_when_one_non_nan_mouse_remains():
    per_mouse = {
        "A": {"best": 0.6, "second": 0.3, "third": 0.1},
        "B": {"best": nan, "second": 0.1, "third": 0.1},
    }
    with pytest.warns(RuntimeWarning):
        _, ses = rank_mean_and_se(per_mouse)
    assert np.isnan(ses[0]) and not np.isnan(ses[1])
