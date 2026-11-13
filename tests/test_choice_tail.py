"""The choice tail: an exponentially weighted choice indicator older than the choice lags."""

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis.choice_logit import choice_logit as cl
from src.behavior_analysis.choice_logit import choice_logit_extensions as ext
from tests.test_choice_logit import trial_frame

PORTS = ["A1", "B3", "C2"]


def _frame(choices, session_starts=(0,)):
    magnitudes = [(1, 1, 1)] * len(choices)
    return trial_frame(choices, magnitudes, set(session_starts))


def _tail_column(design):
    return design.X[:, :, list(design.feature_names).index("choice_tail")]


def test_the_tail_by_hand_on_a_sequence():
    choices = ["A1", "A1", "B3", "A1", "C2", "A1", "B3", "B3"]
    design = cl.design_matrix(
        _frame(choices), n_lags=2, extras=(cl.CHOICE_TAIL,), choice_tail_decay=0.5
    )
    tail = _tail_column(design)
    # tail_a(t) = sum over lags j > 2 of 0.5**(j - 3) * [choice at t-j is a]
    index = {"A1": 0, "B3": 1, "C2": 2}
    for t in range(len(choices)):
        expected = np.zeros(3)
        for j in range(3, t + 1):
            expected[index[choices[t - j]]] += 0.5 ** (j - 3)
        assert tail[t] == pytest.approx(expected)


def test_the_tail_is_cleared_at_a_session_start_like_the_choice_lags():
    choices = ["A1"] * 8
    design = cl.design_matrix(
        _frame(choices, session_starts=(0, 5)),
        n_lags=1,
        extras=(cl.CHOICE_TAIL,),
        choice_tail_decay=0.9,
    )
    tail = _tail_column(design)
    # trial 5 starts a session: nothing before it counts; trial 7 has two earlier trials in its
    # session, trial 5 at lag 2 (older than the one lag): weight 0.9**0
    assert tail[5] == pytest.approx(0.0)
    assert tail[6] == pytest.approx(0.0)
    assert tail[7] == pytest.approx([1.0, 0.0, 0.0])


def test_the_tail_comes_after_the_standard_columns_and_is_its_own_group():
    frame = _frame(["A1", "B3", "C2", "A1"])
    plain = cl.design_matrix(frame, n_lags=2)
    with_tail = cl.design_matrix(frame, n_lags=2, extras=(cl.CHOICE_TAIL,), choice_tail_decay=0.9)
    assert with_tail.feature_names == (*plain.feature_names, "choice_tail")
    np.testing.assert_array_equal(with_tail.X[:, :, :-1], plain.X)
    table = cl.weights_table("m", with_tail.feature_names, np.arange(len(with_tail.feature_names)))
    row = table[table["feature"] == "choice_tail"].iloc[0]
    assert row["group"] == cl.CHOICE_TAIL and np.isnan(row["lag"])
    # the choice group (the kernel) does not include the tail
    assert "choice_tail" not in cl.select_features(with_tail, (cl.CHOICE,)).feature_names
    assert cl.select_features(with_tail, (cl.CHOICE_TAIL,)).feature_names == ("choice_tail",)


def test_the_tail_needs_a_decay_in_zero_to_one():
    frame = _frame(["A1", "B3", "C2"])
    with pytest.raises(ValueError, match="choice_tail_decay"):
        cl.design_matrix(frame, extras=(cl.CHOICE_TAIL,))
    with pytest.raises(ValueError, match="choice_tail_decay"):
        cl.design_matrix(frame, extras=(cl.CHOICE_TAIL,), choice_tail_decay=1.0)
    with pytest.raises(ValueError, match="choice_tail_decay"):
        cl.design_matrix(frame, choice_tail_decay=0.9)  # a decay without the column


def test_the_tail_pair_is_scored_held_out_on_a_design_with_the_tail():
    rng = np.random.default_rng(0)
    choices = [PORTS[i] for i in rng.integers(0, 3, 240)]
    frame = pd.concat(
        [_frame(choices[:120]).assign(problem=1), _frame(choices[120:]).assign(problem=2)],
        ignore_index=True,
    )
    design = cl.design_matrix(frame, n_lags=3, extras=(cl.CHOICE_TAIL,), choice_tail_decay=0.9)
    result = ext.cross_validate_variants(design, ext.TAIL_VARIANTS, cl.DEFAULT_L2)
    assert set(result.scores["model"]) == set(ext.TAIL_VARIANTS)
    assert (result.scores["held_out_log_likelihood"] < 0).all()
