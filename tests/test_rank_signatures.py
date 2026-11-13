"""Every rank test on a toy rank agent and a softmax learner, as predicted."""

import warnings

import numpy as np
import pytest

from src.behavior_analysis.rank_tests import rank_result as rr
from src.behavior_analysis.rank_tests import rank_signatures as sg
from src.behavior_analysis.rank_tests import rank_state as rs
from src.behavior_analysis.rank_tests import rank_transfer as rt
from tests.rank_fixtures import inferring_rank_mouse, rank_mouse, softmax_mouse
from tests.strategy_fixtures import KERNEL_Q

SIZE = dict(n_problems=8, sessions=3, trials=200)
# Readouts on which the toy agents (which do not transfer across problems) differ: tests 1 to 4.
SEPARATED = (
    "w_diff", "w_worst", "step_weight", "gain_indicator", "gain_step_vs_smooth", "cross_leave_1",
    "cross_adopt_1", "worst_slope", "rank_minus_value", "sw_rank_minus_value", "scale_slope_first",
)  # fmt: skip


@pytest.fixture(scope="module")
def rank():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sg.analyze_mouse(rank_mouse(seed=1, **SIZE), kq_params=KERNEL_Q)


@pytest.fixture(scope="module")
def softmax():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sg.analyze_mouse(softmax_mouse(seed=1, **SIZE), kq_params=KERNEL_Q)


@pytest.mark.parametrize("name", SEPARATED)
def test_a_rank_agent_differs_from_a_softmax_learner_in_the_predicted_direction(
    rank, softmax, name
):
    difference = rank.readouts[name] - softmax.readouts[name]
    assert np.sign(difference) == sg.PREDICTED[name], (
        name,
        rank.readouts[name],
        softmax.readouts[name],
    )


def test_every_readout_of_a_test_has_a_prediction_or_is_context():
    assert set(sg.PREDICTED) <= {name for names in sg.TESTS.values() for name in names}
    assert sg.which_test("step_weight") == 2 and sg.which_test("n_problems") is None


def test_the_kernel_q_values_give_a_second_set_of_stay_weights(rank, softmax):
    assert {"q_w_best", "q_w_worst", "q_w_diff"} <= set(rank.readouts)
    assert np.isfinite([rank.readouts["q_w_diff"], softmax.readouts["q_w_diff"]]).all()


def test_without_kernel_q_parameters_the_q_readouts_are_absent():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        readouts = sg.analyze_mouse(rank_mouse(seed=1, **SIZE)).readouts
    assert not any(name.startswith("q_") for name in readouts)


def test_curves_come_with_every_test_that_draws_one(rank):
    assert {"crossing_step", "scale", "third_option", "transfer_p100", "cross_believed"} <= set(
        rank.curves["curve"]
    )
    assert list(rank.curves.columns) == rr.CURVE_COLUMNS


def test_merging_results_with_a_repeated_readout_raises():
    one = rr.RankResult({"a": 1.0}, rr.empty_curves())
    with pytest.raises(ValueError, match="named twice"):
        rr.merge(one, one)


def test_a_rank_agent_that_infers_block_changes_goes_to_the_old_zero_arm():
    state_kwargs = dict(n_problems=10, sessions=4)
    infer = inferring_rank_mouse(seed=1, **state_kwargs)
    plain = rank_mouse(seed=1, **state_kwargs)
    inferring = rt.block_inference(rs.mouse_state(infer)).readouts["block_inherit"]
    ignoring = rt.block_inference(rs.mouse_state(plain)).readouts["block_inherit"]
    assert inferring > 0.8 > 0.4 > ignoring


def test_a_mouse_with_too_few_decisions_raises():
    tiny = rank_mouse(seed=1, n_problems=2, sessions=1, trials=60)
    with pytest.raises(sg.UnanalyzableError, match="decisions"):
        sg.analyze_mouse(tiny)


def test_a_mouse_that_never_leaves_is_unanalyzable():
    stuck = rank_mouse(seed=1, **SIZE)
    first_arm = stuck.groupby("problem")["choice"].transform("first")
    stuck = stuck.assign(choice=first_arm, reward=0)
    with pytest.raises(sg.UnanalyzableError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sg.analyze_mouse(stuck)
