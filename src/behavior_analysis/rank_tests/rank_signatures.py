"""The five behavioral tests of the ranked value algorithm, run on one mouse (or one replay).

A ranked value algorithm ranks the options (best, second, third) and gates choices on comparisons
between adjacent ranks; a softmax value learner chooses among per-arm values. The tests look for
signatures the second does not produce:

1. **Stay depends on the second best, not the third** (:mod:`rank_stay`): ``w_best < 0`` and
   ``w_worst`` about 0.
2. **Abrupt switching at believed rank crossings** (:mod:`rank_crossings`): a step in leaving at
   ``D = 0`` and fast adoption of the new best.
3. **The third option is categorical** (:mod:`rank_third`): P(choose the worst) is flat in its gap
   to the second best.
4. **Ranks matter more than values** (:mod:`rank_stay`, :mod:`rank_switch`): held-out fits prefer
   rank indicators to values, and staying at a rank is invariant to the payout's magnitude.
5. **Transfer and inference** (:mod:`rank_transfer`): learning across problems, and going to the
   old zero arm at a block change.

:func:`analyze_mouse` returns every scalar readout (to compare with replays of fitted agents) and
every curve (to draw). :data:`PREDICTED` gives, for each readout the tests make a prediction about,
the direction in which a rank algorithm differs from a softmax learner.
"""

from src.behavior_analysis.rank_tests import rank_crossings as rc
from src.behavior_analysis.rank_tests import rank_decisions as rd
from src.behavior_analysis.rank_tests import rank_logit as rl
from src.behavior_analysis.rank_tests import rank_state as rs
from src.behavior_analysis.rank_tests import rank_stay as st
from src.behavior_analysis.rank_tests import rank_switch as rw
from src.behavior_analysis.rank_tests import rank_third as r3
from src.behavior_analysis.rank_tests import rank_transfer as rt
from src.behavior_analysis.rank_tests import rank_worst_split as wsp
from src.behavior_analysis.rank_tests.rank_result import RankResult, merge
from src.behavior_analysis.switching import switch_regression as sr

MIN_DECISIONS = 200  # stay/leave decisions a mouse needs for the stay models

# Test of each readout, and the direction in which a rank algorithm lies above (+1) or below
# (-1) a softmax value learner. Readouts not listed are context (counts, errors, nats).
TESTS = {
    1: ("w_diff", "w_best", "w_worst", "q_w_diff", "q_w_best", "q_w_worst"),
    # the probe of the worst alternative's weight (rank_worst_split): no rank/softmax prediction
    6: tuple(wsp.READOUTS),
    2: ("step_weight", "gain_indicator", "gain_step_vs_smooth", "cross_leave_1", "cross_adopt_1",
        "cross_t50", "true_best_next_1", "true_best_t50", "true_second_next_1",
        "true_second_t50"),
    3: ("worst_slope", "top2_w_worst", "top2_w_gap"),
    4: ("rank_minus_value", "rank_over_value", "value_over_rank", "sw_rank_minus_value",
        "sw_rank_over_value", "sw_value_over_rank", "scale_slope_first", "scale_slope_second"),
    5: ("slope_p50", "slope_p100", "slope_threshold", "half_p100", "block_inherit",
        "block_inherit_below"),
}  # fmt: skip
PREDICTED = {
    "w_diff": +1, "q_w_diff": +1, "w_worst": +1, "q_w_worst": +1, "w_best": -1, "q_w_best": -1,
    "step_weight": +1, "gain_indicator": +1, "gain_step_vs_smooth": +1,
    "cross_leave_1": +1, "cross_adopt_1": +1, "cross_t50": -1,
    "worst_slope": +1,
    "rank_minus_value": +1, "sw_rank_minus_value": +1, "scale_slope_first": -1,
    "slope_p100": +1, "slope_threshold": -1, "half_p100": +1,
    "block_inherit": +1, "block_inherit_below": +1,
}  # fmt: skip


class UnanalyzableError(ValueError):
    """A mouse (or replay) has too little data for a test: decisions, switches or problems."""


def which_test(readout: str) -> int | None:
    """The test (1-5) a readout belongs to, or ``None`` for a context readout."""
    return next((test for test, names in TESTS.items() if readout in names), None)


def analyze_mouse(trials, kq_params=None) -> RankResult:
    """Run every rank test on one mouse's trials.

    Args:
        trials (pd.DataFrame): One mouse's rows of the drifting trial table.
        kq_params (Sequence[float] | None): The mouse's fitted kernel Q-learner parameters; with
            them test 1 is also run on the learner's Q values (readouts prefixed ``q_``).
            Default ``None``.

    Returns:
        RankResult: Every readout of the five tests and their curves.

    Raises:
        UnanalyzableError: If the mouse has fewer than ``MIN_DECISIONS`` decisions, the stay or
            leave outcome is a single class, or the switches come from fewer than two problems.
        ValueError: As :func:`rank_state.mouse_state`.
    """
    state = rs.mouse_state(trials)
    decisions = rd.decision_table(state)
    if len(decisions) < MIN_DECISIONS:
        raise UnanalyzableError(
            f"mouse {state.mouse}: {len(decisions)} decisions with all three arms visited; "
            f"at least {MIN_DECISIONS} are needed"
        )
    try:
        parts = [
            st.stay_weights(decisions),
            wsp.worst_weight_split(wsp.with_ages(state, decisions)),
            st.value_or_rank(decisions),
            st.scale_invariance(decisions),
            rc.crossing_step(decisions),
            rc.alignment_readouts(state),
            r3.worst_choice(state, decisions),
            r3.top_two_choice(state, decisions),
            rw.switch_rank_or_value(trials),
            rt.transfer(state),
            rt.block_inference(state),
        ]
        if kq_params is not None:
            q_decisions = rd.decision_table(state, rd.kernel_q_values(state, kq_params))
            parts.append(st.stay_weights(q_decisions, prefix="q_"))
    except (rl.DegenerateOutcomeError, rl.TooFewGroupsError, sr.TooFewSwitchesError) as error:
        raise UnanalyzableError(f"mouse {state.mouse}: {error}") from error
    return merge(*parts)
