"""The win-stay/lose-shift null agents: choice rule, simulation, fits and likelihood."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, log_expit

from src.agents import drifting_simulation as sim
from src.agents import null_agents as na
from src.behavior_analysis.drifting_trial_table import COLUMNS, drifting_trial_table

PORTS = ("A1", "B3", "C2")
LAYOUT = sim.MouseLayout(list(PORTS), "B2", {"A1": 4, "B3": 2, "C2": 0}, [400] * 40)
RULES = sim.TaskRules(drift_rate=0.1, block_lengths=np.arange(30, 36))

WSLS = na.WslsParams(a_win=1.2, a_lose=-0.4)
RECENCY = na.WslsRecencyParams(a_win=1.2, a_lose=-0.4, w=1.5, decay=0.9)
MAGNITUDE = na.WslsMagnitudeParams(a_win=0.2, a_lose=-0.5, b=0.5, w=1.0, decay=0.9)


def hand_table(choices, rewards, starts=(), gaps=(), mouse="m", problem=1):
    """A one-problem trial table; trials listed in *starts* open a new session, those in *gaps*
    follow a dropped trial. Choices are indices into ``PORTS``."""
    rows = []
    session = -1
    for t, (choice, reward) in enumerate(zip(choices, rewards, strict=True)):
        if t == 0 or t in starts:
            session, in_session = session + 1, 0
        rows.append(
            dict(
                mouse=mouse, problem=problem, session=f"s{session}", session_index=session,
                trial=t, trial_in_session=in_session, choice=PORTS[choice],
                magnitudes=dict(zip(PORTS, (1, 2, 3), strict=True)), reward=reward,
                initiation="B2", block=None, after_gap=t in gaps,
            )
        )  # fmt: skip
        in_session += 1
    return pd.DataFrame(rows, columns=COLUMNS)


def pair_table(pairs, mouse="m"):
    """One session built from ``(previous reward, stayed)`` pairs, one pair per transition."""
    choice, choices, rewards = 0, [], []
    for reward, stayed in pairs:
        choices.append(choice)
        rewards.append(reward)
        choice = choice if stayed else (choice + 1) % 3
    return hand_table([*choices, choice], [*rewards, 0], mouse=mouse)


def simulated_table(params, seed=0, mouse="m"):
    policy = na.NullPolicy(params)
    sessions = sim.play_problem(policy, LAYOUT, RULES, np.random.default_rng(seed))
    problems = {1: {mouse: sim.simulated_sessions(mouse, 1, LAYOUT, sessions)}}
    return drifting_trial_table(problems)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


def test_stay_logit_ignores_magnitude_for_plain_wsls_and_grades_it_for_the_magnitude_agent():
    magnitudes = np.array([0, 1, 2, 3, 4])
    assert WSLS.stay_logit(magnitudes).tolist() == [-0.4, 1.2, 1.2, 1.2, 1.2]
    assert RECENCY.stay_logit(magnitudes).tolist() == [-0.4, 1.2, 1.2, 1.2, 1.2]
    assert MAGNITUDE.stay_logit(magnitudes) == pytest.approx([-0.5, 0.2, 0.7, 1.2, 1.7])


def test_zero_slope_magnitude_agent_has_the_plain_stay_rule():
    flat = na.WslsMagnitudeParams(a_win=1.2, a_lose=-0.4, b=0.0, w=1.5, decay=0.9)
    magnitudes = np.arange(5)
    assert flat.stay_logit(magnitudes).tolist() == WSLS.stay_logit(magnitudes).tolist()


def test_parameters_are_validated_and_frozen():
    with pytest.raises(ValueError, match="decay"):
        na.WslsRecencyParams(1.0, -1.0, 1.0, decay=1.0)
    with pytest.raises(ValueError, match="finite"):
        na.WslsParams(np.nan, 0.0)
    with pytest.raises(AttributeError):
        WSLS.a_win = 0.0


def test_switch_weight_is_zero_for_plain_wsls():
    assert WSLS.switch_weight == 0.0
    assert RECENCY.switch_weight == 1.5


# ---------------------------------------------------------------------------
# Decisions: segments and history trace
# ---------------------------------------------------------------------------


def test_segments_restart_at_session_starts_and_after_gaps():
    table = hand_table([0, 0, 1, 1, 2, 2], [1, 1, 1, 1, 1, 1], starts=(3,), gaps=(5,))
    decisions = na.decisions(table, decay=0.9)
    assert decisions.start.tolist() == [True, False, False, True, False, True]


def test_trace_is_an_exponentially_decaying_count_of_past_choices():
    # choices 0, 0, 1, 2 with decay 0.5: before trial 3 the trace is [0.75, 1, 0]
    table = hand_table([0, 0, 1, 2], [1, 1, 1, 1])
    decisions = na.decisions(table, decay=0.5)
    # trial 2 follows a choice of arm 0, so the others are (1, 2) with equal (zero) traces
    assert decisions.delta_h[2] == 0.0
    assert decisions.chose_first[2]
    # trial 3 follows arm 1: the others are (0, 2), traces 0.75 and 0
    assert decisions.delta_h[3] == pytest.approx(0.75)
    assert not decisions.chose_first[3]
    assert decisions.stayed.tolist()[1:] == [True, False, False]


def test_trace_is_cleared_at_a_session_start():
    table = hand_table([0, 0, 0, 1, 2], [1] * 5, starts=(3,))
    decisions = na.decisions(table, decay=0.9)
    # trial 4 follows arm 1, the only choice of its session: arms 0 and 2 both have trace 0
    assert decisions.delta_h[4] == 0.0


def test_decisions_need_one_mouse():
    table = pd.concat([hand_table([0, 1], [1, 1]), hand_table([0, 1], [1, 1], mouse="n")])
    with pytest.raises(ValueError, match="one mouse"):
        na.decisions(table, decay=0.9)


# ---------------------------------------------------------------------------
# Choice rule
# ---------------------------------------------------------------------------


def test_choice_probabilities_follow_the_rule_by_hand():
    h = np.array([0.75, 1.0, 0.0])
    p = na.choice_probabilities(RECENCY, previous=1, magnitude=2, h=h)
    stay = expit(1.2)
    assert p == pytest.approx(
        [(1 - stay) * expit(1.5 * 0.75), stay, (1 - stay) * (1 - expit(1.5 * 0.75))]
    )
    lose = na.choice_probabilities(RECENCY, previous=1, magnitude=0, h=h)
    assert lose[1] == pytest.approx(expit(-0.4))


def test_first_choice_of_a_segment_is_uniform():
    p = na.choice_probabilities(RECENCY, previous=None, magnitude=0, h=np.zeros(3))
    assert p == pytest.approx([1 / 3] * 3)


def test_plain_wsls_switches_uniformly_whatever_the_history():
    p = na.choice_probabilities(WSLS, previous=0, magnitude=0, h=np.array([0.0, 5.0, 0.0]))
    assert p[1] == pytest.approx(p[2])


def test_policy_forgets_the_previous_choice_and_trace_at_a_session_start():
    policy = na.NullPolicy(RECENCY)
    policy.start_session()
    policy.update(0, 3)
    policy.update(0, 3)
    policy.start_session()
    assert policy.probabilities() == pytest.approx([1 / 3] * 3)
    policy.update(2, 1)
    assert policy.trace.tolist() == [0.0, 0.0, 1.0]


def test_stay_probability_depends_on_win_or_lose_not_on_magnitude():
    table = simulated_table(WSLS, seed=1)
    d = na.decisions(table, decay=0.9)
    pairs = ~d.start
    for magnitude in (1, 2, 3, 4):
        mine = pairs & (d.prev_magnitude == magnitude)
        assert mine.sum() > 800
        assert d.stayed[mine].mean() == pytest.approx(expit(1.2), abs=0.045)
    lose = pairs & (d.prev_magnitude == 0)
    assert d.stayed[lose].mean() == pytest.approx(expit(-0.4), abs=0.03)


def test_plain_wsls_switch_target_is_uniform():
    d = na.decisions(simulated_table(WSLS, seed=2), decay=0.9)
    switches = ~d.start & ~d.stayed
    assert switches.sum() > 4000
    assert d.chose_first[switches].mean() == pytest.approx(0.5, abs=0.03)


def test_recency_switch_target_follows_the_logistic_of_the_trace_difference():
    d = na.decisions(simulated_table(RECENCY, seed=3), decay=0.9)
    switches = ~d.start & ~d.stayed
    predicted = expit(RECENCY.w * d.delta_h[switches])
    assert d.chose_first[switches].mean() == pytest.approx(predicted.mean(), abs=0.02)
    positive = d.delta_h[switches] > 1
    assert positive.sum() > 200
    assert d.chose_first[switches][positive].mean() == pytest.approx(
        predicted[positive].mean(), abs=0.05
    )


def test_magnitude_agent_stay_grows_with_its_own_magnitude_only():
    table = simulated_table(MAGNITUDE, seed=4)
    d = na.decisions(table, decay=0.9)
    pairs = ~d.start
    for magnitude in (1, 2, 3, 4):
        mine = pairs & (d.prev_magnitude == magnitude)
        expected = expit(0.2 + 0.5 * (magnitude - 1))
        assert d.stayed[mine].mean() == pytest.approx(expected, abs=0.05)
    # at a fixed own magnitude, the best other arm's magnitude does not matter
    others = np.array(
        [
            max(v for port, v in m.items() if port != c)
            for m, c in zip(table["magnitudes"], table["choice"], strict=True)
        ]
    )
    # rows follow the table order (one problem, in trial order)
    mine = pairs & (d.prev_magnitude == 2)
    rival = np.roll(others, 1)  # best other arm's magnitude on the previous trial
    high, low = mine & (rival > 2), mine & (rival <= 2)
    assert high.sum() > 200 and low.sum() > 200
    assert d.stayed[high].mean() == pytest.approx(d.stayed[low].mean(), abs=0.08)


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------

TOY_CHOICES = [0, 0, 1, 1, 2]
TOY_REWARDS = [2, 0, 3, 1, 0]


def test_log_likelihood_of_a_five_trial_toy_by_hand():
    table = hand_table(TOY_CHOICES, TOY_REWARDS)
    params = na.WslsParams(a_win=1.0, a_lose=-1.0)
    # trial 1 follows reward 2 (win) -> stay; trial 2 follows reward 0 (lose) -> switch;
    # trial 3 follows reward 3 (win) -> stay; trial 4 follows reward 1 (win) -> switch
    expected = (
        np.log(1 / 3)
        + log_expit(1.0)
        + (log_expit(1.0) + np.log(0.5))
        + log_expit(1.0)
        + (log_expit(-1.0) + np.log(0.5))
    )
    assert na.log_likelihood(table, params) == pytest.approx(expected)


def test_recency_and_magnitude_likelihoods_by_hand():
    table = hand_table(TOY_CHOICES, TOY_REWARDS)
    decay, w = 0.9, 0.8
    # trial 2 switches from arm 0 to arm 1 with equal traces on arms 1 and 2;
    # trial 4 switches from arm 1 to arm 2; traces [1.539, 1.9, 0]: first other (arm 0) leads
    delta = 1.539
    recency = na.WslsRecencyParams(a_win=1.0, a_lose=-1.0, w=w, decay=decay)
    expected = (
        np.log(1 / 3)
        + log_expit(1.0)
        + (log_expit(1.0) + log_expit(0.0))
        + log_expit(1.0)
        + (log_expit(-1.0) + log_expit(-w * delta))
    )
    assert na.log_likelihood(table, recency) == pytest.approx(expected)
    graded = na.WslsMagnitudeParams(a_win=1.0, a_lose=-1.0, b=0.3, w=w, decay=decay)
    expected = (
        np.log(1 / 3)
        + log_expit(1.0 + 0.3 * 1)  # stays after 2 pulses
        + (log_expit(1.0) + log_expit(0.0))  # leaves after 0 pulses
        + log_expit(1.0 + 0.3 * 2)  # stays after 3 pulses
        + (log_expit(-1.0) + log_expit(-w * delta))  # leaves after 1 pulse
    )
    assert na.log_likelihood(table, graded) == pytest.approx(expected)


def test_likelihood_matches_the_summed_choice_probabilities_of_the_policy():
    table = simulated_table(MAGNITUDE, seed=5)
    policy = na.NullPolicy(MAGNITUDE)
    table = table.iloc[:600]
    total, previous = 0.0, None
    for row in table.itertuples():
        if row.trial_in_session == 0:
            policy.start_session()
        arm = PORTS.index(row.choice)
        total += np.log(policy.probabilities()[arm])
        policy.update(arm, row.reward)
        previous = arm
    assert previous is not None
    assert na.log_likelihood(table, MAGNITUDE) == pytest.approx(total)
    assert na.nats_per_choice(table, MAGNITUDE) == pytest.approx(-total / len(table))


def test_likelihood_treats_a_gap_like_a_session_start():
    table = hand_table([0, 0, 1, 1], [1, 1, 1, 1], gaps=(2,))
    params = na.WslsParams(1.0, -1.0)
    expected = 2 * np.log(1 / 3) + 2 * log_expit(1.0)
    assert na.log_likelihood(table, params) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------


def test_win_and_lose_logits_are_the_logits_of_the_stay_fractions():
    pairs = [(2, True)] * 30 + [(1, False)] * 10 + [(0, True)] * 15 + [(0, False)] * 25
    # shuffle-free on purpose: pairs are independent of order for the fractions
    fit = na.fit_wsls(pair_table(pairs))["m"]
    assert fit.a_win == pytest.approx(np.log(30 / 10))
    assert fit.a_lose == pytest.approx(np.log(15 / 25))


def test_fractions_of_0_or_1_are_clipped():
    pairs = [(1, True)] * 30 + [(0, False)] * 30
    fit = na.fit_wsls(pair_table(pairs))["m"]
    edge = np.log((1 - na.CLIP) / na.CLIP)
    assert fit.a_win == pytest.approx(edge)
    assert fit.a_lose == pytest.approx(-edge)


def test_fits_are_per_mouse():
    both = pd.concat([pair_table([(1, True)] * 20 + [(0, False)] * 20 + [(1, False)] * 20 +
                                 [(0, True)] * 20),
                      pair_table([(1, True)] * 40 + [(0, False)] * 20 + [(0, True)] * 20,
                                 mouse="n")])  # fmt: skip
    fits = na.fit_wsls(both)
    assert sorted(fits) == ["m", "n"]
    assert fits["m"].a_win == pytest.approx(0.0)
    assert fits["n"].a_win > fits["m"].a_win


def test_wsls_fit_recovers_the_parameters_it_was_simulated_with():
    fit = na.fit_wsls(simulated_table(WSLS, seed=6))["m"]
    assert fit.a_win == pytest.approx(1.2, abs=0.1)
    assert fit.a_lose == pytest.approx(-0.4, abs=0.1)


def test_recency_fit_recovers_the_switch_weight():
    fit = na.fit_wsls_recency(simulated_table(RECENCY, seed=7), decay=0.9)["m"]
    assert fit.a_win == pytest.approx(1.2, abs=0.1)
    assert fit.a_lose == pytest.approx(-0.4, abs=0.1)
    assert fit.w == pytest.approx(1.5, abs=0.2)
    assert fit.decay == 0.9


def test_plain_wsls_data_give_a_recency_weight_near_zero():
    fit = na.fit_wsls_recency(simulated_table(WSLS, seed=8))["m"]
    assert fit.w == pytest.approx(0.0, abs=0.2)


def test_magnitude_fit_recovers_win_intercept_slope_and_switch_weight():
    fit = na.fit_wsls_magnitude(simulated_table(MAGNITUDE, seed=9), decay=0.9)["m"]
    assert fit.a_win == pytest.approx(0.2, abs=0.15)
    assert fit.b == pytest.approx(0.5, abs=0.1)
    assert fit.a_lose == pytest.approx(-0.5, abs=0.1)
    assert fit.w == pytest.approx(1.0, abs=0.2)


def test_plain_wsls_data_give_a_magnitude_slope_near_zero():
    fit = na.fit_wsls_magnitude(simulated_table(WSLS, seed=10))["m"]
    assert fit.b == pytest.approx(0.0, abs=0.1)


def test_magnitude_fit_with_two_levels_is_saturated():
    # two rewarded magnitudes fit exactly: a_win = logit(stay | 1), a_win + b = logit(stay | 2)
    magnitudes = np.array([1] * 30 + [2] * 30)
    stayed = np.array([True] * 15 + [False] * 15 + [True] * 24 + [False] * 6)
    a_win, b = na.fit_win_curve(magnitudes, stayed)
    assert a_win == pytest.approx(0.0, abs=1e-3)
    assert b == pytest.approx(np.log(4), abs=1e-3)


def test_fit_dispatch_by_kind():
    table = pair_table([(1, True)] * 20 + [(1, False)] * 20 + [(0, True)] * 20 + [(0, False)] * 20)
    assert isinstance(na.fit_null_agents(table, "wsls")["m"], na.WslsParams)
    with pytest.raises(ValueError, match="unknown agent"):
        na.fit_null_agents(table, "other")


def test_fits_fail_loudly_with_too_few_pairs():
    few = pair_table([(1, True)] * 5 + [(0, False)] * 40)
    with pytest.raises(ValueError, match="rewarded pairs"):
        na.fit_wsls(few)
    few = pair_table([(1, True)] * 40 + [(0, False)] * 5)
    with pytest.raises(ValueError, match="unrewarded pairs"):
        na.fit_wsls(few)
    with pytest.raises(ValueError, match="mouse m"):
        na.fit_wsls(hand_table([0, 1], [1, 1]))


def test_recency_fit_needs_enough_switches():
    always_stay = pair_table([(1, True)] * 40 + [(0, True)] * 40)
    with pytest.raises(ValueError, match="switches"):
        na.fit_wsls_recency(always_stay)


def test_magnitude_fit_needs_two_distinct_rewarded_magnitudes():
    pairs = [(2, True)] * 20 + [(2, False)] * 20 + [(0, True)] * 20 + [(0, False)] * 20
    with pytest.raises(ValueError, match="distinct"):
        na.fit_wsls_magnitude(pair_table(pairs), min_switches=1)
    with pytest.raises(ValueError, match="distinct"):
        na.fit_win_curve(np.array([3, 3, 3]), np.array([True, False, True]))


def test_perfect_separation_fails_instead_of_returning_a_runaway_weight():
    # after a long run on arm 0 the agent alternates between arms 0 and 1 and never visits 2:
    # every informative switch goes to the other arm with the larger trace, so w diverges
    choices = [0] * 26 + [1, 0] * 20
    rewards = [1] * 25 + [0] * (len(choices) - 25)
    with pytest.raises(RuntimeError, match="bound"):
        na.fit_wsls_recency(hand_table(choices, rewards), min_switches=1)
    with pytest.raises(RuntimeError, match="bound"):
        na.fit_win_curve(np.array([1, 1, 1, 2, 2, 2]), np.array([0, 0, 0, 1, 1, 1], dtype=bool))
