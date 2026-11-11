"""WSLS (Graded): stay logit linear in the magnitude, switches to the most recent alternative."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from src.agents import null_agents as na
from src.behavior_analysis import switch_regression as sr
from src.pipelines import null_agents as nulls
from src.pipelines import switch_regression as switch_pipe
from tests.test_null_agents import RULES, hand_table, simulated_table

GRADED = na.WslsGradedParams(w0=-0.6, w_m=0.5, p_recent=0.8)
ZERO_INTERCEPT = na.WslsGradedParams(w0=0.0, w_m=0.5, p_recent=0.8)

# choices as arm indices (sorted ports A1, B3, C2 = 0, 1, 2)
CHOICES = [0, 1, 2, 0, 2, 1]


# ---------------------------------------------------------------------------
# Parameters and choice rule
# ---------------------------------------------------------------------------


def test_the_stay_logit_is_linear_in_the_magnitude_with_the_intercept_after_zero():
    magnitudes = np.array([0, 1, 2, 3, 4])
    assert GRADED.stay_logit(magnitudes) == pytest.approx([-0.6, -0.1, 0.4, 0.9, 1.4])
    assert ZERO_INTERCEPT.stay_logit(magnitudes)[0] == 0.0  # P(stay) = 0.5 after zero


def test_graded_is_a_registered_kind_after_the_other_three():
    assert na.KINDS[:4] == ("wsls", "wsls_recency", "wsls_magnitude", "wsls_graded")
    assert na.AGENT_PARAMS["wsls_graded"] is na.WslsGradedParams


def test_parameters_are_validated_and_frozen():
    with pytest.raises(ValueError, match="p_recent"):
        na.WslsGradedParams(0.0, 0.5, 1.2)
    with pytest.raises(ValueError, match="finite"):
        na.WslsGradedParams(np.nan, 0.5, 0.5)
    with pytest.raises(AttributeError):
        GRADED.w_m = 0.0


def test_choice_probabilities_follow_the_rule_by_hand():
    h = np.zeros(3)
    # previous arm 0 paid 2 pulses: stay = sigmoid(-0.6 + 1.0); alternatives are arms 1 and 2
    stay = expit(0.4)
    p = na.choice_probabilities(GRADED, 0, 2.0, h, recent=1)  # arm 1 was chosen more recently
    assert p == pytest.approx([stay, (1 - stay) * 0.8, (1 - stay) * 0.2])
    p = na.choice_probabilities(GRADED, 0, 2.0, h, recent=-1)  # arm 2 more recently
    assert p == pytest.approx([stay, (1 - stay) * 0.2, (1 - stay) * 0.8])
    p = na.choice_probabilities(GRADED, 0, 2.0, h, recent=0)  # no history or a tie: 50/50
    assert p == pytest.approx([stay, (1 - stay) / 2, (1 - stay) / 2])
    assert na.choice_probabilities(GRADED, None, 0.0, h) == pytest.approx([1 / 3] * 3)


# ---------------------------------------------------------------------------
# The most recent alternative: bookkeeping
# ---------------------------------------------------------------------------


def _recent(table):
    d = na.decisions(table, na.DEFAULT_DECAY)
    return d.recent.tolist()


def test_the_recent_alternative_by_hand_on_a_sequence():
    table = hand_table(CHOICES, [1] * len(CHOICES))
    # t=1: neither alternative chosen yet (tie); t=2: arm 0 only; t=3: arm 1 after arm 0 -> second;
    # t=4: arm 2 after arm 1 -> second; t=5: arm 0 (t=3) over arm 1 (t=1) -> first
    assert _recent(table) == [0, 0, 1, -1, -1, 1]


def test_the_memory_runs_on_across_sessions_within_a_problem():
    table = hand_table(CHOICES, [1] * len(CHOICES), starts=(3,))
    # t=3 opens a session (no decision); arms 1 and 2 were chosen at t=1 and t=2 before it
    assert _recent(table) == [0, 0, 1, 0, -1, 1]


def test_the_memory_resets_at_a_problem_start():
    first = hand_table([0, 1, 2], [1, 1, 1], problem=1)
    second = hand_table([1, 0, 2], [1, 1, 1], problem=2)
    # problem 2, t=2: previous arm 0, alternatives 1 (chosen at its t=0) and 2 (never in problem
    # 2, though chosen in problem 1): arm 1 is the recent one
    assert _recent(pd.concat([first, second], ignore_index=True)) == [0, 0, 1, 0, 0, 1]


def test_the_definition_matches_the_switch_regressions_trials_since_chosen():
    rng = np.random.default_rng(0)
    chosen = rng.integers(0, 3, 300)
    table = hand_table(chosen.tolist(), [1] * 300)
    since = sr.trials_since_chosen(chosen, np.arange(300) == 0)
    expected = []
    for t in range(1, 300):
        first, second = na.OTHERS[chosen[t - 1]]
        a, b = since[t, first], since[t, second]
        expected.append(0 if np.isnan(a) and np.isnan(b) else
                        1 if np.isnan(b) or a < b else -1)  # fmt: skip
    assert _recent(table)[1:] == expected  # defined at every pair, switch or stay


def test_the_policy_remembers_across_sessions_but_not_across_problems():
    policy = na.NullPolicy(GRADED)
    for arm in (1, 2):
        policy.update(arm, 1.0)
    policy.start_session()
    policy.update(0, 1.0)  # previous arm 0; alternatives 1 (t=0) and 2 (t=1): arm 2 recent
    p = policy.probabilities()
    stay = expit(-0.6 + 0.5)
    assert p == pytest.approx([stay, (1 - stay) * 0.2, (1 - stay) * 0.8])
    fresh = na.NullPolicy(GRADED)  # a new problem: no memory
    fresh.update(0, 1.0)
    assert fresh.probabilities()[1] == pytest.approx((1 - stay) / 2)


# ---------------------------------------------------------------------------
# Simulation follows the rule (large samples)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def played():
    table = simulated_table(GRADED, seed=3)
    return table, na.decisions(table, na.DEFAULT_DECAY)


def test_simulated_staying_follows_the_logistic_of_the_magnitude(played):
    _, d = played
    pairs = ~d.start
    for m in (0, 1, 2, 3):
        mask = pairs & (d.prev_magnitude == m)
        if mask.sum() > 300:
            assert d.stayed[mask].mean() == pytest.approx(expit(GRADED.w0 + GRADED.w_m * m),
                                                          abs=0.05)  # fmt: skip


def test_simulated_switches_go_to_the_recent_alternative_with_p_recent(played):
    _, d = played
    informative = d.switched & (d.recent != 0)
    to_recent = (d.chose_first == (d.recent == 1))[informative].mean()
    assert informative.sum() > 500 and to_recent == pytest.approx(0.8, abs=0.04)
    tied = d.switched & (d.recent == 0)
    if tied.sum() > 100:
        assert d.chose_first[tied].mean() == pytest.approx(0.5, abs=0.1)


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------


def test_the_log_likelihood_of_a_toy_by_hand():
    rewards = [2, 0, 1, 3, 0, 1]
    table = hand_table(CHOICES, rewards)
    # t=0 start: log(1/3). Pair t: previous reward r[t-1], stay logit z = -0.6 + 0.5 r[t-1]
    # every pair is a switch: log(1 - sigmoid(z)) + log(p) with p = 0.8 to the recent alternative
    # (recent: t=1 tie 0.5, t=2 first and chose second, t=3 second and chose first,
    #  t=4 second and chose second, t=5 first and chose second)
    z = [-0.6 + 0.5 * r for r in rewards[:-1]]
    p_to = [0.5, 0.2, 0.2, 0.8, 0.2]
    expected = np.log(1 / 3) + sum(
        np.log(1 - expit(zi)) + np.log(p) for zi, p in zip(z, p_to, strict=True)
    )
    assert na.log_likelihood(table, GRADED) == pytest.approx(expected)


def test_a_stay_uses_only_the_stay_probability():
    table = hand_table([0, 0, 0], [3, 1, 0])
    z = [-0.6 + 0.5 * 3, -0.6 + 0.5 * 1]
    expected = np.log(1 / 3) + sum(np.log(expit(zi)) for zi in z)
    assert na.log_likelihood(table, GRADED) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------


def test_the_fit_recovers_the_parameters_it_was_simulated_with(played):
    table, _ = played
    fit = na.fit_null_agents(table, "wsls_graded")["m"]
    assert fit.w0 == pytest.approx(GRADED.w0, abs=0.15)
    assert fit.w_m == pytest.approx(GRADED.w_m, abs=0.1)
    assert fit.p_recent == pytest.approx(GRADED.p_recent, abs=0.04)


def test_the_intercept_can_be_fixed_to_zero():
    table = simulated_table(ZERO_INTERCEPT, seed=4)
    fit = na.fit_null_agents(table, "wsls_graded", fix_intercept=True)["m"]
    assert fit.w0 == 0.0
    assert fit.w_m == pytest.approx(0.5, abs=0.1) and fit.p_recent == pytest.approx(0.8, abs=0.04)
    free = na.fit_null_agents(table, "wsls_graded")["m"]
    assert free.w0 != 0.0  # the default leaves it free


def test_p_recent_is_the_fraction_of_informative_switches_clipped_away_from_0_and_1():
    recent = np.array([1, 1, -1, -1, 0, 1])
    chose_first = np.array([True, False, False, True, True, True])
    # informative: five; to the recent alternative: t0, t2, t5 -> 3/5
    assert na.recent_fraction(recent, chose_first, min_switches=1) == pytest.approx(0.6)
    always = na.recent_fraction(np.ones(40), np.ones(40, bool), min_switches=1)
    assert always == pytest.approx(1 - na.CLIP)
    never = na.recent_fraction(np.ones(40), np.zeros(40, bool), min_switches=1)
    assert never == pytest.approx(na.CLIP)


def test_the_fit_fails_loudly_with_too_few_pairs_or_switches():
    short = hand_table([0, 1, 2, 0, 1], [1, 0, 1, 0, 1])
    with pytest.raises(ValueError, match="mouse m: .*pairs"):
        na.fit_null_agents(short, "wsls_graded")
    with pytest.raises(ValueError, match="informative switches"):
        na.recent_fraction(np.array([1, -1]), np.array([True, False]), min_switches=30)


def test_the_fit_dispatches_by_kind_and_is_per_mouse(played):
    table, _ = played
    other = table.assign(mouse="n")
    fits = na.fit_null_agents(pd.concat([table, other], ignore_index=True), "wsls_graded")
    assert list(fits) == ["m", "n"] and fits["m"] == fits["n"]
    assert isinstance(fits["m"], na.WslsGradedParams)


# ---------------------------------------------------------------------------
# In the pipelines
# ---------------------------------------------------------------------------


def test_the_graded_agent_is_simulated_like_the_others_and_lengthens_with_the_factor(played):
    table, _ = played
    short = nulls.simulate_null_cohort(table, {"m": GRADED}, "wsls_graded", 1, RULES)
    long = nulls.simulate_null_cohort(
        table, {"m": GRADED}, "wsls_graded", 1, RULES, length_factor=2.0
    )
    assert short.kind == "wsls_graded" and len(short.trials) == len(table)
    assert len(long.trials) == 2 * len(table)
    assert nulls.AGENT_LABELS["wsls_graded"] == "WSLS (Graded)"


def test_the_graded_agent_is_the_third_calibration_null_after_the_one_whose_band_is_drawn():
    assert switch_pipe.NULL_AGENTS == ("wsls_magnitude", "wsls_recency", "wsls_graded")
