"""Tests for the stay/switch psychometrics: P(stay) against the previously chosen arm's value
and its best alternative.

Hand-built sessions with known values and choices, so every trial pair and its outcome is known.
Arms are A, B, C; a missed trial has the choice "".
"""

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis.stay_switch_psychometrics import (
    CHOSEN_NONZERO,
    CHOSEN_ROWS,
    CHOSEN_ZERO,
    OUTCOMES,
    TIE_ALL_THREE,
    TIE_ALTERNATIVES,
    TIE_CHOSEN_ALTERNATIVE,
    X_AXES,
    split_by_chosen_value,
    stay_heatmap,
    stay_switch_curves,
    stay_switch_curves_by_row,
    stay_switch_pairs,
)


def mag(a, b, c):
    return {"A": a, "B": b, "C": c}


def session(magnitudes, choices):
    """One session dict with one recording."""
    assert len(magnitudes) == len(choices)
    return {
        "trial_variables": [
            {
                "reward_magnitudes": list(magnitudes),
                "chosen_rank": ["best"] * len(choices),
                "choice": list(choices),
            }
        ]
    }


def key(n):
    return f"ses-{n}_date-2026010{n}"


def pairs_of(problems, **kwargs):
    kwargs.setdefault("use_believed", False)
    kwargs.setdefault("tie_mode", "split")
    return stay_switch_pairs(problems, **kwargs)


COLUMNS = ["v_chosen", "v_alt", "v_other", "stay", "switch_best_alt", "switch_other"]


# ------------------------------------------------------------------ trial pairs, true values


def test_trial_pairs_have_the_previous_arm_its_alternatives_and_the_outcome():
    values = [mag(4, 2, 1)] * 7
    choices = ["A", "A", "B", "C", "", "A", "A"]
    result = pairs_of({1: {"m": {key(1): session(values, choices)}}})
    rows = result.pairs[COLUMNS].to_numpy().tolist()
    assert rows == [
        [4, 2, 1, 1, 0, 0],  # t1: A after A, stay
        [4, 2, 1, 0, 1, 0],  # t2: B after A, B (2) is the best alternative
        [2, 4, 1, 0, 0, 1],  # t3: C after B, A (4) is the best alternative, C the other
        [4, 2, 1, 1, 0, 0],  # t6: A after A (t4 and t5 have a missed trial in the pair)
    ]
    assert result.pairs["trial"].tolist() == [1, 2, 3, 6]
    assert set(result.pairs["mouse"]) == {"m"}
    assert result.dropped["m"] == {"session_start": 1, "no_choice": 2}


def test_values_are_those_of_the_current_trial_also_across_a_block_change():
    # the values change at trial 2 (a block change or a drift): the pair 1 -> 2 is kept and
    # uses the new values
    values = [mag(4, 2, 1), mag(4, 2, 1), mag(1, 2, 4)]
    result = pairs_of({1: {"m": {key(1): session(values, ["A", "A", "A"])}}})
    assert result.pairs[COLUMNS].to_numpy().tolist() == [
        [4, 2, 1, 1, 0, 0],
        [1, 4, 2, 1, 0, 0],
    ]


def test_pairs_never_span_sessions_but_do_span_the_recordings_of_one_session():
    values = [mag(4, 2, 1)] * 2
    sessions = {key(1): session(values, ["A", "A"]), key(2): session(values, ["A", "B"])}
    two_recordings = {
        "trial_variables": [
            {"reward_magnitudes": values, "chosen_rank": ["best"] * 2, "choice": ["A", "A"]},
            {"reward_magnitudes": values, "chosen_rank": ["best"] * 2, "choice": ["B", "B"]},
        ]
    }
    sessions[key(3)] = two_recordings
    result = pairs_of({1: {"m": sessions}})
    assert result.pairs["session"].tolist() == [key(1), key(2), key(3), key(3), key(3)]
    assert result.pairs["stay"].tolist() == [1, 0, 1, 0, 1]
    assert result.dropped["m"]["session_start"] == 3


def test_mice_and_problems_are_pooled_per_mouse():
    values = [mag(4, 2, 1)] * 3
    problems = {
        1: {"m1": {key(1): session(values, "AAA")}, "m2": {key(1): session(values, "ABA")}},
        2: {"m1": {key(2): session(values, "ABB")}},
    }
    result = pairs_of(problems)
    by_mouse = result.pairs.groupby("mouse")["stay"].apply(list).to_dict()
    assert by_mouse == {"m1": [1, 1, 0, 1], "m2": [0, 0]}
    assert result.pairs.loc[result.pairs["mouse"] == "m1", "problem"].tolist() == [1, 1, 2, 2]


def test_a_choice_outside_the_offer_is_an_error():
    with pytest.raises(ValueError, match="'D'"):
        pairs_of({1: {"m": {key(1): session([mag(4, 2, 1)] * 2, ["A", "D"])}}})


# ------------------------------------------------------------------ ties


def tie_session():
    values = [mag(3, 3, 1), mag(3, 3, 1), mag(3, 3, 1), mag(2, 2, 2), mag(4, 1, 1)]
    # t1: C -> A, the alternatives A and B tie at 3; t2: A -> A, chosen 3 == alternative 3;
    # t3: A -> B, all three tie; t4: B -> B, chosen B (1) ties only the other arm C (1), below
    # the alternative A (4), so it is no tie
    return {1: {"m": {key(1): session(values, ["C", "A", "A", "B", "B"])}}}


def test_split_mode_keeps_ties_and_splits_a_switch_between_tied_alternatives():
    result = pairs_of(tie_session(), tie_mode="split")
    p = result.pairs
    assert p[COLUMNS].to_numpy().tolist() == [
        [1, 3, 3, 0, 0.5, 0.5],
        [3, 3, 1, 1, 0, 0],
        [2, 2, 2, 0, 0.5, 0.5],
        [1, 4, 1, 1, 0, 0],
    ]
    assert p["tie"].tolist() == [TIE_ALTERNATIVES, TIE_CHOSEN_ALTERNATIVE, TIE_ALL_THREE, None]
    assert result.ties == {TIE_ALTERNATIVES: 1, TIE_CHOSEN_ALTERNATIVE: 1, TIE_ALL_THREE: 1}


def test_exclude_mode_drops_every_tied_pair_and_counts_it():
    result = pairs_of(tie_session(), tie_mode="exclude")
    assert result.pairs[COLUMNS].to_numpy().tolist() == [[1, 4, 1, 1, 0, 0]]
    assert result.ties == {TIE_ALTERNATIVES: 1, TIE_CHOSEN_ALTERNATIVE: 1, TIE_ALL_THREE: 1}
    assert result.dropped["m"]["tie_excluded"] == 3


def test_a_tie_between_the_chosen_arm_and_the_other_arm_is_not_a_tie():
    # chosen B (1) and other C (1) tie below the alternative A (4): the outcome is clear
    result = pairs_of(tie_session(), tie_mode="exclude")
    assert result.pairs["tie"].isna().all()


def test_unknown_tie_mode_raises():
    with pytest.raises(ValueError, match="tie_mode"):
        pairs_of(tie_session(), tie_mode="average")


# ------------------------------------------------------------------ believed values


def test_believed_values_are_the_last_seen_magnitudes_before_the_choice():
    # A pays 4 at t0, B 2 at t1, C 1 at t2; from t3 on every arm is seen. At t4 the believed
    # value of A is its t3 magnitude 5, though the true value is 0 by then.
    values = [mag(4, 2, 1), mag(4, 2, 1), mag(4, 2, 1), mag(5, 2, 1), mag(0, 2, 1), mag(0, 2, 1)]
    result = pairs_of({1: {"m": {key(1): session(values, "ABCAAB")}}}, use_believed=True)
    assert result.pairs["trial"].tolist() == [3, 4, 5]
    assert result.pairs[COLUMNS].to_numpy().tolist() == [
        [1, 4, 2, 0, 1, 0],  # t3: C before; believed C 1, A 4, B 2; chose A
        [5, 2, 1, 1, 0, 0],  # t4: A before, believed 5 (t3's outcome); stays
        [0, 2, 1, 0, 1, 0],  # t5: A before, believed 0 (t4's outcome); switches to B
    ]
    assert result.dropped["m"] == {"session_start": 1, "fewer_than_three_arms_seen": 2}


def test_believed_values_carry_over_sessions_of_a_problem_but_not_problems():
    values = [mag(4, 2, 1)] * 3
    problems = {
        1: {"m": {key(1): session(values, "ABC"), key(2): session(values, "AA" + "A")}},
        2: {"m": {key(3): session(values, "ABC")}},
    }
    result = pairs_of(problems, use_believed=True)
    assert result.pairs["session"].tolist() == [key(2), key(2)]


# ------------------------------------------------------------------ curves


def weighting_pairs():
    """m1: four pairs at diff 2: cell (4, 2) three stays, cell (3, 1) one switch to the best
    alternative. m2: one stay at (4, 2) and one switch to the other arm at (2, 4)."""
    rows = [
        ("m1", 4, 2, 1, 1, 0, 0),
        ("m1", 4, 2, 1, 1, 0, 0),
        ("m1", 4, 2, 1, 1, 0, 0),
        ("m1", 3, 1, 0, 0, 1, 0),
        ("m2", 4, 2, 1, 1, 0, 0),
        ("m2", 2, 4, 1, 0, 0, 1),
    ]
    return pd.DataFrame(rows, columns=["mouse", *COLUMNS])


def test_curves_cover_every_outcome_and_axis():
    curves = stay_switch_curves(weighting_pairs(), pool_trials=True)
    assert set(curves) == set(OUTCOMES) == {"stay", "switch", "switch_best_alt", "switch_other"}
    assert all(set(curves[o]) == set(X_AXES) == {"diff", "chosen", "alt"} for o in OUTCOMES)


def test_pooled_curve_against_the_difference():
    stay = stay_switch_curves(weighting_pairs(), pool_trials=True)["stay"]["diff"]
    assert stay.x.tolist() == [-2, 2]
    assert stay.mice == ["m1", "m2"]
    np.testing.assert_allclose(stay.per_mouse, [[np.nan, 0.75], [0.0, 1.0]])
    np.testing.assert_allclose(stay.mean, [0.0, 0.875])
    assert np.isnan(stay.se[0])
    np.testing.assert_allclose(stay.se[1], np.std([0.75, 1.0], ddof=1) / np.sqrt(2))
    assert stay.n_mice.tolist() == [1, 2] and stay.n_trials.tolist() == [1, 5]


def test_cell_averaged_curve_weights_cells_equally():
    stay = stay_switch_curves(weighting_pairs(), pool_trials=False)["stay"]["diff"]
    np.testing.assert_allclose(stay.per_mouse[0], [np.nan, 0.5])  # cells (4, 2): 1, (3, 1): 0


def test_switch_is_one_minus_stay_and_splits_into_its_two_targets():
    curves = stay_switch_curves(weighting_pairs(), pool_trials=True)
    for axis in X_AXES:
        stay, switch = curves["stay"][axis].per_mouse, curves["switch"][axis].per_mouse
        np.testing.assert_allclose(switch, 1 - stay)
        parts = curves["switch_best_alt"][axis].per_mouse + curves["switch_other"][axis].per_mouse
        np.testing.assert_allclose(parts, switch)
    best_alt = curves["switch_best_alt"]["diff"]
    np.testing.assert_allclose(best_alt.per_mouse, [[np.nan, 0.25], [0.0, 0.0]])


def test_curves_against_the_chosen_and_the_alternative_value():
    curves = stay_switch_curves(weighting_pairs(), pool_trials=True)
    chosen, alt = curves["stay"]["chosen"], curves["stay"]["alt"]
    assert chosen.x.tolist() == [2, 3, 4] and alt.x.tolist() == [1, 2, 4]
    np.testing.assert_allclose(chosen.per_mouse, [[np.nan, 0, 1], [0, np.nan, 1]])
    np.testing.assert_allclose(alt.per_mouse, [[0, 1, np.nan], [np.nan, 1, 0]])


def test_curves_of_no_pairs_raise():
    with pytest.raises(ValueError, match="no trial pairs"):
        stay_switch_curves(weighting_pairs().iloc[:0], pool_trials=True)


# ------------------------------------------------------------------ heatmap


def test_heatmap_is_the_mean_across_mice_with_summed_counts():
    tables = stay_heatmap(weighting_pairs())
    assert tables.p_stay.index.name == "v_chosen" and tables.p_stay.columns.name == "v_alt"
    assert tables.p_stay.loc[4, 2] == 1.0 and tables.n_pairs.loc[4, 2] == 4
    assert tables.p_stay.loc[3, 1] == 0.0 and tables.n_pairs.loc[3, 1] == 1
    assert tables.p_stay.loc[2, 4] == 0.0
    assert np.isnan(tables.p_stay.loc[2, 1]) and tables.n_pairs.loc[2, 1] == 0
    assert tables.per_mouse_p_stay["m1"].loc[4, 2] == 1.0
    assert tables.n_mice == 2


def test_heatmap_mean_weights_mice_not_pairs():
    pairs = weighting_pairs()
    pairs.loc[pairs.index[-1], ["v_chosen", "v_alt"]] = [4, 2]  # m2: one stay, one switch at (4,2)
    tables = stay_heatmap(pairs)
    assert tables.p_stay.loc[4, 2] == pytest.approx((1.0 + 0.5) / 2)


def test_a_trial_without_three_finite_magnitudes_has_no_pair():
    values = [mag(4, 2, 1), {}, mag(4, 2, np.nan), mag(4, 2, 1)]
    result = pairs_of({1: {"m": {key(1): session(values, "AAAA")}}})
    assert result.pairs["trial"].tolist() == [3]
    assert result.dropped["m"]["missing_values"] == 2


# ------------------------------------------------------------------ rows by the chosen value


def test_rows_split_pairs_by_whether_the_chosen_value_is_zero():
    values = [mag(4, 2, 0)] * 4
    result = pairs_of({1: {"m": {key(1): session(values, "CCAC")}}})
    rows = split_by_chosen_value(result.pairs)
    assert tuple(rows) == CHOSEN_ROWS == ("chosen_zero", "chosen_nonzero")
    assert rows[CHOSEN_ZERO]["trial"].tolist() == [1, 2]  # C (0) before
    assert rows[CHOSEN_NONZERO]["trial"].tolist() == [3]  # A (4) before
    assert (rows[CHOSEN_ZERO]["v_chosen"] == 0).all()


def test_believed_rows_use_the_believed_zero_not_the_true_one():
    # C pays 0 at t2; at t3 its true value is 3, but the mouse last saw it pay 0
    values = [mag(4, 2, 0)] * 3 + [mag(4, 2, 3)] * 2
    problems = {1: {"m": {key(1): session(values, "ABCCC")}}}
    believed = split_by_chosen_value(pairs_of(problems, use_believed=True).pairs)
    assert believed[CHOSEN_ZERO]["trial"].tolist() == [3]
    assert believed[CHOSEN_NONZERO]["trial"].tolist() == [4]  # C paid 3 at t3
    true = split_by_chosen_value(pairs_of(problems).pairs)
    assert true[CHOSEN_ZERO].empty  # by its true magnitude C is 3 at t3 and t4
    assert true[CHOSEN_NONZERO]["trial"].tolist() == [1, 2, 3, 4]


def test_a_negative_chosen_value_is_an_error():
    pairs = weighting_pairs()
    pairs.loc[0, "v_chosen"] = -1
    with pytest.raises(ValueError, match="negative"):
        split_by_chosen_value(pairs)


def row_pairs():
    """m1 and m2 have pairs with V(chosen) = 0 and above; m3 only above."""
    rows = [
        ("m1", 0, 4, 2, 0, 1, 0),
        ("m1", 0, 4, 2, 1, 0, 0),
        ("m1", 4, 2, 0, 1, 0, 0),
        ("m2", 0, 2, 1, 0, 0, 1),
        ("m2", 3, 2, 0, 0, 1, 0),
        ("m3", 2, 4, 0, 0, 0.5, 0.5),
        ("m3", 4, 3, 2, 1, 0, 0),
    ]
    return pd.DataFrame(rows, columns=["mouse", *COLUMNS])


def test_curves_by_row_count_only_the_mice_with_pairs_in_that_row():
    rows = stay_switch_curves_by_row(row_pairs(), pool_trials=True)
    zero, nonzero = rows[CHOSEN_ZERO], rows[CHOSEN_NONZERO]
    assert zero["stay"]["diff"].mice == ["m1", "m2"]
    assert nonzero["stay"]["diff"].mice == ["m1", "m2", "m3"]
    assert zero["stay"]["chosen"].x.tolist() == [0]
    np.testing.assert_allclose(zero["stay"]["chosen"].per_mouse, [[0.5], [0.0]])
    np.testing.assert_allclose(zero["stay"]["chosen"].mean, [0.25])


def test_switch_targets_add_up_to_switch_within_each_row():
    for pool_trials in (True, False):
        rows = stay_switch_curves_by_row(row_pairs(), pool_trials=pool_trials)
        for curves in rows.values():
            for axis in X_AXES:
                parts = (
                    curves["switch_best_alt"][axis].per_mouse
                    + curves["switch_other"][axis].per_mouse
                )
                np.testing.assert_allclose(parts, curves["switch"][axis].per_mouse)
                np.testing.assert_allclose(
                    curves["switch"][axis].per_mouse, 1 - curves["stay"][axis].per_mouse
                )


def test_a_row_without_pairs_has_no_curves():
    pairs = row_pairs()
    rows = stay_switch_curves_by_row(pairs[pairs["v_chosen"] > 0], pool_trials=True)
    assert rows[CHOSEN_ZERO] is None
    assert rows[CHOSEN_NONZERO]["stay"]["diff"].mice == ["m1", "m2", "m3"]
