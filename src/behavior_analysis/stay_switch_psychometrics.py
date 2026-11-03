"""Stay/switch psychometrics: P(stay) against the value of the arm chosen last and its alternative.

On every trial t whose previous trial t - 1 is in the same session, the arm chosen on t - 1 is
the *chosen* option. With the values of trial t (true magnitudes, or believed magnitudes: what
each arm paid the last time it was chosen, before t's choice):

- ``v_chosen``: the value of that arm;
- ``v_alt``: the highest value of the two other arms, the next-best alternative;
- ``v_other``: the value of the remaining arm.

The outcome is ``stay`` (t picks the same arm), else a switch, split into ``switch_best_alt``
(to the arm holding ``v_alt``) and ``switch_other``. ``switch = 1 - stay``.

Trial pairs
-----------
- A pair never spans sessions: a session's first trial has no pair (``session_start``). The
  recordings of one session are joined, as everywhere else, so a pair can span a restart.
- A pair is kept across a block change or any value change: the values of t are the ones the
  stay/switch decision is made against, and a change is what makes them differ from t - 1's.
- Both trials need a choice; a missed trial on either side drops the pair (``no_choice``). A
  pair is not built across a missed trial.
- Believed values reset at every problem and carry over its sessions in session order; a pair
  counts only once all three arms have been chosen in the problem
  (``fewer_than_three_arms_seen``).
- A trial t without three finite true magnitudes has no pair (``missing_values``).

Ties
----
Ties are named relative to the chosen arm, mutually exclusive, and counted in either mode:

- ``chosen_alternative``: ``v_chosen == v_alt`` (the chosen arm shares the top value of the
  others), not all three;
- ``alternatives``: the two other arms tie (``v_alt == v_other``), not all three, so a switch has
  no single best alternative;
- ``all_three``: all three values are equal.

A tie of the chosen arm with the remaining arm below ``v_alt`` changes nothing and is no tie.
``tie_mode="exclude"`` drops every pair tied in one of these three ways. This is not the same
set the exclude mode of :mod:`src.behavior_analysis.value_psychometrics` drops: that module also
drops a tie between the chosen arm and the remaining arm below ``v_alt`` (e.g. chosen 2, alt 4,
other 2), which is kept here because the stay/switch outcome and the switch target are
unambiguous. Tie totals in the captions therefore differ from the value-psychometrics totals.
``"split"`` keeps them: a pair with ``v_chosen == v_alt`` sits at difference 0, and a switch
between tied alternatives counts half to ``switch_best_alt`` and half to ``switch_other``, as the
split mode shares tied rank positions evenly.

Mice are the replicates: every mean and standard error is across mice. The three pairs a trial
could form with the three arms are not used; each trial gives at most one pair.
"""

from collections import Counter
from collections.abc import Mapping
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.behavior_analysis.get_believed_value import believed_magnitudes_by_trial
from src.behavior_analysis.get_choice_probs_by_value import (
    N_ARMS,
    TIE_EXCLUDE,
    TIE_MODES,
    session_trial_lists,
)
from src.behavior_analysis.stats import nan_mean, nan_standard_error
from src.behavior_analysis.value_psychometrics import Curve
from src.behavior_import.session_order import ordered_session_keys

TIE_CHOSEN_ALTERNATIVE = "chosen_alternative"
TIE_ALTERNATIVES = "alternatives"
TIE_ALL_THREE = "all_three"
TIE_CATEGORIES = (TIE_CHOSEN_ALTERNATIVE, TIE_ALTERNATIVES, TIE_ALL_THREE)

REASON_SESSION_START = "session_start"
REASON_NO_CHOICE = "no_choice"
REASON_FEWER_THAN_THREE_SEEN = "fewer_than_three_arms_seen"
REASON_MISSING_VALUES = "missing_values"
REASON_TIE_EXCLUDED = "tie_excluded"

OUTCOMES = ("stay", "switch", "switch_best_alt", "switch_other")
# Rows of the curve figures: pairs whose chosen arm had value 0, and the others.
CHOSEN_ZERO = "chosen_zero"
CHOSEN_NONZERO = "chosen_nonzero"
CHOSEN_ROWS = (CHOSEN_ZERO, CHOSEN_NONZERO)
X_AXES = ("diff", "chosen", "alt")

PAIR_COLUMNS = (
    "mouse",
    "problem",
    "session",
    "trial",
    "v_chosen",
    "v_alt",
    "v_other",
    "stay",
    "switch_best_alt",
    "switch_other",
    "tie",
)


class StaySwitchPairs(NamedTuple):
    """Every counted trial pair, and what was not counted.

    Attributes:
        pairs (pd.DataFrame): One row per pair, columns ``PAIR_COLUMNS``: ``trial`` is t's
            index within its session; ``tie`` a ``TIE_*`` category or None.
        dropped (dict[str, Counter]): ``{mouse: {reason: trials}}`` of trials without a pair.
        ties (dict[str, int]): Tied pairs of all mice by category, counted before the tie mode
            drops any.
    """

    pairs: pd.DataFrame
    dropped: dict[str, Counter]
    ties: dict[str, int]


class StayHeatmap(NamedTuple):
    """P(stay) by chosen value (rows) and alternative value (columns).

    Attributes:
        p_stay (pd.DataFrame): Mean across mice of each mouse's P(stay) in the cell (every
            mouse equal, mice without pairs there left out); NaN in a cell with no pairs.
            Index ``v_chosen``, columns ``v_alt``, both the values that occur.
        n_pairs (pd.DataFrame): Pairs summed over mice, same grid.
        per_mouse_p_stay (dict[str, pd.DataFrame]): ``{mouse: grid}``.
        n_mice (int): Mice with at least one pair.
    """

    p_stay: pd.DataFrame
    n_pairs: pd.DataFrame
    per_mouse_p_stay: dict[str, pd.DataFrame]
    n_mice: int


# ---------------------------------------------------------------------------
# Trial pairs
# ---------------------------------------------------------------------------


def _check_tie_mode(tie_mode: str) -> None:
    if tie_mode not in TIE_MODES:
        raise ValueError(f"tie_mode must be one of {TIE_MODES}, got {tie_mode!r}")


def _tie(v_chosen: float, v_alt: float, v_other: float) -> str | None:
    """The tie category of one pair's three values (see the module docstring)."""
    if v_chosen == v_alt == v_other:
        return TIE_ALL_THREE
    if v_chosen == v_alt:
        return TIE_CHOSEN_ALTERNATIVE
    if v_alt == v_other:
        return TIE_ALTERNATIVES
    return None


def _outcome(previous: str, choice: str, values: Mapping) -> dict:
    """Values and outcome of one pair: arm *previous* chosen before, *choice* now.

    Raises:
        ValueError: If either choice is not one of the arms of *values*.
    """
    for arm in (previous, choice):
        if arm not in values:
            raise ValueError(f"choice {arm!r} is not one of the arms {sorted(values)}")
    (alt_arm, v_alt), (other_arm, v_other) = sorted(
        ((arm, value) for arm, value in values.items() if arm != previous),
        key=lambda item: -item[1],
    )
    stay = float(choice == previous)
    switch = 1.0 - stay
    if v_alt == v_other:
        # no single best alternative: the switch is shared evenly between the tied arms
        best_alt = other = switch / 2
    else:
        best_alt, other = float(choice == alt_arm), float(choice == other_arm)
    v_chosen = values[previous]
    return {
        "v_chosen": v_chosen,
        "v_alt": v_alt,
        "v_other": v_other,
        "stay": stay,
        "switch_best_alt": best_alt,
        "switch_other": other,
        "tie": _tie(v_chosen, v_alt, v_other),
    }


def _three_finite(values: Mapping | None) -> bool:
    """Whether *values* holds a finite number for each of three arms."""
    if not values or len(values) != N_ARMS:
        return False
    return all(isinstance(v, (int, float, np.number)) and np.isfinite(v) for v in values.values())


def _drop_reason(trial: int, choices: list, magnitudes: list, values: Mapping) -> str | None:
    """Why trial *trial* of a session has no pair, or None if it has one.

    *values* are the values the pair would use (the true magnitudes or the believed ones). Only
    trial *trial*'s magnitudes are checked: the pair uses trial t's values, so trial t-1 needs a
    choice but not three finite magnitudes.
    """
    if trial == 0:
        return REASON_SESSION_START
    if not choices[trial - 1] or not choices[trial]:
        return REASON_NO_CHOICE
    if not _three_finite(magnitudes[trial]):
        return REASON_MISSING_VALUES
    if not _three_finite(values):
        return REASON_FEWER_THAN_THREE_SEEN
    return None


def _session_pairs(
    magnitudes: list, choices: list, values_by_trial: list, *, where: str
) -> tuple[list[dict], Counter]:
    """The pair rows of one session and the reasons its other trials have none."""
    rows, dropped = [], Counter()
    for trial, values in enumerate(values_by_trial):
        reason = _drop_reason(trial, choices, magnitudes, values)
        if reason is not None:
            dropped[reason] += 1
            continue
        try:
            row = _outcome(choices[trial - 1], choices[trial], values)
        except ValueError as error:
            raise ValueError(f"{where} trial {trial}: {error}") from None
        rows.append({"trial": trial, **row})
    return rows, dropped


def _problem_pairs(
    subject: str, sessions: Mapping, *, use_believed: bool
) -> tuple[list[dict], Counter]:
    """Pair rows (with ``session``) of one mouse in one problem, sessions in session order.

    Believed values run over the whole problem, so they carry over its sessions.
    """
    keys, lists = [], []
    for key in ordered_session_keys(subject, sessions):
        trial_lists = session_trial_lists(sessions[key])
        if trial_lists is not None:
            keys.append(key)
            lists.append((list(trial_lists[0]), list(trial_lists[2])))
    all_magnitudes = [m for magnitudes, _ in lists for m in magnitudes]
    all_choices = [c for _, choices in lists for c in choices]
    if use_believed:
        all_values = believed_magnitudes_by_trial(all_choices, all_magnitudes)
    else:
        all_values = all_magnitudes
    rows, dropped, start = [], Counter(), 0
    for key, (magnitudes, choices) in zip(keys, lists, strict=True):
        values = all_values[start : start + len(magnitudes)]
        start += len(magnitudes)
        session_rows, session_dropped = _session_pairs(
            magnitudes, choices, values, where=f"{subject} {key}"
        )
        rows += [{"session": key, **row} for row in session_rows]
        dropped += session_dropped
    return rows, dropped


def stay_switch_pairs(
    subjects_trials_by_problem: Mapping, *, use_believed: bool, tie_mode: str
) -> StaySwitchPairs:
    """Every trial pair (t - 1, t) of every mouse, with its values and stay/switch outcome.

    Args:
        subjects_trials_by_problem (dict): ``{problem: {subject: {session_key: session}}}``, a
            session as for :func:`src.behavior_analysis.get_choice_probs_by_value.
            count_choice_probs`.
        use_believed (bool): Use believed instead of true values. Keyword-only.
        tie_mode (str): ``"exclude"`` drops tied pairs, ``"split"`` keeps them (see the module
            docstring). Keyword-only.

    Returns:
        StaySwitchPairs: The pairs, the uncounted trials per mouse, and the tie totals.

    Raises:
        ValueError: If *tie_mode* is unknown, a choice is not one of the arms of its trial, or
            a recording's trial lists differ in length.
    """
    _check_tie_mode(tie_mode)
    rows, dropped = [], {}
    for problem, subjects in subjects_trials_by_problem.items():
        for subject, sessions in subjects.items():
            problem_rows, problem_dropped = _problem_pairs(
                subject, sessions, use_believed=use_believed
            )
            rows += [{"mouse": subject, "problem": problem, **row} for row in problem_rows]
            dropped[subject] = dropped.get(subject, Counter()) + problem_dropped
    pairs = pd.DataFrame(rows, columns=list(PAIR_COLUMNS))
    ties = Counter(pairs["tie"].dropna())
    ties = {category: int(ties[category]) for category in TIE_CATEGORIES}
    if tie_mode == TIE_EXCLUDE:
        tied = pairs["tie"].notna()
        for subject, n in pairs.loc[tied, "mouse"].value_counts().items():
            dropped[subject][REASON_TIE_EXCLUDED] += int(n)
        pairs = pairs.loc[~tied].reset_index(drop=True)
    return StaySwitchPairs(pairs, dropped, ties)


# ---------------------------------------------------------------------------
# Curves and heatmap
# ---------------------------------------------------------------------------


def _with_outcomes_and_axes(pairs: pd.DataFrame) -> pd.DataFrame:
    """*pairs* with the ``switch`` outcome and the x values of each axis."""
    return pairs.assign(
        switch=1.0 - pairs["stay"],
        diff=pairs["v_chosen"] - pairs["v_alt"],
        chosen=pairs["v_chosen"],
        alt=pairs["v_alt"],
    )


def _mouse_by_x(frame: pd.DataFrame, outcome: str, axis: str, pool_trials: bool) -> pd.Series:
    """Each mouse's mean *outcome* at each x of *axis*, indexed ``(mouse, x)``.

    Pooled, a mouse's pairs at an x are averaged together; otherwise each (v_chosen, v_alt)
    cell's mean counts once, so a cell with few pairs weighs as much as one with many.
    """
    if pool_trials:
        return frame.groupby(["mouse", axis])[outcome].mean()
    cells = frame.groupby(["mouse", axis, "v_chosen", "v_alt"])[outcome].mean()
    return cells.groupby(level=["mouse", axis]).mean()


def _curve(frame: pd.DataFrame, outcome: str, axis: str, pool_trials: bool) -> Curve:
    by_x = _mouse_by_x(frame, outcome, axis, pool_trials).unstack(axis)
    x = np.sort(frame[axis].unique())
    mice = sorted(frame["mouse"].unique())
    per_mouse = by_x.reindex(index=mice, columns=x).to_numpy(dtype=float)
    n_trials = frame.groupby(axis).size().reindex(x).to_numpy()
    return Curve(
        x=x,
        mice=mice,
        per_mouse=per_mouse,
        mean=nan_mean(per_mouse, axis=0),
        se=nan_standard_error(per_mouse),
        n_mice=np.sum(~np.isnan(per_mouse), axis=0),
        n_trials=n_trials,
    )


def stay_switch_curves(pairs: pd.DataFrame, *, pool_trials: bool) -> dict[str, dict[str, Curve]]:
    """P(stay), P(switch) and the two switch targets against the three x axes.

    The x axes are ``"diff"`` (``v_chosen - v_alt``), ``"chosen"`` and ``"alt"``; their x values
    are the ones that occur in *pairs*. ``Curve.n_trials`` counts trial pairs.

    Args:
        pairs (pd.DataFrame): ``StaySwitchPairs.pairs`` (at least ``mouse``, ``v_chosen``,
            ``v_alt``, ``stay``, ``switch_best_alt``, ``switch_other``).
        pool_trials (bool): Within each mouse, average all its pairs at an x (``True``) or
            average its (v_chosen, v_alt) cells equally (``False``). Mice are the replicates
            for the mean and standard error either way. Keyword-only.

    Returns:
        dict[str, dict[str, Curve]]: ``curves[outcome][axis]`` for every outcome of
        ``OUTCOMES`` and axis of ``X_AXES``.

    Raises:
        ValueError: If there are no pairs.
    """
    if pairs.empty:
        raise ValueError("no trial pairs to make stay/switch curves from")
    frame = _with_outcomes_and_axes(pairs)
    return {
        outcome: {axis: _curve(frame, outcome, axis, pool_trials) for axis in X_AXES}
        for outcome in OUTCOMES
    }


def split_by_chosen_value(pairs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The pairs whose chosen arm had value 0, and those whose chosen arm had a value above 0.

    The value is the one the pairs were made with: true, or believed (a believed 0 is the
    arm's last payout, whatever its true magnitude is now).

    Args:
        pairs (pd.DataFrame): ``StaySwitchPairs.pairs``.

    Returns:
        dict[str, pd.DataFrame]: ``{CHOSEN_ZERO: V(chosen) == 0, CHOSEN_NONZERO: V(chosen) >
        0}``, in the order of ``CHOSEN_ROWS``; either may be empty.

    Raises:
        ValueError: If a chosen value is negative (magnitudes are never below 0).
    """
    if (pairs["v_chosen"] < 0).any():
        raise ValueError("negative chosen values; reward magnitudes are never below 0")
    zero = pairs["v_chosen"] == 0
    return {
        CHOSEN_ZERO: pairs.loc[zero].reset_index(drop=True),
        CHOSEN_NONZERO: pairs.loc[~zero].reset_index(drop=True),
    }


def stay_switch_curves_by_row(
    pairs: pd.DataFrame, *, pool_trials: bool
) -> dict[str, dict[str, dict[str, Curve]] | None]:
    """:func:`stay_switch_curves` separately for each row of :func:`split_by_chosen_value`.

    A mouse without pairs in a row is not one of that row's mice, so it is left out of the
    row's mean, standard error and n.

    Args:
        pairs (pd.DataFrame): ``StaySwitchPairs.pairs``.
        pool_trials (bool): As in :func:`stay_switch_curves`. Keyword-only.

    Returns:
        dict: ``{row: curves}`` for each row of ``CHOSEN_ROWS``; ``None`` for a row without
        pairs.
    """
    return {
        row: stay_switch_curves(subset, pool_trials=pool_trials) if not subset.empty else None
        for row, subset in split_by_chosen_value(pairs).items()
    }


def stay_heatmap(pairs: pd.DataFrame) -> StayHeatmap:
    """P(stay) over chosen value x alternative value, per mouse and as the mean across mice.

    Args:
        pairs (pd.DataFrame): ``StaySwitchPairs.pairs``.

    Returns:
        StayHeatmap: The grids cover every chosen and alternative value that occurs.

    Raises:
        ValueError: If there are no pairs.
    """
    if pairs.empty:
        raise ValueError("no trial pairs to make a stay heatmap from")
    rows = pd.Index(np.sort(pairs["v_chosen"].unique()), name="v_chosen")
    columns = pd.Index(np.sort(pairs["v_alt"].unique()), name="v_alt")

    def grid(series: pd.Series, fill: float) -> pd.DataFrame:
        table = series.unstack("v_alt").reindex(index=rows, columns=columns)
        return table.fillna(fill) if not np.isnan(fill) else table

    per_mouse = {
        mouse: grid(group.groupby(["v_chosen", "v_alt"])["stay"].mean(), np.nan)
        for mouse, group in pairs.groupby("mouse")
    }
    stack = np.stack([table.to_numpy(dtype=float) for table in per_mouse.values()])
    p_stay = pd.DataFrame(nan_mean(stack, axis=0), index=rows, columns=columns)
    n_pairs = grid(pairs.groupby(["v_chosen", "v_alt"]).size(), 0).astype(int)
    return StayHeatmap(p_stay, n_pairs, per_mouse, len(per_mouse))
