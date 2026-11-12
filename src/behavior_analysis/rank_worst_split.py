"""Where does the worst alternative's positive stay weight live? (a probe of test 1)

In the stay model of :func:`rank_stay.stay_weights` the mice's weight on the worst alternative's
believed value is positive: the better the worst alternative is believed to be, the more they
stay. Neither a rank algorithm (about 0) nor a softmax learner (negative) predicts that.

The task has exactly one zero arm at all times. After an own payout above 0, one of the two
alternatives must therefore be the zero arm; if *both* are believed to pay, one of those beliefs
is stale (the arm turned zero since it was last visited). After an own payout of 0 both
alternatives pay, so an alternative believed 0 is stale. A high believed value of the worst
alternative is thus a sign of stale beliefs, which could make leaving less attractive without
being a comparison of values at all. This module separates the two readings:

- the stay model with the trials since each alternative was last visited, and with a
  ``worst_pos`` flag (the worst alternative is believed to pay): the weight of the value that
  remains once the flag is in the model, and the weight of the flag itself;
- the same models within own payouts above 0 and at 0;
- P(stay) in the four cells of own payout (0 or above) by consistency with one zero arm;
- artifact checks: unpenalized refits, the variance inflation of every regressor and the
  correlations of the worst alternative's value with the other regressors.

Pure: tables in, a :class:`~src.behavior_analysis.rank_result.RankResult` out.
"""

import numpy as np
import pandas as pd

from src.behavior_analysis import rank_logit as rl
from src.behavior_analysis import rank_state as rs
from src.behavior_analysis.rank_result import RankResult, empty_curves
from src.behavior_analysis.rank_stay import HISTORY, stay_outcome

BASE = ("payout", "own_zero", "v_best", "v_worst")
AGES = ("log_age_best", "log_age_worst")
FLAG = "worst_pos"
MIN_CELL = 100  # decisions a cell needs for a fit; fewer give NaN
CELLS = ("nz", "z")  # own payout above 0, own payout 0
CELL_FITS = ("w_worst", "w_best", "w_worst_flag", "w_flag")
CELL_STAY = ("stay_ok", "stay_stale", "share_stale")
READOUTS = (
    "ws_w_worst_ages", "ws_w_best_ages", "ws_w_age_best", "ws_w_age_worst", "ws_w_worst_flag",
    "ws_w_flag", "ws_w_worst_noridge", "ws_w_worst_flag_noridge",
    *[f"ws_{kind}_{cell}" for cell in CELLS for kind in CELL_FITS],
    *[f"ws_{kind}_{cell}" for cell in CELLS for kind in CELL_STAY],
    "ws_vif_worst_base", "ws_vif_worst_full", "ws_vif_max_full", "ws_corr_worst_payout",
    "ws_corr_worst_best", "ws_corr_worst_flag", "ws_corr_worst_age",
)  # fmt: skip


def trials_since_visit(state: rs.MouseState) -> np.ndarray:
    """``(n, 3)`` trials since each arm was last chosen, counted at every trial (0 for the arm
    chosen on it); NaN for an arm not yet chosen in the problem."""
    rows = np.arange(len(state.chosen))
    ages = np.full((len(rows), rs.N_ARMS), np.nan)
    for arm in range(rs.N_ARMS):
        last = np.maximum.accumulate(np.where(state.chosen == arm, rows, -1))
        seen = (last >= 0) & (state.problem_code[last] == state.problem_code)
        ages[:, arm] = np.where(seen, state.trial - state.trial[last], np.nan)
    return ages


def with_ages(state: rs.MouseState, table: pd.DataFrame) -> pd.DataFrame:
    """*table* (``rank_decisions.decision_table`` of *state*) with the alternatives' ages.

    Adds ``age_best``, ``age_worst`` (trials since the best / worst alternative was last chosen),
    their ``log1p`` as ``log_age_best``, ``log_age_worst``, and ``worst_pos`` (1 if the worst
    alternative is believed to pay).
    """
    rows = table["t"].to_numpy()
    own = table["own"].to_numpy()
    believed, ages = state.believed[rows], trials_since_visit(state)[rows]
    pick = np.arange(len(rows))
    first, second = (own + 1) % rs.N_ARMS, (own + 2) % rs.N_ARMS
    first_is_best = believed[pick, first] >= believed[pick, second]
    age_best = np.where(first_is_best, ages[pick, first], ages[pick, second])
    age_worst = np.where(first_is_best, ages[pick, second], ages[pick, first])
    return table.assign(
        age_best=age_best,
        age_worst=age_worst,
        log_age_best=np.log1p(age_best),
        log_age_worst=np.log1p(age_worst),
        worst_pos=(table["v_worst"].to_numpy() > 0).astype(float),
    )


def consistent_with_one_zero(table: pd.DataFrame) -> pd.Series:
    """True where exactly one of the own payout and the two believed alternative values is 0."""
    zeros = (table[["payout", "v_best", "v_worst"]] == 0).sum(axis=1)
    return zeros == 1


def _r_squared(x: np.ndarray, others: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(x)), others])
    residual = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    return 1.0 - residual.var() / x.var()


def variance_inflation(X: np.ndarray) -> np.ndarray:
    """``(k,)`` variance inflation factor ``1 / (1 - R^2)`` of each column on the others.

    Args:
        X (np.ndarray): ``(n, k)`` regressors without the intercept, none constant.

    Returns:
        np.ndarray: ``inf`` for a column that is a linear combination of the others.
    """
    unexplained = np.array(
        [1.0 - _r_squared(X[:, j], np.delete(X, j, axis=1)) for j in range(X.shape[1])]
    )
    with np.errstate(divide="ignore"):
        return np.where(unexplained < 1e-10, np.inf, 1.0 / unexplained)


def _columns(table: pd.DataFrame, terms) -> tuple[np.ndarray, tuple]:
    """Design without the intercept, ``terms`` then the history, dropping constant columns."""
    names = tuple(c for c in (*terms, *HISTORY) if table[c].nunique() > 1)
    return np.column_stack([table[c].to_numpy(float) for c in names]), names


def _fit(table: pd.DataFrame, terms, l2: float | None = None) -> rl.LogitFit | None:
    """Stay logit on *terms* and the history, or ``None`` where the data cannot support it.

    ``None`` (too few decisions, a single stay/leave class, or an unpenalized fit that does not
    exist) is a deliberate result for a cell: its readouts are NaN.
    """
    if len(table) < MIN_CELL:
        return None
    X, names = _columns(table, terms)
    design = np.column_stack([np.ones(len(table)), X])
    options = {} if l2 is None else {"l2": l2}
    try:
        return rl.fit_logit(design, stay_outcome(table), ("intercept", *names), **options)
    except (rl.DegenerateOutcomeError, RuntimeError, np.linalg.LinAlgError):
        return None


def _weight(fit: rl.LogitFit | None, name: str) -> float:
    if fit is None or name not in fit.names:
        return np.nan
    return fit.weight(name)


def _pooled(table: pd.DataFrame) -> dict:
    plain = _fit(table, (*BASE, *AGES))
    flagged = _fit(table, (*BASE, *AGES, FLAG))
    return {
        "ws_w_worst_ages": _weight(plain, "v_worst"),
        "ws_w_best_ages": _weight(plain, "v_best"),
        "ws_w_age_best": _weight(plain, "log_age_best"),
        "ws_w_age_worst": _weight(plain, "log_age_worst"),
        "ws_w_worst_flag": _weight(flagged, "v_worst"),
        "ws_w_flag": _weight(flagged, FLAG),
        "ws_w_worst_noridge": _weight(_fit(table, (*BASE, *AGES), l2=0.0), "v_worst"),
        "ws_w_worst_flag_noridge": _weight(_fit(table, (*BASE, *AGES, FLAG), l2=0.0), "v_worst"),
    }


def _cell(table: pd.DataFrame, label: str) -> dict:
    plain = _fit(table, (*BASE, *AGES))
    flagged = _fit(table, (*BASE, *AGES, FLAG))
    return {
        f"ws_w_worst_{label}": _weight(plain, "v_worst"),
        f"ws_w_best_{label}": _weight(plain, "v_best"),
        f"ws_w_worst_flag_{label}": _weight(flagged, "v_worst"),
        f"ws_w_flag_{label}": _weight(flagged, FLAG),
    }


def _by_own_payout(table: pd.DataFrame) -> dict:
    paid = table["own_zero"] == 0
    return _cell(table[paid], "nz") | _cell(table[~paid], "z")


def _stay_in(table: pd.DataFrame, mask: pd.Series) -> float:
    return float(1 - table.loc[mask, "leave"].mean()) if mask.any() else np.nan


def _by_consistency(table: pd.DataFrame) -> dict:
    ok = consistent_with_one_zero(table)
    out = {}
    for label, own in (("nz", table["own_zero"] == 0), ("z", table["own_zero"] == 1)):
        out[f"ws_stay_ok_{label}"] = _stay_in(table, own & ok)
        out[f"ws_stay_stale_{label}"] = _stay_in(table, own & ~ok)
        out[f"ws_share_stale_{label}"] = (
            float((own & ~ok).sum() / own.sum()) if own.any() else np.nan
        )
    return out


def _correlation(a: pd.Series, b: pd.Series) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if a.nunique() > 1 and b.nunique() > 1 else np.nan


def _inflation(table: pd.DataFrame, terms) -> tuple[float, float]:
    """``(variance inflation of the worst alternative's value, the largest of any regressor)``."""
    X, names = _columns(table, terms)
    factors = variance_inflation(X)
    worst = factors[names.index("v_worst")] if "v_worst" in names else np.nan
    return float(worst), float(factors.max())


def _checks(table: pd.DataFrame) -> dict:
    worst = table["v_worst"]
    base, _ = _inflation(table, BASE)
    full, largest = _inflation(table, (*BASE, *AGES, FLAG))
    return {
        "ws_vif_worst_base": base,
        "ws_vif_worst_full": full,
        "ws_vif_max_full": largest,
        "ws_corr_worst_payout": _correlation(worst, table["payout"]),
        "ws_corr_worst_best": _correlation(worst, table["v_best"]),
        "ws_corr_worst_flag": _correlation(worst, table[FLAG]),
        "ws_corr_worst_age": _correlation(worst, table["log_age_worst"]),
    }


def worst_weight_split(table: pd.DataFrame) -> RankResult:
    """The stay weight of the worst alternative with ages and the stale-belief flag, by cell.

    Args:
        table (pd.DataFrame): :func:`with_ages` of a mouse's (or a replay's) decision table.

    Returns:
        RankResult: Readouts prefixed ``ws_``, NaN where a cell cannot support the fit:

        - ``ws_w_worst_ages``, ``ws_w_best_ages``, ``ws_w_age_best``, ``ws_w_age_worst``: stay
          weights with the log ages of the alternatives added to the test-1 model;
        - ``ws_w_worst_flag``, ``ws_w_flag``: the worst value's weight and the weight of the
          ``worst_pos`` flag with the flag added;
        - ``ws_w_worst_noridge``, ``ws_w_worst_flag_noridge``: the same without the ridge;
        - ``ws_w_worst_<cell>``, ``ws_w_best_<cell>``, ``ws_w_worst_flag_<cell>``,
          ``ws_w_flag_<cell>`` for ``<cell>`` ``nz`` (own payout above 0) and ``z`` (own payout 0);
        - ``ws_stay_ok_<cell>`` / ``ws_stay_stale_<cell>``: P(stay) where the beliefs are
          consistent with one zero arm or not, and ``ws_share_stale_<cell>``: the share of stale
          decisions;
        - ``ws_vif_worst_base``, ``ws_vif_worst_full``, ``ws_vif_max_full``,
          ``ws_corr_worst_{payout,best,flag,age}``: collinearity of the regressors.
    """
    readouts = _pooled(table) | _by_own_payout(table) | _by_consistency(table) | _checks(table)
    return RankResult(readouts, empty_curves())
