"""The choice logit fitted to each problem on its own, across structured and drifting training.

Do the mice integrate more history as they learn the task? Samborska et al. (2022, *Nat
Neurosci*, Fig. 1i-l) fitted a logistic regression of choices on past choices and past
outcome x choice in every problem of a two-arm reversal task, and found the weights grow across
problems. Here the standard three-arm logit of :mod:`.choice_logit` (reward magnitude and choice
at lags 1 to ``n_lags`` per port, a bias per tower) is fitted to each mouse's problems one at a
time, through the structured (blocked) problems that come first and then the drifting ones.

- **Training order** (:func:`~src.behavior_analysis.data.training_stages.training_trials`): a
  mouse's structured problems are the blocked problems before its first drifting problem, then
  its drifting problems up to the cohort's cutoff.
- **Fits** (:func:`fit_by_problem`): one fit per mouse and problem with at least ``MIN_TRIALS``
  trials; the tower biases are fitted per problem. ``part`` fits the whole problem (``ALL``), its
  first ``N_EARLY_TRIALS`` trials (``EARLY_TRIALS``) or the trials after them
  (``LATE_TRIALS``). Learning to learn predicts that the early-trial weights grow across problems
  (a new problem is picked up faster) and the gap to the late-trial weights shrinks; growth only
  in the late trials points to a more exploitative end state instead. The history of the first
  late trials still reaches into the early trials: the design is built on the whole problem and
  then cut.
- **Summaries** (:func:`weight_summaries`): as in the paper, the lag-1 weight and the mean weight
  of lags 2 to ``n_lags``, for reward and choice.
- **Trends** (:func:`phase_trends`, :func:`trend_tests`): per mouse, the least-squares slope of a
  summary against the problem number within each phase, and the step at the switch (first
  drifting problem minus last structured problem); tested across mice with a one-sample t-test
  against 0.

Two codings of the history (``coding``):

- ``MAGNITUDE`` (the standard logit): per port, ``choice_lag<k>`` is 1 if the port was chosen
  ``k`` trials back, and ``reward_lag<k>`` is the number of pulses it paid then. The reward weight
  is the analogue of the paper's outcome x choice term, per pulse; the choice weight is the pull
  of a past choice that paid 0.
- ``OUTCOME`` (the paper's coding): the outcome is +1 if the choice paid anything and -1 if it
  paid 0, and per port ``choice_lag<k>`` (1 if chosen) is joined by
  ``outcome_x_choice_lag<k>`` (the outcome if chosen, else 0). The choice weight is then the
  average pull of a past choice (rewarded or not, win-stay and lose-stay alike), and the
  outcome x choice weight is half the difference between a rewarded and an unrewarded one
  (win-stay versus lose-shift). The paper's outcome-alone regressor is the same for every port
  in a three-arm logit, so it cancels and has no weight.

The reward weight of ``MAGNITUDE`` is per pulse. Structured problems pay 0, 1 or 4 pulses
(sometimes 3) and drifting problems every magnitude from 0 to 4, so a step at the switch can
come from the different payouts as well as from learning; the slopes within each phase compare
like with like. ``OUTCOME`` ignores the magnitude, as the paper's binary reward did.

Everything here is pure: trial tables in, tables out.
"""

import numpy as np
import pandas as pd
from scipy import stats

from src.behavior_analysis.choice_logit import choice_logit as cl
from src.behavior_analysis.data.training_stages import DRIFTING, PHASES, STRUCTURED

MIN_TRIALS = 150  # fewer trials than this in a problem: no fit (too few for 2 * lags + 2 weights)
MIN_PROBLEMS = 2  # a slope needs this many problems in the phase
MAGNITUDE, OUTCOME = "magnitude", "outcome"
CODINGS = (MAGNITUDE, OUTCOME)
OUTCOME_X_CHOICE = "outcome_x_choice"
ALL, EARLY_TRIALS, LATE_TRIALS = "all", "early", "late"
PARTS = (ALL, EARLY_TRIALS, LATE_TRIALS)
# the early part of a problem: its first trials, as many as one fit needs
N_EARLY_TRIALS = MIN_TRIALS
LAG_ONE, OLDER = "lag 1", "lags 2+"
SUMMARY_COLUMNS = ["mouse", "cohort", "phase", "phase_problem", "group", "summary", "weight"]


def outcome_design(design: cl.LogitDesign) -> cl.LogitDesign:
    """The paper's coding from a design with the ``BINARY`` columns: choice and outcome x choice.

    Args:
        design (choice_logit.LogitDesign): A design made with ``extras=(BINARY,)`` and no tail.

    Returns:
        choice_logit.LogitDesign: The choice lags, the outcome x choice lags (``c * (2b - 1)``
        with ``c`` the choice and ``b`` the paid-anything indicator of the same lag, so 0 where
        the port was not chosen, and on a lag that reaches across a session start, where the
        choice history is cleared) and the towers. The outcome x choice columns keep the
        ``binary_lag<k>`` names; :func:`fit_problem` renames them in the weights.
    """
    kept = cl.select_features(design, (cl.CHOICE, cl.TOWER, cl.BINARY))
    names = kept.feature_names
    X = kept.X.copy()
    for k, name in enumerate(names):
        if name.startswith("binary_lag"):
            choice = X[:, :, names.index(name.replace("binary_lag", "choice_lag"))]
            X[:, :, k] = choice * (2 * X[:, :, k] - 1)
    return kept._replace(X=X)


def _part_mask(n_trials: int, part: str, n_early: int) -> np.ndarray:
    """Which of a problem's trials (in design order) belong to *part*."""
    position = np.arange(n_trials)
    if part == EARLY_TRIALS:
        return position < n_early
    if part == LATE_TRIALS:
        return position >= n_early
    return np.ones(n_trials, dtype=bool)


def fit_problem(
    rows: pd.DataFrame,
    n_lags: int = cl.DEFAULT_LAGS,
    l2: float = cl.DEFAULT_L2,
    coding: str = MAGNITUDE,
    part: str = ALL,
    n_early: int = N_EARLY_TRIALS,
) -> pd.DataFrame:
    """The logit weights of one mouse's problem.

    Args:
        rows (pd.DataFrame): The trials of one mouse and one problem.
        n_lags (int): Reward (or outcome x choice) and choice lags. Default
            ``choice_logit.DEFAULT_LAGS``.
        l2 (float): Penalty weight. Default ``choice_logit.DEFAULT_L2``.
        coding (str): ``MAGNITUDE`` or ``OUTCOME`` (see the module docstring). Default
            ``MAGNITUDE``.
        part (str): ``ALL``, ``EARLY_TRIALS`` (the first *n_early* trials) or ``LATE_TRIALS``
            (the rest). Default ``ALL``.
        n_early (int): Trials in the early part. Default ``N_EARLY_TRIALS``.

    Returns:
        pd.DataFrame: :func:`choice_logit.weights_table` (``feature``, ``group``, ``lag``,
        ``weight``) of the fit; with ``OUTCOME`` the outcome x choice weights have the group
        ``OUTCOME_X_CHOICE`` and features ``outcome_x_choice_lag<k>``.

    Raises:
        ValueError: For an unknown *coding* or *part*.
    """
    if coding not in CODINGS:
        raise ValueError(f"unknown coding {coding!r}; expected one of {CODINGS}")
    if part not in PARTS:
        raise ValueError(f"unknown part {part!r}; expected one of {PARTS}")
    if coding == MAGNITUDE:
        design = cl.design_matrix(rows, n_lags=n_lags)
    else:
        design = outcome_design(cl.design_matrix(rows, n_lags=n_lags, extras=(cl.BINARY,)))
    mask = _part_mask(len(design.chosen), part, n_early)
    fit = cl.fit_logit(design.X[mask], design.chosen[mask], design.feature_names, l2=l2)
    weights = cl.weights_table(rows["mouse"].iloc[0], design.feature_names, fit.beta)
    binary = weights["group"] == cl.BINARY
    weights.loc[binary, "group"] = OUTCOME_X_CHOICE
    weights.loc[binary, "feature"] = weights.loc[binary, "feature"].str.replace(
        "binary_lag", f"{OUTCOME_X_CHOICE}_lag"
    )
    return weights


def fit_by_problem(
    trials: pd.DataFrame,
    n_lags: int = cl.DEFAULT_LAGS,
    l2: float = cl.DEFAULT_L2,
    min_trials: int = MIN_TRIALS,
    coding: str = MAGNITUDE,
    part: str = ALL,
    n_early: int = N_EARLY_TRIALS,
) -> pd.DataFrame:
    """Fit the logit to every mouse's problems (or one part of each) one at a time.

    Args:
        trials (pd.DataFrame): :func:`training_trials`.
        n_lags (int): Reward and choice lags. Default ``choice_logit.DEFAULT_LAGS``.
        l2 (float): Penalty weight. Default ``choice_logit.DEFAULT_L2``.
        min_trials (int): A problem (or part) with fewer trials gets no fit. Default
            ``MIN_TRIALS``.
        coding (str): ``MAGNITUDE`` or ``OUTCOME``. Default ``MAGNITUDE``.
        part (str): ``ALL``, ``EARLY_TRIALS`` or ``LATE_TRIALS`` (see :func:`fit_problem`).
            Default ``ALL``.
        n_early (int): Trials in the early part. Default ``N_EARLY_TRIALS``.

    Returns:
        pd.DataFrame: The weights of every fit, with ``cohort``, ``phase``, ``phase_problem``
        and ``n_trials`` in front of the weights-table columns.
    """
    keys = ["mouse", "cohort", "phase", "phase_problem"]
    fits = []
    for key, rows in trials.groupby(keys, sort=True):
        n_part = int(_part_mask(len(rows), part, n_early).sum())
        if n_part < min_trials:
            continue
        weights = fit_problem(rows, n_lags, l2, coding, part, n_early).drop(columns="mouse")
        fits.append(weights.assign(**dict(zip(keys, key, strict=True)), n_trials=n_part))
    if not fits:
        return pd.DataFrame(columns=[*keys, "n_trials", "feature", "group", "lag", "weight"])
    table = pd.concat(fits, ignore_index=True)
    return table[[*keys, "n_trials", *[c for c in table.columns if c not in [*keys, "n_trials"]]]]


def weight_summaries(weights: pd.DataFrame) -> pd.DataFrame:
    """The lag-1 weight and the mean weight of the older lags, for every lagged group.

    Args:
        weights (pd.DataFrame): :func:`fit_by_problem`.

    Returns:
        pd.DataFrame: ``SUMMARY_COLUMNS``: one row per fit, group (``"reward"`` or
        ``"outcome_x_choice"``, and ``"choice"``) and summary (``"lag 1"``, ``"lags 2+"``).
    """
    lagged = weights[weights["lag"].notna()]
    lagged = lagged.assign(summary=np.where(lagged["lag"] == 1, LAG_ONE, OLDER))
    keys = ["mouse", "cohort", "phase", "phase_problem", "group", "summary"]
    out = lagged.groupby(keys, as_index=False)["weight"].mean()
    return out[SUMMARY_COLUMNS]


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of *y* on *x*; NaN with fewer than ``MIN_PROBLEMS`` points."""
    if len(x) < MIN_PROBLEMS or np.unique(x).size < 2:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def _mouse_trends(rows: pd.DataFrame) -> pd.Series:
    """One mouse's slope in each phase and its step at the switch, for one summary."""
    out = {}
    for phase in PHASES:
        mine = rows[rows["phase"] == phase].sort_values("phase_problem")
        out[f"slope_{phase}"] = _slope(
            mine["phase_problem"].to_numpy(float), mine["weight"].to_numpy()
        )
    structured = rows[rows["phase"] == STRUCTURED]
    drifting = rows[rows["phase"] == DRIFTING]
    if len(structured) and len(drifting):
        last = structured.loc[structured["phase_problem"].idxmax(), "weight"]
        first = drifting.loc[drifting["phase_problem"].idxmin(), "weight"]
        out["step"] = first - last
    else:
        out["step"] = np.nan
    return pd.Series(out)


def phase_trends(summaries: pd.DataFrame) -> pd.DataFrame:
    """Per mouse and summary: the slope within each phase and the step at the switch.

    Args:
        summaries (pd.DataFrame): :func:`weight_summaries`.

    Returns:
        pd.DataFrame: ``mouse, cohort, group, summary, slope_structured, slope_drifting, step``.
        A slope is NaN with fewer than ``MIN_PROBLEMS`` problems in the phase; the step is NaN
        without both phases.
    """
    keys = ["mouse", "cohort", "group", "summary"]
    return (
        summaries.groupby(keys)[["phase", "phase_problem", "weight"]]
        .apply(_mouse_trends)
        .reset_index()
    )


def trend_tests(trends: pd.DataFrame) -> pd.DataFrame:
    """One-sample t-tests across mice of each slope and step against 0.

    Args:
        trends (pd.DataFrame): :func:`phase_trends`.

    Returns:
        pd.DataFrame: ``group, summary, measure, mean, se, n_mice, t, p`` per summary and measure
        (``slope_structured``, ``slope_drifting``, ``step``); ``t`` and ``p`` are NaN with fewer
        than 2 mice.
    """
    rows = []
    for (group, summary), mine in trends.groupby(["group", "summary"], sort=False):
        for measure in (f"slope_{STRUCTURED}", f"slope_{DRIFTING}", "step"):
            values = mine[measure].dropna().to_numpy()
            n = len(values)
            se = values.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            t, p = stats.ttest_1samp(values, 0.0) if n > 1 else (np.nan, np.nan)
            mean = values.mean() if n else np.nan
            rows.append(dict(group=group, summary=summary, measure=measure, mean=mean, se=se,
                             n_mice=n, t=float(t), p=float(p)))  # fmt: skip
    return pd.DataFrame(rows)


GAP = "late minus early"


def early_late_gap(early: pd.DataFrame, late: pd.DataFrame) -> pd.DataFrame:
    """Per mouse, problem and summary: the late-trial weight minus the early-trial weight.

    Args:
        early (pd.DataFrame): :func:`weight_summaries` of the ``EARLY_TRIALS`` fits.
        late (pd.DataFrame): :func:`weight_summaries` of the ``LATE_TRIALS`` fits.

    Returns:
        pd.DataFrame: ``SUMMARY_COLUMNS`` with ``weight`` the difference, for the problems with
        both parts.
    """
    keys = [c for c in SUMMARY_COLUMNS if c != "weight"]
    both = early.merge(late, on=keys, suffixes=("_early", "_late"))
    return both.assign(weight=both["weight_late"] - both["weight_early"])[SUMMARY_COLUMNS]


def early_late_tests(early: pd.DataFrame, late: pd.DataFrame) -> pd.DataFrame:
    """The trend tests of the early-trial weights, the late-trial weights and their gap.

    Args:
        early (pd.DataFrame): :func:`weight_summaries` of the ``EARLY_TRIALS`` fits.
        late (pd.DataFrame): :func:`weight_summaries` of the ``LATE_TRIALS`` fits.

    Returns:
        pd.DataFrame: :func:`trend_tests` of each, with ``series`` (``"early"``, ``"late"``,
        ``GAP``) in front.
    """
    series = {EARLY_TRIALS: early, LATE_TRIALS: late, GAP: early_late_gap(early, late)}
    tests = [trend_tests(phase_trends(table)).assign(series=name) for name, table in series.items()]
    out = pd.concat(tests, ignore_index=True)
    return out[["series", *[c for c in out.columns if c != "series"]]]
