"""Each mouse's training sequence, structured then drifting problems, and three stages of it.

The analyses of learning across problems
(:mod:`src.behavior_analysis.choice_logit.logit_by_problem`,
:mod:`src.behavior_analysis.reversals.reversal_curves_by_stage`,
:mod:`src.behavior_analysis.pokes.port_finding_by_stage`) all follow a mouse through the same
sequence:

- **Training order** (:func:`training_problems`): a mouse's structured problems are the blocked
  problems before its first drifting problem; blocked problems that come later (cohort 1, after
  surgery) are not part of the training sequence and are dropped, as are drifting problems after
  the cohort's cutoff. Each problem gets its ``phase`` and its number within the phase.
- **Stages** (:func:`training_stages`): three groups of two problems each, the analogue of the
  first and last problems that Samborska et al. (2022, Fig. 1f,h) compare: the first two
  structured problems (``EARLY``), the first two drifting problems (``SWITCH``) and the mouse's
  last two drifting problems (``LATE``). ``LATE`` uses only drifting problems from the third on,
  so it never shares a problem with ``SWITCH``; a mouse with fewer than three drifting problems
  has no ``LATE`` stage, and one with a single problem in a stage still gets that stage.

**Comparisons** (:func:`stage_tests`): a per-mouse value compared between two stages with a
paired t-test across the mice that have both.

Pure: tables in, tables out.
"""

import numpy as np
import pandas as pd
from scipy import stats

STRUCTURED, DRIFTING = "structured", "drifting"
PHASES = (STRUCTURED, DRIFTING)
TASK_PHASE = {"blocked": STRUCTURED, "drifting": DRIFTING}
EARLY, SWITCH, LATE = "early", "switch", "late"
STAGES = (EARLY, SWITCH, LATE)
STAGE_LABELS = {
    EARLY: "First 2 Structured",
    SWITCH: "First 2 Drifting",
    LATE: "Last 2 Drifting",
}
PROBLEMS_PER_STAGE = 2
PROBLEM_KEYS = ["mouse", "phase", "phase_problem"]
# the pairs compared: start and end of training, then the switch against each end
STAGE_PAIRS = ((EARLY, LATE), (SWITCH, LATE), (EARLY, SWITCH))


def _first_drifting(problems: pd.DataFrame) -> float:
    """The training position of a mouse's first drifting problem (inf if it has none)."""
    drifting = problems.loc[problems["phase"] == DRIFTING, "position"]
    return drifting.min() if len(drifting) else np.inf


def training_problems(
    problems: pd.DataFrame, max_drifting_by_cohort: dict, order_column: str = "overall_problem"
) -> pd.DataFrame:
    """The problems of each mouse's training sequence: structured problems, then drifting ones.

    Args:
        problems (pd.DataFrame): One or more rows per mouse and problem with ``mouse``,
            ``cohort``, ``task`` (``"blocked"`` or ``"drifting"``), ``k`` (the problem number
            within the task type) and *order_column*, which orders a mouse's problems in time.
        max_drifting_by_cohort (dict): ``{cohort: last drifting problem kept}``, ``None`` for all.
            Every cohort of *problems* needs an entry.
        order_column (str): Orders the problems (the first value of a problem is used). Default
            ``"overall_problem"`` (the problem's number in the mouse's whole sequence).

    Returns:
        pd.DataFrame: One row per kept problem: ``mouse, cohort, task, k, phase, phase_problem``
        (1, 2, ... within the phase, as ``k``) and ``position`` (the first *order_column* value),
        in training order per mouse.

    Raises:
        ValueError: If a cohort has no cutoff entry or a task is neither blocked nor drifting.
    """
    missing = sorted(set(problems["cohort"]) - set(max_drifting_by_cohort))
    if missing:
        raise ValueError(f"cohort {missing[0]} has no entry in max_drifting_by_cohort")
    unknown = sorted(set(problems["task"]) - set(TASK_PHASE))
    if unknown:
        raise ValueError(f"unknown task {unknown[0]!r}; expected one of {sorted(TASK_PHASE)}")
    table = (
        problems.groupby(["mouse", "cohort", "task", "k"], as_index=False)[order_column]
        .min()
        .rename(columns={order_column: "position"})
    )
    table = table.assign(phase=table["task"].map(TASK_PHASE), phase_problem=table["k"])
    first = table.groupby("mouse").apply(_first_drifting, include_groups=False)
    structured_in_time = table["position"] < table["mouse"].map(first)
    last = {c: np.inf if k is None else k for c, k in max_drifting_by_cohort.items()}
    drifting_in_cutoff = table["phase_problem"] <= table["cohort"].map(last)
    keep = np.where(table["phase"] == DRIFTING, drifting_in_cutoff, structured_in_time)
    columns = ["mouse", "cohort", "task", "k", "phase", "phase_problem", "position"]
    return table.loc[keep, columns].sort_values(["mouse", "position"]).reset_index(drop=True)


def training_trials(
    trials: pd.DataFrame, max_drifting_by_cohort: dict, order_column: str = "overall_problem"
) -> pd.DataFrame:
    """The trials of each mouse's training sequence (see :func:`training_problems`).

    Args:
        trials (pd.DataFrame): A trial table of both task types with ``cohort``, ``task``,
            ``k`` and *order_column*.
        max_drifting_by_cohort (dict): As :func:`training_problems`.
        order_column (str): As :func:`training_problems`. Default ``"overall_problem"``.

    Returns:
        pd.DataFrame: The kept rows with ``phase`` and ``phase_problem`` added.
    """
    kept = training_problems(trials, max_drifting_by_cohort, order_column)
    kept = kept[["mouse", "task", "k", "phase", "phase_problem"]]
    return trials.merge(kept, on=["mouse", "task", "k"]).reset_index(drop=True)


def _mouse_stages(problems: pd.DataFrame) -> pd.Series:
    """The stage of each of one mouse's problems (``None`` outside the three stages)."""
    stage = pd.Series(None, index=problems.index, dtype=object)
    structured = problems[problems["phase"] == STRUCTURED].sort_values("phase_problem")
    drifting = problems[problems["phase"] == DRIFTING].sort_values("phase_problem")
    stage[structured.index[:PROBLEMS_PER_STAGE]] = EARLY
    stage[drifting.index[:PROBLEMS_PER_STAGE]] = SWITCH
    later = drifting[drifting["phase_problem"] > PROBLEMS_PER_STAGE]
    stage[later.index[-PROBLEMS_PER_STAGE:]] = LATE
    return stage


def training_stages(problems: pd.DataFrame) -> pd.DataFrame:
    """The stage of every problem in the three stages.

    Args:
        problems (pd.DataFrame): :func:`training_problems` (any table with ``mouse, phase,
            phase_problem``, one row per problem).

    Returns:
        pd.DataFrame: ``mouse, phase, phase_problem, stage`` for the problems in a stage.
    """
    unique = problems[PROBLEM_KEYS].drop_duplicates().reset_index(drop=True)
    stage = pd.concat([_mouse_stages(rows) for _, rows in unique.groupby("mouse")])
    staged = unique.assign(stage=stage.reindex(unique.index))
    return staged.dropna(subset=["stage"]).reset_index(drop=True)


def stage_tests(
    per_mouse: pd.DataFrame, value: str, by: list[str] | tuple[str, ...] = ()
) -> pd.DataFrame:
    """Paired t-tests of a per-mouse value between stages, across the mice with both stages.

    Args:
        per_mouse (pd.DataFrame): One row per mouse, stage and *by* group, with ``mouse``,
            ``stage`` and *value* (NaN values are left out).
        value (str): The column compared.
        by (list[str] | tuple[str, ...]): Columns that split the tests (each group is tested on
            its own). Default ``()``.

    Returns:
        pd.DataFrame: The *by* columns, then ``first, second`` (the stages, ``STAGE_PAIRS``),
        ``mean_first, mean_second, mean_difference`` (second minus first), ``se`` (of the
        difference), ``n_mice, t, p``; ``t`` and ``p`` are NaN with fewer than 2 mice.
    """
    rows = []
    groups = per_mouse.groupby(list(by), sort=False) if by else [((), per_mouse)]
    for key, table in groups:
        labels = dict(zip(by, key if isinstance(key, tuple) else (key,), strict=True))
        wide = table.pivot_table(index="mouse", columns="stage", values=value, aggfunc="mean")
        for first, second in STAGE_PAIRS:
            if first not in wide or second not in wide:
                pair = pd.DataFrame(columns=[first, second])
            else:
                pair = wide[[first, second]].dropna()
            difference = (pair[second] - pair[first]).to_numpy(dtype=float)
            n = len(difference)
            t, p = stats.ttest_rel(pair[second], pair[first]) if n > 1 else (np.nan, np.nan)
            rows.append(dict(
                **labels, first=first, second=second,
                mean_first=pair[first].mean() if n else np.nan,
                mean_second=pair[second].mean() if n else np.nan,
                mean_difference=difference.mean() if n else np.nan,
                se=difference.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan,
                n_mice=n, t=float(t), p=float(p),
            ))  # fmt: skip
    return pd.DataFrame(rows)
