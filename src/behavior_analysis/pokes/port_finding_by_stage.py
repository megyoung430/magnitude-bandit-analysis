"""Where mice poke at the start of a problem, early and late in training (Samborska Fig. 1h).

Samborska et al. (2022, Fig. 1g,h) counted out-of-sequence pokes (pokes at a choice port that do
not fit the initiate-then-choose sequence) and found fewer of them both within a problem and
across problems: the mice learned the trial structure, and a new problem's layout faster. Here
the analogue is where the mice poke as they meet a new layout, from the port-finding summary
(:mod:`src.behavior_analysis.pokes.port_finding_summary`): the fraction of pokes at the current
choice ports, the previous problem's ports, the initiation port and other ports, in three windows
(port finding, the first 50 task trials, the first three blocks), compared between the three
training stages of :mod:`src.behavior_analysis.data.training_stages` (first two structured,
first two drifting, last two drifting problems).

- **Per stage** (:func:`stage_fractions`): each mouse's fractions averaged over the problems of a
  stage, each problem weighted equally. The previous-port fraction of a mouse's first problem is
  NaN (it has no previous ports) and is left out of the mean.
- **Comparisons** (:func:`in_sequence_fractions`): the in-sequence fraction, per poke kind and
  window, between stages with paired t-tests
  (:func:`~src.behavior_analysis.data.training_stages.stage_tests`). In sequence means the next
  step of the trial: after initiation, a current choice port; after the choice, the initiation
  port (the paper's out-of-sequence pokes are the choice-port visits after a choice). Learning
  to learn predicts more in-sequence pokes, sooner, in later problems.

Pure: tables in, tables out.
"""

import pandas as pd

from src.behavior_analysis.data.training_stages import PROBLEM_KEYS, TASK_PHASE
from src.behavior_analysis.pokes.port_finding_summary import CURRENT, INITIATION

# the port category that is the next step of the trial, per poke kind
IN_SEQUENCE = {"after_initiation": CURRENT, "after_choice": INITIATION}


def problem_keys(fractions: pd.DataFrame) -> pd.DataFrame:
    """*fractions* with ``phase`` and ``phase_problem`` from its ``task_type`` and ``k`` labels.

    Args:
        fractions (pd.DataFrame): ``category_fraction_table`` of entries labeled with
            ``task_type`` (``"blocked"`` or ``"drifting"``) and ``k``.

    Returns:
        pd.DataFrame: The same rows with ``phase`` and ``phase_problem`` added.

    Raises:
        ValueError: If a task type is neither blocked nor drifting.
    """
    unknown = sorted(set(fractions["task_type"]) - set(TASK_PHASE))
    if unknown:
        raise ValueError(f"unknown task type {unknown[0]!r}; expected one of {sorted(TASK_PHASE)}")
    return fractions.assign(
        phase=fractions["task_type"].map(TASK_PHASE), phase_problem=fractions["k"]
    )


def stage_fractions(fractions: pd.DataFrame, stages: pd.DataFrame) -> pd.DataFrame:
    """Each mouse's port-category fractions averaged over the problems of each stage.

    Args:
        fractions (pd.DataFrame): :func:`problem_keys` of a ``category_fraction_table``.
        stages (pd.DataFrame): :func:`~src.behavior_analysis.data.training_stages.training_stages`.

    Returns:
        pd.DataFrame: ``mouse, stage, kind, window, category, fraction, n_problems``;
        ``n_problems`` counts the problems with a value.
    """
    staged = fractions.merge(stages, on=PROBLEM_KEYS)
    grouped = staged.dropna(subset=["fraction"]).groupby(
        ["mouse", "stage", "kind", "window", "category"], sort=False
    )
    return grouped.agg(fraction=("fraction", "mean"), n_problems=("fraction", "size")).reset_index()


def in_sequence_fractions(fractions: pd.DataFrame) -> pd.DataFrame:
    """The rows of the in-sequence category of each poke kind (``IN_SEQUENCE``).

    Args:
        fractions (pd.DataFrame): :func:`stage_fractions` (or any table with ``kind`` and
            ``category``).

    Returns:
        pd.DataFrame: The rows whose category is the next step of the trial for their kind.

    Raises:
        ValueError: If a poke kind has no in-sequence category.
    """
    unknown = sorted(set(fractions["kind"]) - set(IN_SEQUENCE))
    if unknown:
        raise ValueError(f"no in-sequence category for poke kind {unknown[0]!r}")
    return fractions[fractions["category"] == fractions["kind"].map(IN_SEQUENCE)]
