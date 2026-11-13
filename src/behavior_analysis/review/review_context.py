"""Where a session sits in a mouse's problems, for the rows of the review tables.

Every review table names the task type and the problem (number within the type, and overall) of
the session a row is about. A session that is in a problem has them. A session that the
importer left out of every problem (a skipped or flagged one) has none of its own; it gets the
problem the mouse was running before it, so the row can be placed in the mouse's history.
"""

from dataclasses import dataclass

import pandas as pd

from src.behavior_import.problem_numbering import classify_task_name
from src.behavior_import.session_order import session_sort_key

PROBLEM_COLUMNS = ("task_type", "problem_in_type", "overall_problem")
PROBLEM_DTYPES = {"task_type": "object", "problem_in_type": "Int64", "overall_problem": "Int64"}


def task_type_of_name(task_name: str | None) -> str | None:
    """``"blocked"`` or ``"drifting"`` for a task name, ``None`` if the name says neither.

    A review table must list sessions with a pretraining or unfamiliar task name, so unlike
    :func:`~src.behavior_import.problem_numbering.classify_task_name` this does not raise.

    Args:
        task_name (str | None): The ``task_name`` info row, or ``None``.

    Returns:
        str | None: The task type, or ``None`` for a missing, unrecognised or ambiguous name.
    """
    try:
        return classify_task_name(task_name)
    except ValueError:  # deliberate: a name that is not a bandit task is a fact to list
        return None


@dataclass(frozen=True)
class ProblemIndex:
    """Which problem each session of each mouse belongs to.

    Attributes:
        labels (dict): ``{(mouse, overall problem): (task type, problem within the type)}``.
        sessions (dict): ``{mouse: [(session key, overall problem), ...]}`` for the sessions in
            a problem, oldest first.
    """

    labels: dict
    sessions: dict

    @classmethod
    def from_cohort(cls, trials_by_problem: dict, problem_map: pd.DataFrame) -> "ProblemIndex":
        """The index of a cohort.

        Args:
            trials_by_problem (dict): ``{overall problem: {mouse: {session key: session}}}``.
            problem_map (pd.DataFrame): The cohort's problem map (columns ``mouse``,
                ``task_type``, ``problem_in_type``, ``overall_problem``).

        Returns:
            ProblemIndex: The index.
        """
        labels = {
            (row.mouse, row.overall_problem): (row.task_type, row.problem_in_type)
            for row in problem_map.itertuples()
        }
        placed: dict[str, list] = {}
        for problem, mice in trials_by_problem.items():
            for mouse, sessions in mice.items():
                placed.setdefault(mouse, []).extend((key, problem) for key in sessions)
        sessions = {
            mouse: sorted(pairs, key=lambda pair: session_sort_key(pair[0]))
            for mouse, pairs in placed.items()
        }
        return cls(labels, sessions)

    def columns(self, mouse: str | None, session: str | None, task_name: str | None = None) -> dict:
        """``task_type``, ``problem_in_type`` and ``overall_problem`` for a row about a session.

        Args:
            mouse (str | None): Subject id; ``None`` if the row has none.
            session (str | None): Session key; ``None`` if the row has none.
            task_name (str | None): The session's task name. Used for a session that is in no
                problem, to give its own task type. Default ``None``.

        Returns:
            dict: The three values. A session in a problem has its problem's. Any other session
            gets the overall number of the latest earlier session of the mouse that is in one,
            that problem's number within its type only when it is of the session's own task type,
            and the session's own task type if its name gives one (else that problem's). Values
            that cannot be found are ``None``.
        """
        known = dict(self.sessions.get(mouse, ()))
        if session in known:
            kind, in_type = self.labels[(mouse, known[session])]
            return _columns(kind, in_type, known[session])
        own = task_type_of_name(task_name)
        earlier = _latest_before(self.sessions.get(mouse, ()), session)
        if earlier is None:
            return _columns(own, None, None)
        kind, in_type = self.labels[(mouse, earlier)]
        own = own or kind
        return _columns(own, in_type if own == kind else None, earlier)

    def problem_columns(self, mouse: str | None, task_type: str, problem_in_type: int) -> dict:
        """The three columns for a row about a problem rather than a session.

        Args:
            mouse (str | None): Subject id; ``None`` if the row names no mouse.
            task_type (str): ``"blocked"`` or ``"drifting"``.
            problem_in_type (int): The problem's number within its task type.

        Returns:
            dict: ``task_type`` and ``problem_in_type`` as given, and the mouse's overall problem
            number there; ``None`` without a mouse, or if the mouse did not run that problem
            (the number differs between mice).
        """
        overall = next(
            (
                number
                for (name, number), label in self.labels.items()
                if name == mouse and label == (task_type, problem_in_type)
            ),
            None,
        )
        return _columns(task_type, problem_in_type, overall)


def _columns(task_type, problem_in_type, overall_problem) -> dict:
    return {
        "task_type": task_type,
        "problem_in_type": problem_in_type,
        "overall_problem": overall_problem,
    }


def _latest_before(placed, session):
    """The overall problem of the latest session in *placed* before *session*, or ``None``."""
    if session is None:
        return None
    key = session_sort_key(session)
    problems = [problem for other, problem in placed if session_sort_key(other) < key]
    return problems[-1] if problems else None


def magnitudes_text(magnitudes: dict) -> str:
    """``{"A1": 4, "B1": None}`` as ``"A1=4, B1=missing"``: one cell holds every tower.

    Args:
        magnitudes (dict): ``{tower: magnitude or None}``.

    Returns:
        str: ``tower=value`` pairs in the dict's order, ``missing`` for ``None``.
    """
    return ", ".join(f"{t}={'missing' if v is None else v}" for t, v in magnitudes.items())


def typed_frame(rows: list[dict], dtypes: dict) -> pd.DataFrame:
    """*rows* as a frame with exactly the columns and dtypes of *dtypes*, even when empty.

    Args:
        rows (list[dict]): One dict per row, with every key of *dtypes*.
        dtypes (dict): ``{column: dtype}``, in column order.

    Returns:
        pd.DataFrame: The rows, with ``None`` as missing values.
    """
    return pd.DataFrame(rows, columns=list(dtypes)).astype(dtypes)
