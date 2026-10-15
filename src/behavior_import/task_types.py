"""Task type (blocked or drifting) of each session, and per-mouse numbering within a type.

The type comes from the pyControl ``task_name`` that ``import_data`` stores on every session,
read as its last path component, case-insensitively: a name containing ``blocked`` or
``structured`` is a blocked task, one containing ``drifting`` is a drifting task. Anything
else raises, so a new or misspelled task name gets fixed instead of silently analysed as the
wrong kind. Pretraining sessions never reach this module: problem numbering leaves them out.

The number of blocked problems differs between mice, so the overall problem number of the
first drifting problem differs too. :func:`regroup_by_task_type` renumbers each mouse's
problems within their type (blocked 1, 2, ...; drifting 1, 2, ...) so that "drifting problem 1"
lines up across mice; :func:`problem_map_table` keeps the overall number next to it.
"""

import pandas as pd

from src.behavior_import.import_data import session_source_paths
from src.behavior_import.import_data import session_task_type as _classify
from src.behavior_import.problem_numbering import (  # noqa: F401  (re-exported)
    BLOCKED,
    DRIFTING,
    classify_task_name,
)
from src.behavior_import.session_order import ordered_session_keys

MAP_COLUMNS = ("mouse", "task_type", "problem_in_type", "overall_problem", "sessions")


def session_task_type(subject: str, session_key: str, session: dict) -> str:
    """:func:`classify_task_name` of one session; errors name the subject, session and file."""
    return _classify(
        subject,
        session_key,
        session.get("task_name"),
        session_source_paths(session),
        skip_pretraining=False,
    )


def _problem_task_type(subject: str, problem: int, sessions: dict) -> str:
    """The one task type shared by all *sessions* of a problem; raises if they differ."""
    types = {key: session_task_type(subject, key, sess) for key, sess in sessions.items()}
    if len(set(types.values())) > 1:
        listed = "; ".join(f"{key}: {kind}" for key, kind in types.items())
        raise ValueError(
            f"{subject} problem {problem} mixes task types across its sessions ({listed})"
        )
    return next(iter(types.values()))


def number_within_task_type(subjects_trials_by_problem: dict) -> dict:
    """``{(subject, overall_problem): (task_type, k)}``, numbering each mouse within a type.

    Each mouse's problems are taken in order of their overall number; the first blocked one is
    ``("blocked", 1)``, the second ``("blocked", 2)``, and likewise for drifting.

    Args:
        subjects_trials_by_problem: ``{problem: {subject: {session_key: session_dict}}}``,
            every session carrying the importer's ``"task_name"``.

    Raises:
        ValueError: If a session's task name is not recognised, or one problem of a mouse has
            sessions of both types.
    """
    seen_per_type: dict[str, dict[str, int]] = {BLOCKED: {}, DRIFTING: {}}
    numbering = {}
    for problem in sorted(subjects_trials_by_problem):
        for subject, sessions in subjects_trials_by_problem[problem].items():
            kind = _problem_task_type(subject, problem, sessions)
            k = seen_per_type[kind].get(subject, 0) + 1
            seen_per_type[kind][subject] = k
            numbering[(subject, problem)] = (kind, k)
    return numbering


def regroup_by_task_type(subjects_trials_by_problem: dict) -> dict:
    """``{(task_type, k): {subject: sessions}}`` from ``{problem: {subject: sessions}}``.

    The sessions are the input's own dicts (not copies) and keep their ``"problem"``, the
    mouse's overall problem number. Keys are sorted: blocked before drifting, then by ``k``.
    """
    numbering = number_within_task_type(subjects_trials_by_problem)
    grouped: dict[tuple[str, int], dict] = {}
    for problem in sorted(subjects_trials_by_problem):
        for subject, sessions in subjects_trials_by_problem[problem].items():
            grouped.setdefault(numbering[(subject, problem)], {})[subject] = sessions
    return dict(sorted(grouped.items()))


def problem_map_table(subjects_trials_by_problem: dict) -> pd.DataFrame:
    """One row per mouse and problem: its type, number within the type and overall number.

    Columns are :data:`MAP_COLUMNS`; ``sessions`` lists the session keys in session order,
    separated by ``;``.
    """
    numbering = number_within_task_type(subjects_trials_by_problem)
    rows = [
        {
            "mouse": subject,
            "task_type": kind,
            "problem_in_type": k,
            "overall_problem": problem,
            "sessions": ";".join(ordered_session_keys(subject, sessions)),
        }
        for (subject, problem), (kind, k) in numbering.items()
        for sessions in [subjects_trials_by_problem[problem][subject]]
    ]
    table = pd.DataFrame(rows, columns=list(MAP_COLUMNS))
    return table.sort_values(["mouse", "overall_problem"], ignore_index=True)
