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
    DEFAULT_INITIATION_TOWER,
    DRIFTING,
    classify_task_name,
)
from src.behavior_import.session_order import (
    compress_numbers,
    ordered_session_keys,
    session_number,
)

MAP_COLUMNS = (
    "mouse",
    "task_type",
    "problem_in_type",
    "overall_problem",
    "sessions",
    "initiation_tower",
    "choice_towers",
    "n_sessions",
    "session_range",
)


def session_task_type(subject: str, session_key: str, session: dict) -> str:
    """:func:`classify_task_name` of one session; errors name the subject, session and file.

    Args:
        subject (str): Subject id, named in the error message.
        session_key (str): Session key, named in the error message.
        session (dict): Session dict carrying the importer's ``"task_name"``.

    Returns:
        str: ``"blocked"`` or ``"drifting"``.

    Raises:
        ValueError: If the task name is missing, unrecognised or matches both types.
    """
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
        subjects_trials_by_problem (dict): ``{problem: {subject: {session_key: session_dict}}}``,
            every session carrying the importer's ``"task_name"``.

    Returns:
        dict[tuple[str, int], tuple[str, int]]: ``{(subject, overall_problem): (task_type, k)}``
        with ``task_type`` ``"blocked"`` or ``"drifting"`` and ``k`` counting from 1.

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


def regroup_by_task_type(subjects_trials_by_problem: dict, numbering: dict | None = None) -> dict:
    """``{(task_type, k): {subject: sessions}}`` from ``{problem: {subject: sessions}}``.

    The sessions are the input's own dicts (not copies) and keep their ``"problem"``, the
    mouse's overall problem number. Keys are sorted: blocked before drifting, then by ``k``.

    Args:
        subjects_trials_by_problem (dict): ``{problem: {subject: {session_key: session_dict}}}``.
        numbering (dict | None): The result of :func:`number_within_task_type` of the same
            data, to avoid computing it again. Default ``None``: compute it.

    Returns:
        dict[tuple[str, int], dict]: ``{(task_type, k): {subject: {session_key: session_dict}}}``.

    Raises:
        ValueError: As :func:`number_within_task_type`, when *numbering* is not given.
    """
    if numbering is None:
        numbering = number_within_task_type(subjects_trials_by_problem)
    grouped: dict[tuple[str, int], dict] = {}
    for problem in sorted(subjects_trials_by_problem):
        for subject, sessions in subjects_trials_by_problem[problem].items():
            grouped.setdefault(numbering[(subject, problem)], {})[subject] = sessions
    return dict(sorted(grouped.items()))


def problem_map_table(
    subjects_trials_by_problem: dict, numbering: dict | None = None
) -> pd.DataFrame:
    """One row per mouse and problem: its type, number within the type and overall number.

    Columns are :data:`MAP_COLUMNS`; ``sessions`` lists the session keys in session order,
    separated by ``;``. ``initiation_tower`` and ``choice_towers`` (sorted, separated by ``;``)
    are the layout of the problem's first session; the sessions of a problem share one layout.
    ``n_sessions`` counts the sessions and ``session_range`` gives their numbers as ranges, for
    example ``"1-10"``, or ``"3-8, 20"`` when the mouse was run on other settings in between
    and the problem resumed later. *numbering* is as for :func:`regroup_by_task_type`.

    Args:
        subjects_trials_by_problem (dict): ``{problem: {subject: {session_key: session_dict}}}``.
        numbering (dict | None): The result of :func:`number_within_task_type` of the same
            data. Default ``None``: compute it.

    Returns:
        pd.DataFrame: One row per mouse and problem, sorted by ``mouse`` then
        ``overall_problem``; columns ``mouse`` (str), ``task_type`` (str), ``problem_in_type``
        (int), ``overall_problem`` (int), ``sessions`` (str, ``;``-separated keys),
        ``initiation_tower`` (str; the protocol default ``B2`` if the session logged none),
        ``choice_towers`` (str), ``n_sessions`` (int) and ``session_range`` (str).
    """
    if numbering is None:
        numbering = number_within_task_type(subjects_trials_by_problem)
    rows = [
        _map_row(subject, kind, k, problem, subjects_trials_by_problem[problem][subject])
        for (subject, problem), (kind, k) in numbering.items()
    ]
    table = pd.DataFrame(rows, columns=list(MAP_COLUMNS))
    return table.sort_values(["mouse", "overall_problem"], ignore_index=True)


def _choice_ports(sessions) -> set[str]:
    """The union of the sessions' choice ports.

    A session holds them as one set (just imported) or as the list of the towers on offer in
    each trial (after trial extraction).
    """
    ports: set[str] = set()
    for session in sessions:
        for item in session.get("choice_towers") or ():
            if item is not None:  # a trial that did not log its towers
                ports.update([item] if isinstance(item, str) else item)
    return ports


def _map_row(subject: str, kind: str, k: int, problem: int, sessions: dict) -> dict:
    keys = ordered_session_keys(subject, sessions)
    first = sessions[keys[0]]
    return {
        "mouse": subject,
        "task_type": kind,
        "problem_in_type": k,
        "overall_problem": problem,
        "sessions": ";".join(keys),
        "initiation_tower": first.get("initiation_tower") or DEFAULT_INITIATION_TOWER,
        "choice_towers": ";".join(sorted(_choice_ports(sessions.values()))),
        "n_sessions": len(keys),
        "session_range": compress_numbers(session_number(key) for key in keys),
    }
