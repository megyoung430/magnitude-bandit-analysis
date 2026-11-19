"""Which sessions of each problem were imaged, and whether the LED was on.

A session is *imaged* when its folder has both a ``behav`` and a ``funcimg`` subfolder. For each
mouse and problem (the import's overall problem number, see
:mod:`src.behavior_import.problem_numbering`) this lists the imaged sessions, how many of the
problem's sessions they are, and the LED setting of each.

**LED.** The task file records the LED setting as a task variable (``led`` in the recording
task; ``with_LED`` is accepted too, see ``LED_KEYS``). Its value is read from every variable row
of a session's files that carries it: ``run_start``, ``run_end`` and any ``user_set`` change
during the session. A session whose files never carry the variable ran a task without the
setting; ``led_on`` then follows *unset_means_on* (default ``True``: the LED was always on before
the setting existed).

Pure: session dicts and tables in, tables out.
"""

import json
import re
from collections.abc import Iterable, Mapping

import pandas as pd

LED_KEYS = ("with_LED", "with_led", "led", "LED")
ON, OFF, MIXED, UNSET = "on", "off", "mixed", "not set"
SESSION_COLUMNS = ["mouse", "ses_id", "problem", "task_type", "led_setting"]
# the suffix of a session the import split by layout (see split_recording_configurations)
PART_SUFFIX = re.compile(r"_part-\d+$")


def _contents(df: pd.DataFrame) -> Iterable[dict]:
    """The JSON dicts of a raw pyControl frame's variable rows."""
    if df is None or not {"type", "content"}.issubset(df.columns):
        return
    for content in df.loc[df["type"].eq("variable"), "content"]:
        try:
            value = json.loads(content) if isinstance(content, str) else content
        except ValueError:
            continue
        if isinstance(value, dict):
            yield value


def led_values(frames) -> set[bool]:
    """Every LED value a session's raw frames record.

    Args:
        frames (pd.DataFrame | list[pd.DataFrame] | None): The session's ``"df"``: one raw frame
            per file (a single frame for a one-file session).

    Returns:
        set[bool]: The values of the first key of ``LED_KEYS`` found in each variable row; empty
        if no row carries one.
    """
    frames = frames if isinstance(frames, list) else [frames]
    values = set()
    for df in frames:
        for content in _contents(df):
            key = next((k for k in LED_KEYS if k in content), None)
            if key is not None:
                values.add(bool(content[key]))
    return values


def led_setting(frames) -> str:
    """``ON``, ``OFF``, ``MIXED`` (both during the session) or ``UNSET`` (never recorded).

    Args:
        frames (pd.DataFrame | list[pd.DataFrame] | None): As :func:`led_values`.

    Returns:
        str: The session's LED setting.
    """
    values = led_values(frames)
    if not values:
        return UNSET
    if len(values) > 1:
        return MIXED
    return ON if values.pop() else OFF


def _combined_setting(settings: pd.Series) -> str:
    """The LED setting of a session folder from those of its parts."""
    recorded = set(settings) - {UNSET}
    if not recorded:
        return UNSET
    return recorded.pop() if len(recorded) == 1 else MIXED


def session_table(trials_by_problem: Mapping[int, Mapping[str, Mapping]]) -> pd.DataFrame:
    """One row per session folder and problem: the problem's task type and the LED setting.

    A session that the import split into ``_part-NN`` segments (its files ran different
    layouts) is one folder; its parts in the same problem are one row, and a folder whose parts
    belong to two problems has a row in each.

    Args:
        trials_by_problem (Mapping): ``{overall problem: {mouse: {session key: session dict}}}``
            as ``load_cohort`` returns it.

    Returns:
        pd.DataFrame: ``SESSION_COLUMNS`` (``ses_id`` is the folder name), sorted by mouse and
        problem.
    """
    rows = [
        dict(
            mouse=mouse,
            ses_id=PART_SUFFIX.sub("", key),
            problem=problem,
            task_type=session.get("task_type"),
            led_setting=led_setting(session.get("df")),
        )
        for problem, mice in trials_by_problem.items()
        for mouse, sessions in mice.items()
        for key, session in sessions.items()
    ]
    if not rows:
        return pd.DataFrame(columns=SESSION_COLUMNS)
    table = (
        pd.DataFrame(rows)
        .groupby(["mouse", "ses_id", "problem"], as_index=False)
        .agg(task_type=("task_type", "first"), led_setting=("led_setting", _combined_setting))
    )
    return table[SESSION_COLUMNS].sort_values(["mouse", "problem", "ses_id"]).reset_index(drop=True)


def _session_list(ses_ids: pd.Series) -> str:
    """Session ids joined in session-number order (``ses-12_...`` sorts after ``ses-9_...``)."""

    def number(ses_id: str) -> float:
        digits = ses_id.split("_")[0].removeprefix("ses-")
        return int(digits) if digits.isdigit() else float("inf")

    return ", ".join(sorted(ses_ids, key=number))


def imaged_sessions(
    sessions: pd.DataFrame, modality: pd.DataFrame, unset_means_on: bool = True
) -> pd.DataFrame:
    """The sessions with both ``behav`` and ``funcimg``, with their problem and LED.

    Args:
        sessions (pd.DataFrame): :func:`session_table`.
        modality (pd.DataFrame): The notebook's modality table: ``subject_id, ses_id, date,
            has_behav, has_funcimg``.
        unset_means_on (bool): Whether a session without an LED variable counts as LED on.
            Default ``True``.

    Returns:
        pd.DataFrame: ``mouse, ses_id, date, problem, task_type, led_setting, led_on``; an
        imaged session the import has no problem for (a flagged or unreadable session) keeps a
        missing problem.
    """
    both = modality[modality["has_behav"] & modality["has_funcimg"]]
    imaged = both[["subject_id", "ses_id", "date"]].rename(columns={"subject_id": "mouse"})
    out = imaged.merge(sessions, on=["mouse", "ses_id"], how="left")
    out["led_setting"] = out["led_setting"].fillna(UNSET)
    on = out["led_setting"].eq(ON) | (out["led_setting"].eq(UNSET) & unset_means_on)
    return out.assign(led_on=on).sort_values(["mouse", "date", "ses_id"]).reset_index(drop=True)


def _marked(rows: pd.DataFrame) -> str:
    """Session ids in session-number order, a setting that is not plain on/off marked."""
    marks = {UNSET: " (not set)", MIXED: " (mixed)"}
    labels = {
        s: s + marks.get(m, "") for s, m in zip(rows["ses_id"], rows["led_setting"], strict=True)
    }
    return ", ".join(labels[s] for s in _session_list(rows["ses_id"]).split(", ") if s)


def coverage_by_problem(sessions: pd.DataFrame, imaged: pd.DataFrame) -> pd.DataFrame:
    """Per mouse and problem: how many of its sessions were imaged, and with the LED on.

    Args:
        sessions (pd.DataFrame): :func:`session_table` (every session of the problem).
        imaged (pd.DataFrame): :func:`imaged_sessions`.

    Returns:
        pd.DataFrame: ``mouse, problem, task_type, n_sessions, n_imaged, imaged_fraction,
        n_led_on, n_led_not_set, imaged_sessions, led_on_sessions, led_off_sessions``; only
        problems with at least one imaged session. ``n_led_on`` follows the ``led_on`` of
        *imaged* (sessions without the setting included when it counted them as on), and
        ``n_led_not_set`` says how many imaged sessions have no setting. In the session lists a
        session without the setting is marked ``(not set)`` and one that changed it ``(mixed)``.
    """
    totals = sessions.groupby(["mouse", "problem"], as_index=False).agg(
        task_type=("task_type", "first"), n_sessions=("ses_id", "size")
    )
    rows = []
    for (mouse, problem), mine in imaged.dropna(subset=["problem"]).groupby(["mouse", "problem"]):
        rows.append(
            dict(
                mouse=mouse,
                problem=int(problem),
                n_imaged=len(mine),
                n_led_on=int(mine["led_on"].sum()),
                n_led_not_set=int(mine["led_setting"].eq(UNSET).sum()),
                imaged_sessions=_session_list(mine["ses_id"]),
                led_on_sessions=_marked(mine[mine["led_on"]]),
                led_off_sessions=_marked(mine[~mine["led_on"]]),
            )
        )
    columns = [
        "mouse",
        "problem",
        "task_type",
        "n_sessions",
        "n_imaged",
        "imaged_fraction",
        "n_led_on",
        "n_led_not_set",
        "imaged_sessions",
        "led_on_sessions",
        "led_off_sessions",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows).merge(totals, on=["mouse", "problem"], how="left")
    out["imaged_fraction"] = out["n_imaged"] / out["n_sessions"]
    return out[columns].sort_values(["mouse", "problem"]).reset_index(drop=True)
