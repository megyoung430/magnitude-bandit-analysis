"""A table of sessions where a mouse may have been run on incorrect settings.

Three kinds of row, told apart by ``comparison``:

``problem numbering``
    The problem numbering (:mod:`src.behavior_import.problem_numbering`) flagged the session.
    Here the one-session layout A in "A A B A" is a *suspected settings error* and the sessions
    on either side are pooled; this lists the sessions the numbering treated that way, and the
    other outcomes it only reports as warnings: ``review_settings`` (a new problem whose first
    session does not look like a problem start), ``provisional`` (the number of a final
    one-session run, which a later session can turn into a settings error), ``port_finding``,
    ``pretraining`` and ``no_choice_towers`` (sessions left out of the problems). ``flag`` names
    the outcome and ``reason`` explains it in words.

``run_start vs run_end`` and ``run_end vs next run_start``
    A configuration setting differs between the ``run_start`` and the ``run_end`` of one
    recording, or between the ``run_end`` of one recording and the ``run_start`` of the next
    recording of the same session. Each differing setting is one row, with ``key``,
    ``value_before`` and ``value_after``. Only the sessions that are in a problem are compared.
    Differences between sessions are not settings changes of a session and are not listed.

``not checked``
    A recording has no ``run_start`` or no ``run_end`` (the task crashed or was stopped before
    logging it), so its settings cannot be compared. One row per such recording, with the missing
    snapshot named in ``reason``.

Settings are compared through :data:`SETTINGS_KEYS` only. Every other key the task logs is in
:data:`NOT_SETTINGS`, with the reason it is not compared: counters and totals, state that
changes as the task runs (current magnitudes and arms, the last poke, choice and rank counts,
``with_init_rew`` switched off after the initial rewards, ``can_switch``) and the solenoid
calibration (``hw_solenoid_calibration``, and the drifting task's shared intercept ``i`` and
slope ``s``), which the experimenter updates between sessions. A key logged on one side only
(files from another task version) is not compared.

One change is expected and not listed: ``found_ports`` going ``False`` to ``True`` during a
recording of a problem's first session, which is port finding at the start of a new problem. In
any other session, or the other way round, it is listed.

Columns: see :data:`COLUMNS`. ``file`` is the recording of the ``run_start`` (or of the
``run_end`` for the between-recordings comparison) and ``file_after`` the next recording of the
between-recordings comparison. ``task_type``, ``problem_in_type`` and ``overall_problem`` place
the session as in :meth:`~src.behavior_analysis.review_context.ProblemIndex.columns`.
"""

import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from src.behavior_analysis.review_context import ProblemIndex, typed_frame
from src.behavior_import.import_data import session_source_paths
from src.behavior_import.import_log import SessionFlag
from src.behavior_import.problem_numbering import PROVISIONAL_REASON
from src.behavior_import.session_order import ordered_session_keys, session_sort_key

# The configuration keys of a recording's run_start / run_end, grouped by what they set. Chosen
# by comparing the run_start and run_end of every staged recording (drifting and blocked
# tasks): these never change while the task runs unless someone changes the setting.
SETTINGS_KEY_GROUPS = {
    "timing and thresholds": (
        "ITI_dur",
        "init_cue_dur",
        "no_rew_cue_dur",
        "pulse_isi",
        "poke_hold_dur",
        "min_hold",
        "ses_length",
        "found_t_thresh",
        "t_thresh",
    ),
    "reward": ("base_choice_vol", "init_rew_vol", "num_init_rew", "rew_mags", "rew_mag"),
    "block structure, drifting task": ("min_block_len", "max_block_len", "p_ud", "p_bi"),
    "block structure, blocked task": (
        "perf_thresh",
        "second_perf_thresh",
        "num_back_b",
        "num_back_s",
        "best_val",
        "switch_best",
    ),
    "cues and options": ("led", "with_LED", "change_chosen"),
    "problem": ("num_problem", "problems"),
    "port finding": ("found_ports",),
}
SETTINGS_KEYS = tuple(key for keys in SETTINGS_KEY_GROUPS.values() for key in keys)
# The other keys of the staged run_start / run_end rows, with the reason each is not compared.
NOT_SETTINGS = {
    "counters and totals": (
        "num_t",
        "num_t_in_block",
        "num_t_found",
        "num_t_rev",
        "num_t_switch",
        "num_blocks",
        "num_good_rev",
        "num_bad_rev",
        "tot_num_t",
        "tot_num_t_switch",
        "tot_num_blocks",
        "tot_num_good_rev",
        "tot_num_bad_rev",
        "tot_rew_amt",
        "num_rew_pulses",
        "pulses_deliv",
        "num_long_pokes",
        "long_pokes",
        "found_port_counts",
        "rank_counts",
    ),
    "changes as the task runs": (
        "curr_block_len",
        "curr_poke",
        "curr_rew_mag",
        "arm_to_stream",
        "inactive_arm",
        "stream_hold",
        "stream_state",
        "t_ses_start",
        "with_init_rew",
        "can_switch",
    ),
    "solenoid calibration": ("hw_solenoid_calibration", "i", "s"),
}

NUMBERING = "problem numbering"
NOT_CHECKED = "not checked"
WITHIN_RECORDING = "run_start vs run_end"
BETWEEN_RECORDINGS = "run_end vs next run_start"

_DTYPES = {
    "mouse": "object",
    "session": "object",
    "file": "object",
    "file_after": "object",
    "task_type": "object",
    "problem_in_type": "Int64",
    "overall_problem": "Int64",
    "comparison": "object",
    "flag": "object",
    "key": "object",
    "value_before": "object",
    "value_after": "object",
    "reason": "object",
}
COLUMNS = tuple(_DTYPES)


def read_snapshots(frame: pd.DataFrame) -> tuple[dict | None, dict | None]:
    """The first ``run_start`` and the last ``run_end`` of one recording.

    Args:
        frame (pd.DataFrame): A recording with ``type``, ``subtype`` and ``content`` columns,
            ``content`` as the raw JSON string.

    Returns:
        tuple[dict | None, dict | None]: The parsed rows, ``None`` for a row the recording
        does not have. A ``run_end`` that is not valid JSON counts as missing: a crash can cut
        the last row short.

    Raises:
        ValueError: If the first ``run_start`` is not valid JSON.
    """
    variables = frame.loc[frame["type"] == "variable"]
    starts = variables.loc[variables["subtype"] == "run_start", "content"]
    ends = variables.loc[variables["subtype"] == "run_end", "content"]
    try:
        start = json.loads(starts.iloc[0]) if len(starts) else None
    except ValueError as err:
        raise ValueError(f"run_start is not valid JSON: {starts.iloc[0]!r}") from err
    try:
        end = json.loads(ends.iloc[-1]) if len(ends) else None
    except ValueError:  # deliberate: a crashed recording can end mid-row
        end = None
    return start, end


def setting_changes(before: dict, after: dict, keys: Iterable[str] = SETTINGS_KEYS) -> list:
    """The keys, among *keys*, that both snapshots log and that have a different value.

    Args:
        before (dict): The earlier snapshot.
        after (dict): The later snapshot.
        keys (Iterable[str]): The keys to compare. Default :data:`SETTINGS_KEYS`.

    Returns:
        list[tuple]: ``(key, value before, value after)`` for each changed key, in the order of
        *keys*.
    """
    return [
        (k, before[k], after[k])
        for k in keys
        if k in before and k in after and before[k] != after[k]
    ]


def _text(value) -> str:
    return json.dumps(value) if isinstance(value, (dict, list)) else str(value)


def _is_port_finding(key: str, before, after) -> bool:
    return key == "found_ports" and before is False and after is True


def _file_name(paths: list[str], i: int) -> str | None:
    return Path(paths[i]).name if i < len(paths) else None


def _change_rows(
    comparison: str, changes: list, expected_port_finding: bool, files: tuple
) -> list[dict]:
    rows = []
    for key, before, after in changes:
        if expected_port_finding and _is_port_finding(key, before, after):
            continue
        rows.append(
            {
                "file": files[0],
                "file_after": files[1],
                "comparison": comparison,
                "key": key,
                "value_before": _text(before),
                "value_after": _text(after),
                "reason": f"setting {key!r} changed from {_text(before)} to {_text(after)} "
                + (
                    "within a recording"
                    if comparison == WITHIN_RECORDING
                    else "between two recordings"
                ),
            }
        )
    return rows


def _not_checked_row(start: dict | None, end: dict | None, file: str | None) -> dict | None:
    """The row for a recording that lacks a snapshot, ``None`` if it has both."""
    missing = [name for name, snap in (("run_start", start), ("run_end", end)) if snap is None]
    if not missing:
        return None
    return {
        "file": file,
        "file_after": None,
        "comparison": NOT_CHECKED,
        "reason": "not checked: no " + ", no ".join(missing),
    }


def session_setting_rows(sess: dict, *, first_of_problem: bool) -> list[dict]:
    """The settings that changed within one session, as table rows without the session columns.

    Args:
        sess (dict): A session with its raw recordings under ``"df"``.
        first_of_problem (bool): Whether this is the first session of its problem, where
            port finding (``found_ports`` ``False`` to ``True``) is expected.

    Returns:
        list[dict]: One row per changed setting, recording by recording: the change inside the
        recording, then the change into the next one; a recording without a ``run_start`` or
        ``run_end`` gets a ``not checked`` row.
    """
    frames = sess["df"] if isinstance(sess["df"], list) else [sess["df"]]
    paths = session_source_paths(sess)
    snapshots = [read_snapshots(frame) for frame in frames]
    rows = []
    for i, (start, end) in enumerate(snapshots):
        unchecked = _not_checked_row(start, end, _file_name(paths, i))
        if unchecked:
            rows.append(unchecked)
        if start is not None and end is not None:
            files = (_file_name(paths, i), None)
            rows += _change_rows(
                WITHIN_RECORDING, setting_changes(start, end), first_of_problem, files
            )
        following = snapshots[i + 1][0] if i + 1 < len(snapshots) else None
        if end is not None and following is not None:
            files = (_file_name(paths, i), _file_name(paths, i + 1))
            rows += _change_rows(BETWEEN_RECORDINGS, setting_changes(end, following), False, files)
    return rows


def _flag_rows(flags: Iterable[SessionFlag]) -> list[dict]:
    rows = []
    for item in flags:
        common = {
            "mouse": item.mouse,
            "session": item.session,
            "file": Path(item.files[0]).name if item.files else None,
            "comparison": NUMBERING,
        }
        if item.flag:
            rows.append({**common, "flag": item.flag, "reason": item.reason})
        if item.provisional:
            rows.append({**common, "flag": "provisional", "reason": PROVISIONAL_REASON})
    return rows


def suspect_sessions_table(
    flags: list[SessionFlag], grouped: dict, index: ProblemIndex
) -> pd.DataFrame:
    """The numbering flags and the settings changes within sessions, one row each.

    Args:
        flags (list[SessionFlag]): ``ImportLog.flags`` of the cohort's import.
        grouped (dict): ``{(task_type, k): {mouse: {session_key: session_dict}}}`` from
            :func:`~src.behavior_import.task_types.regroup_by_task_type`; the sessions carry their
            raw recordings (``"df"``).
        index (ProblemIndex): Where each session sits in its mouse's problems.

    Returns:
        pd.DataFrame: The columns of :data:`COLUMNS`, ordered by mouse and session (numbering
        rows before settings rows of one session); empty (with the columns) if nothing is
        suspect.

    Raises:
        ValueError: If a recording's first ``run_start`` is not valid JSON.
    """
    rows = [(row, 0) for row in _flag_rows(flags)]
    for mice in grouped.values():
        for mouse, sessions in mice.items():
            first = ordered_session_keys(mouse, sessions)[0]
            for key in sessions:
                found = session_setting_rows(sessions[key], first_of_problem=key == first)
                rows += [({"mouse": mouse, "session": key, **row}, 1) for row in found]
    placed = [{**row, **index.columns(row["mouse"], row["session"])} for row, _ in rows]
    order = sorted(
        range(len(rows)),
        key=lambda i: (rows[i][0]["mouse"], session_sort_key(rows[i][0]["session"]), rows[i][1], i),
    )
    return typed_frame([placed[i] for i in order], _DTYPES)
