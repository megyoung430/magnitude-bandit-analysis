"""When the outcome of each trial reached the mouse, read from one pyControl event log.

Each trial ends in a summary ``print`` row. What follows it in the log depends on the outcome:

- **Rewarded** (``num_rew_pulses > 0``): the task enters ``choice`` within a few ms of the print.
  The first reward pulse starts on entry to ``choice``, so that entry is the outcome time
  (``reward_onset``). The first ``pulse_off_timer`` after it ends pulse 1; the notebook
  compares that delay with the logged ``pulse_isi``.
- **Unrewarded**: there is no ``choice`` state; the task goes straight to ``ITI``, which starts
  the no-reward cue. ``ITI`` entry is the outcome time (``no_reward_cue_onset``). The
  ``no_rew_cue_timer`` fires ``no_rew_cue_dur`` later, so it marks the cue's *offset*; it is
  kept as ``check_time`` to verify the onset against the logged cue duration.

Only rows between a print and the next print belong to that trial, so a missing state is never
filled in from the following trial. A state that is missing, logged twice, or logged more than
100 ms after the print means the log does not have the structure the definitions rely on, and
raises with the trial number and file.

The parsers are pure functions of a DataFrame; :func:`load_outcome_events` is the file edge.
"""

import numpy as np
import pandas as pd

from src.behavior_import.import_data import (
    drop_non_numeric_times,
    parse_content_json,
    read_tsv_file,
    source_path,
)

OUTCOME_COLUMNS = [
    "trial",
    "phase",
    "num_t",
    "num_t_found",
    "chosen_tow",
    "chosen_rank",
    "num_rew_pulses",
    "rewarded",
    "print_time",
    "outcome_time",
    "outcome_kind",
    "check_time",
]

# Longest delay (s) between the print and the state that starts the outcome.
MAX_STATE_LAG_S = 0.1

REWARD_ONSET = "reward_onset"
NO_REWARD_CUE_ONSET = "no_reward_cue_onset"


class OutcomeParseError(ValueError):
    """The event log does not have the structure the outcome definitions rely on."""


def _clean(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """*df* without traceback rows (non-numeric ``time``), and the file it came from."""
    label = source_path(df)
    return drop_non_numeric_times(df, label), label


def _print_payload(content, where: str) -> dict:
    """The fields of a print row, validated; *where* names the trial and file."""
    payload = parse_content_json(content)
    if not isinstance(payload, dict) or not payload:
        raise OutcomeParseError(f"{where}: the print row is not a JSON object.")
    has_task, has_found = "num_t" in payload, "num_t_found" in payload
    if has_task and has_found:
        raise OutcomeParseError(f"{where}: the print has both num_t and num_t_found.")
    if not (has_task or has_found):
        raise OutcomeParseError(f"{where}: the print has neither num_t nor num_t_found.")
    for key in ("chosen_tow", "num_rew_pulses"):
        if key not in payload:
            raise OutcomeParseError(f"{where}: the print has no {key}.")
    pulses = payload["num_rew_pulses"]
    if not isinstance(pulses, int | float) or pulses < 0 or pulses != int(pulses):
        raise OutcomeParseError(f"{where}: num_rew_pulses is {pulses!r}, not a count.")
    return payload


def _state_time(
    segment: pd.DataFrame, state: str, print_time: float, where: str
) -> tuple[int, float]:
    """Position (within *segment*) and time of the one entry to *state* shortly after the print."""
    entries = np.flatnonzero(
        ((segment["type"] == "state") & (segment["content"] == state)).to_numpy()
    )
    if len(entries) == 0:
        raise OutcomeParseError(f"{where}: the {state!r} state is missing after the print.")
    if len(entries) > 1:
        raise OutcomeParseError(f"{where}: the {state!r} state appears twice after the print.")
    position = int(entries[0])
    time = float(segment["time"].iloc[position])
    if time - print_time > MAX_STATE_LAG_S:
        raise OutcomeParseError(
            f"{where}: the {state!r} state is {1000 * (time - print_time):.0f} ms after the "
            f"print, more than {1000 * MAX_STATE_LAG_S:.0f} ms."
        )
    return position, time


def _timer_time(segment: pd.DataFrame, after: int, timer: str, where: str) -> float:
    """Time of the first *timer* event in *segment* after row position *after*."""
    later = segment.iloc[after + 1 :]
    hits = later.loc[(later["type"] == "event") & (later["content"] == timer), "time"]
    if hits.empty:
        raise OutcomeParseError(f"{where}: no {timer} follows the outcome.")
    return float(hits.iloc[0])


def _trial_outcome(trial: int, print_row: pd.Series, segment: pd.DataFrame, label: str) -> dict:
    """One row of the outcome table, from a trial's print and the rows up to the next print."""
    where = f"{label}: trial {trial}"
    payload = _print_payload(print_row["content"], where)
    rewarded = payload["num_rew_pulses"] > 0
    print_time = float(print_row["time"])
    if rewarded:
        state, timer, kind = "choice", "pulse_off_timer", REWARD_ONSET
    else:
        state, timer, kind = "ITI", "no_rew_cue_timer", NO_REWARD_CUE_ONSET
        if ((segment["type"] == "state") & (segment["content"] == "choice")).any():
            raise OutcomeParseError(f"{where}: an unrewarded trial has a 'choice' state.")
    position, outcome_time = _state_time(segment, state, print_time, where)
    return {
        "trial": trial,
        "phase": "task" if "num_t" in payload else "port_finding",
        "num_t": payload.get("num_t", pd.NA),
        "num_t_found": payload.get("num_t_found", pd.NA),
        "chosen_tow": payload["chosen_tow"],
        "chosen_rank": payload.get("chosen_rank"),
        "num_rew_pulses": int(payload["num_rew_pulses"]),
        "rewarded": bool(rewarded),
        "print_time": print_time,
        "outcome_time": outcome_time,
        "outcome_kind": kind,
        "check_time": _timer_time(segment, position, timer, where),
    }


def outcome_events(df: pd.DataFrame) -> pd.DataFrame:
    """One row per trial print with the time its outcome started.

    Args:
        df (pd.DataFrame): A pyControl session as read by ``read_tsv_file``, with ``time``,
            ``type``, ``subtype`` and ``content`` columns. Traceback rows are dropped with a
            warning, as everywhere else in the repo.

    Returns:
        pd.DataFrame: Columns ``trial`` (1..n in file order), ``phase`` (``"task"`` for a print
        with ``num_t``, ``"port_finding"`` for one with ``num_t_found``), ``num_t`` and
        ``num_t_found`` (nullable integers, missing for the other phase), ``chosen_tow``,
        ``chosen_rank`` (None if the print has none), ``num_rew_pulses``, ``rewarded``,
        ``print_time``, ``outcome_time``, ``outcome_kind`` (``"reward_onset"`` or
        ``"no_reward_cue_onset"``) and ``check_time`` (see the module docstring). Times are in
        seconds on the session clock.

    Raises:
        OutcomeParseError: If a print is malformed, or the state that starts a trial's outcome
            is missing, appears twice, or comes more than 100 ms after the print; if a
            rewarded trial has no ``choice`` state or an unrewarded one has one; or if the
            timer that ends the first pulse or the cue is missing.
    """
    df, label = _clean(df)
    is_print = ((df["type"] == "variable") & (df["subtype"] == "print")).to_numpy()
    print_positions = np.flatnonzero(is_print)
    ends = np.r_[print_positions, len(df)][1:]  # each segment stops at the next print
    rows = [
        _trial_outcome(trial, df.iloc[start], df.iloc[start + 1 : end], label)
        for trial, (start, end) in enumerate(zip(print_positions, ends, strict=True), start=1)
    ]
    table = pd.DataFrame(rows, columns=OUTCOME_COLUMNS)
    for column in ("num_t", "num_t_found"):
        table[column] = table[column].astype("Int64")
    table["chosen_rank"] = (
        table["chosen_rank"].astype(object).where(table["chosen_rank"].notna(), None)
    )
    return table


def outcome_settings(df: pd.DataFrame) -> dict:
    """Durations the task logged at the start of the run, to check ``check_time`` against.

    Args:
        df (pd.DataFrame): A pyControl session as read by ``read_tsv_file``.

    Returns:
        dict: ``pulse_isi`` and ``no_rew_cue_dur``, both in milliseconds as logged. The delay
        from a rewarded trial's outcome to its ``check_time`` should be about ``pulse_isi``;
        for an unrewarded trial about ``no_rew_cue_dur``.

    Raises:
        OutcomeParseError: If the session has no ``run_start`` row, more than one, or it lacks
            either setting.
    """
    df, label = _clean(df)
    run_starts = df.loc[(df["type"] == "variable") & (df["subtype"] == "run_start"), "content"]
    if run_starts.empty:
        raise OutcomeParseError(f"{label}: the session has no run_start row.")
    if len(run_starts) > 1:
        raise OutcomeParseError(f"{label}: the session has more than one run_start row.")
    payload = parse_content_json(run_starts.iloc[0])
    settings = {}
    for key in ("pulse_isi", "no_rew_cue_dur"):
        if not isinstance(payload, dict) or key not in payload:
            raise OutcomeParseError(f"{label}: run_start has no {key}.")
        settings[key] = float(payload[key])
    return settings


def session_intervals(df: pd.DataFrame) -> tuple[float, float]:
    """Time of the first and last row of the session, the span random control times fall in.

    Args:
        df (pd.DataFrame): A pyControl session as read by ``read_tsv_file``.

    Returns:
        tuple[float, float]: ``(start, end)`` in seconds.

    Raises:
        OutcomeParseError: If the table has no rows.
    """
    df, label = _clean(df)
    if df.empty:
        raise OutcomeParseError(f"{label}: the session has no rows.")
    return float(df["time"].iloc[0]), float(df["time"].iloc[-1])


def load_outcome_events(tsv_path) -> pd.DataFrame:
    """Read a pyControl ``.tsv`` and return its :func:`outcome_events`.

    Args:
        tsv_path (str | Path): The pyControl file.

    Returns:
        pd.DataFrame: See :func:`outcome_events`; errors name *tsv_path*.
    """
    df = read_tsv_file(tsv_path)
    df.attrs["source_path"] = str(tsv_path)  # names the file in parse errors
    return outcome_events(df)
