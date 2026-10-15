"""Pokes around each trial, read from the raw pyControl event log (no plotting, no file access).

Two kinds of pokes are of interest, both attributed to the trial whose ``print`` row ends it:

- **Long pokes after initiation.** A poke held until ``poke_hold_timer`` fires goes
  ``hold_poke`` -> ``poked_in``. Every such poke between the initiating poke and the choice is
  listed, the choice itself being the last one (a long poke at a choice tower is the choice).
- **Pokes after the choice.** Every ``<port>_in`` event from the trial's ``ITI`` state until the
  next trial's initiation, which is the ``wait_init`` -> ``wait_choice`` transition. The
  initiating poke is the ``_in`` event that triggered that transition and is not part of the
  window.

The parsers keep everything they saw; the counting rules are separate small functions
(:func:`pokes_after_initiation`, :func:`pokes_after_choice`) so each can be tested alone.

Trials are identified by the recording's ``print`` rows, as the importer does, so the k-th
trial here is the k-th trial of the session's merged trial variables (checked in
:func:`parse_session`).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.behavior_import.extract_trials import ALIASES, bandit_trial_info
from src.behavior_import.import_data import parse_content_json
from src.behavior_import.problem_numbering import DEFAULT_INITIATION_TOWER

ARMS9 = ("A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3")

_POKE_IN_RE = re.compile(r"^([ABC][123])_in$")


class TrialPokes(NamedTuple):
    """The pokes of one trial.

    ``long_pokes`` are the ports of all long pokes from initiation to the choice, in order, the
    choice poke last. ``printed_long_pokes`` is the trial print's own ``long_pokes`` list (the
    non-choice long pokes), or ``None`` if the print has no such list. ``window_pokes`` are the
    ports of every ``_in`` event between the trial's ``ITI`` state and the next initiation, minus
    the initiating poke; it is ``None`` when that window was never closed (the recording ended,
    or the next initiation never came).
    """

    chosen: str
    printed_long_pokes: tuple[str, ...] | None
    long_pokes: tuple[str, ...]
    window_pokes: tuple[str, ...] | None


class LongPokeCheck(NamedTuple):
    """Raw-event long pokes against the printed ``long_pokes`` lists.

    ``mismatches`` holds ``(trial, raw, printed)`` for each disagreeing trial (1-based within the
    compared trials); ``raw`` is the raw long pokes including the choice poke and ``printed`` the
    printed non-choice list followed by the chosen tower.
    """

    n_trials: int
    n_agree: int
    mismatches: list[tuple[int, tuple[str, ...], tuple[str, ...] | None]]


def _first_alias(printed: dict, names: Sequence[str]):
    """Value of the first of *names* present in the trial print *printed*, else ``None``."""
    return next((printed[n] for n in names if n in printed), None)


def _check_time_order(times: np.ndarray, label: str) -> None:
    if np.any(np.diff(times) < 0):
        raise ValueError(f"{label}: event times are not in increasing order")


class _RecordingState:
    """The mutable position in a recording while its rows are read in order."""

    def __init__(self, initiation_tower: str, label: str):
        self.initiation_tower = initiation_tower
        self.label = label
        self.state: str | None = None
        self.last_in: str | None = None
        self.hold_port: str | None = None
        self.initiated = False
        self.long_pokes: list[str] = []
        self.trials: list[tuple[str, tuple | None, tuple]] = []
        self.windows: list[tuple[str, ...] | None] = []
        self.window: list[str] | None = None  # open window of the last trial, or None

    def fail(self, message: str) -> ValueError:
        return ValueError(f"{self.label}: {message}")

    def enter_state(self, name: str) -> None:
        previous, self.state = self.state, name
        if name == "hold_poke":
            self.hold_port = self.last_in
        elif name == "poked_in":
            if previous != "hold_poke" or self.hold_port is None:
                raise self.fail("state poked_in is not preceded by hold_poke")
            self.long_pokes.append(self.hold_port)
        elif name == "ITI":
            self.open_window()
        elif name == "wait_choice" and previous == "wait_init":
            self.initiate()

    def open_window(self) -> None:
        if not self.trials:
            return  # an ITI with no printed trial to attach to
        if self.window is not None or self.windows[-1] is not None:
            raise self.fail("a second ITI state before the next initiation")
        self.window = []

    def initiate(self) -> None:
        if self.last_in != self.initiation_tower:
            raise self.fail(
                f"initiation (wait_init -> wait_choice) was triggered by {self.last_in!r}, "
                f"not the initiation tower {self.initiation_tower!r}"
            )
        if self.window is not None:
            if not self.window:
                raise self.fail("initiating poke missing from the after-choice window")
            self.window.pop()  # the initiating poke itself
            self.windows[-1] = tuple(self.window)
            self.window = None
        self.initiated = True
        self.long_pokes = []

    def poke_in(self, port: str) -> None:
        self.last_in = port
        if self.window is not None:
            self.window.append(port)

    def trial_printed(self, printed: dict) -> None:
        chosen = _first_alias(printed, ALIASES["choice"])
        if chosen is None:
            raise self.fail(f"trial print without a chosen tower: {printed}")
        if not self.initiated or not self.long_pokes:
            raise self.fail(
                f"trial {len(self.trials) + 1} printed without an initiation and a choice poke"
            )
        printed_list = _first_alias(printed, ALIASES["long_pokes"])
        self.trials.append(
            (
                str(chosen),
                None if printed_list is None else tuple(str(p) for p in printed_list),
                tuple(self.long_pokes),
            )
        )
        self.windows.append(None)
        self.initiated = False
        self.long_pokes = []


def parse_recording(
    df: pd.DataFrame,
    initiation_tower: str = DEFAULT_INITIATION_TOWER,
    *,
    label: str = "recording",
) -> list[TrialPokes]:
    """The pokes of every bandit trial printed in one raw recording.

    Args:
        df: The raw four-column pyControl DataFrame (``time``, ``type``, ``subtype``,
            ``content``), in log order.
        initiation_tower: Port whose ``_in`` starts a trial, e.g. ``"B2"``.
        label: Names the recording in error messages.

    Returns:
        One :class:`TrialPokes` per trial print, in order; ``[]`` for a recording without
        bandit trials (port finding, pretraining).

    Raises:
        ValueError: If the log is not in time order, a state sequence is impossible
            (``poked_in`` without ``hold_poke``, two ITIs in a row, a trial print without a
            choice poke), or a trial is initiated by a port other than *initiation_tower*.
    """
    if not bandit_trial_info(df, label, label):
        return []
    _check_time_order(df["time"].to_numpy(dtype=float), label)
    position = _RecordingState(initiation_tower, label)
    rows = zip(df["type"], df["subtype"], df["content"], strict=True)
    for row_type, subtype, content in rows:
        if row_type == "state":
            position.enter_state(content)
        elif row_type == "event" and subtype == "input":
            match = _POKE_IN_RE.match(str(content))
            if match:
                position.poke_in(match[1])
        elif row_type == "variable" and subtype == "print":
            printed = parse_content_json(content)
            if isinstance(printed, dict) and "num_t_found" not in printed:
                position.trial_printed(printed)
    return [
        TrialPokes(chosen, printed, long, window)
        for (chosen, printed, long), window in zip(position.trials, position.windows, strict=True)
    ]


def _recordings(session: dict) -> list[pd.DataFrame]:
    if "df" not in session:
        raise ValueError(
            "session has no raw 'df' recordings; the poke analysis reads the raw event log"
        )
    frames = session["df"]
    return [frames] if isinstance(frames, pd.DataFrame) else list(frames)


def _check_against_session_choices(trials: list[TrialPokes], choices: Sequence, label: str) -> None:
    if len(trials) != len(choices):
        raise ValueError(
            f"{label}: parsed {len(trials)} trials from the raw events but the session lists "
            f"{len(choices)} choices"
        )
    for k, (trial, choice) in enumerate(zip(trials, choices, strict=True), start=1):
        if trial.chosen != choice:
            raise ValueError(
                f"{label}: trial {k} was {trial.chosen!r} in the raw print but the session's "
                f"merged choice is {choice!r}; trials are misaligned"
            )


def parse_session(
    session: dict, initiation_tower: str | None = None, *, label: str = "session"
) -> list[TrialPokes]:
    """:func:`parse_recording` over all of a session's recordings, trials in recording order.

    The trials line up with the session's merged trial variables (the importer's
    ``merge_trial_variables``): when the session holds a ``"choice"`` list, the trial count and
    every chosen tower are checked against it.

    Args:
        session: A session record with the importer's ``"df"`` (a DataFrame or a list of them).
        initiation_tower: Defaults to the session's ``"initiation_tower"``, or the protocol
            default ``"B2"`` when the session has none.
        label: Names the session in error messages.

    Raises:
        ValueError: As :func:`parse_recording`, if the session has no ``"df"``, or if the parsed
            trials do not match the session's ``"choice"`` list.
    """
    initiation = initiation_tower or session.get("initiation_tower") or DEFAULT_INITIATION_TOWER
    trials: list[TrialPokes] = []
    for i, frame in enumerate(_recordings(session)):
        trials.extend(parse_recording(frame, initiation, label=f"{label} recording {i}"))
    if "choice" in session:
        _check_against_session_choices(trials, session["choice"], label)
    return trials


def pokes_after_initiation(long_pokes: Sequence[str], initiation_tower: str) -> list[str]:
    """Long pokes to count after initiation: leading initiation-tower pokes are dropped.

    Long pokes at the initiation tower count only once the mouse has made a long poke at
    another port; the choice poke is a long poke like any other.
    """
    start = next((i for i, port in enumerate(long_pokes) if port != initiation_tower), None)
    return [] if start is None else list(long_pokes[start:])


def pokes_after_choice(window_pokes: Sequence[str] | None, chosen: str) -> list[str]:
    """Pokes to count after the choice: leading pokes at the chosen tower are dropped.

    The mouse may be drinking at the chosen port, so its pokes there count only after a poke
    at some other port. Repeats are kept.

    Raises:
        ValueError: If *window_pokes* is ``None`` (the window was never closed).
    """
    if window_pokes is None:
        raise ValueError("the after-choice window was never closed; it has no pokes to count")
    start = next((i for i, port in enumerate(window_pokes) if port != chosen), None)
    return [] if start is None else list(window_pokes[start:])


def cross_check_long_pokes(trials: Sequence[TrialPokes]) -> LongPokeCheck:
    """Compare the raw long pokes of each trial with its printed ``long_pokes`` list.

    Agreement means the raw long pokes equal the printed non-choice long pokes followed by the
    chosen tower; trials whose print has no list disagree.
    """
    mismatches = []
    for k, trial in enumerate(trials, start=1):
        printed = (
            None if trial.printed_long_pokes is None else (*trial.printed_long_pokes, trial.chosen)
        )
        if printed != trial.long_pokes:
            mismatches.append((k, trial.long_pokes, printed))
    return LongPokeCheck(len(trials), len(trials) - len(mismatches), mismatches)
