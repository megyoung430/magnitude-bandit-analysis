"""Pokes around each trial, read from the raw pyControl event log (no plotting, no file access).

Two kinds of pokes are of interest, both attributed to the trial whose ``print`` row ends it:

- **Long pokes after initiation.** A poke held until ``poke_hold_timer`` fires goes
  ``hold_poke`` -> ``poked_in``. Every such poke between the initiating poke and the choice is
  listed, the choice itself being the last one (a long poke at a choice tower is the choice).
- **Pokes after the choice.** Every ``<port>_in`` event from the trial's ``ITI`` state until the
  next trial's initiation, which is the ``wait_init`` -> ``wait_choice`` transition. The
  initiating poke is the ``_in`` event that triggered that transition and is not part of the
  window. Pokes during the rewarded ``choice`` state (which precedes the ``ITI``) are outside
  the window; an unrewarded trial goes straight to the ``ITI``.

The parsers keep everything they saw; the counting rules are separate small functions
(:func:`pokes_after_initiation`, :func:`pokes_after_choice`) so each can be tested alone.

Trials are identified by the recording's ``print`` rows, as the importer does, so the k-th
printed trial here is the k-th trial of the session's merged trial variables (checked in
:func:`parse_session`).

A trial whose pokes cannot be trusted is kept in the list with a ``drop_reason`` and no pokes, and
:func:`parse_session` warns once with the count: a trial that ended without a summary print (a
timeout or aborted trial, or a plain-text print), a held port that differs from the printed
chosen tower, a trial that went back to ``wait_init`` without an ITI. Only the last trial of a
recording may have an open after-choice window. Structurally impossible logs raise
:class:`PokeParseError`.
"""

from __future__ import annotations

import re
import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.behavior_import.extract_trials import ALIASES, bandit_trial_info
from src.behavior_import.import_data import parse_content_json
from src.behavior_import.problem_numbering import DEFAULT_INITIATION_TOWER

ARMS9 = ("A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3")

_POKE_IN_RE = re.compile(r"^([ABC][123])_in$")

NO_PRINT = "no summary print row"


class PokeParseError(ValueError):
    """The raw event log cannot be read as trials (impossible state sequence, misaligned trials)."""


class TrialPokes(NamedTuple):
    """The pokes of one trial.

    ``long_pokes`` are the ports of all long pokes from initiation to the choice, in order, the
    choice poke last. ``printed_long_pokes`` is the trial print's own ``long_pokes`` list (the
    non-choice long pokes), or ``None`` if the print has no such list. ``window_pokes`` are the
    ports of every ``_in`` event between the trial's ``ITI`` state and the next initiation, minus
    the initiating poke; it is ``None`` when that window was never closed (the recording ended
    after the trial).

    ``drop_reason`` is ``None`` for a trial to analyse; otherwise the trial's pokes are not to be
    used (``long_pokes`` is empty, ``window_pokes`` ``None``) and the reason says why.
    ``has_print`` is False for a trial that ended without any print row (it is not one of the
    session's trials); ``chosen`` is ``None`` when there is no usable print.
    """

    chosen: str | None
    printed_long_pokes: tuple[str, ...] | None
    long_pokes: tuple[str, ...]
    window_pokes: tuple[str, ...] | None
    drop_reason: str | None = None
    has_print: bool = True


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
        raise PokeParseError(f"{label}: event times are not in increasing order")


@dataclass
class _Trial:
    chosen: str | None
    printed: tuple[str, ...] | None
    long_pokes: tuple[str, ...]
    window: tuple[str, ...] | None = None
    drop_reason: str | None = None
    has_print: bool = True

    def dropped(self, reason: str) -> None:
        self.drop_reason, self.long_pokes, self.window = reason, (), None


class _RecordingState:
    """The mutable position in a recording while its rows are read in order."""

    def __init__(self, initiation_tower: str, label: str):
        self.initiation_tower = initiation_tower
        self.label = label
        self.state: str | None = None
        self.last_in: str | None = None
        self.hold_port: str | None = None
        self.initiated = False  # a trial was initiated and has had neither a print nor an ITI
        self.long_pokes: list[str] = []
        self.trials: list[_Trial] = []
        self.awaiting_window = False  # the last trial's after-choice window is not yet closed
        self.window: list[str] | None = None  # the open window's pokes, once its ITI began

    def fail(self, message: str) -> PokeParseError:
        return PokeParseError(f"{self.label}: {message}")

    def enter_state(self, name: str) -> None:
        previous, self.state = self.state, name
        if name == "hold_poke":
            self.hold_port = self.last_in
        elif name == "poked_in":
            if previous != "hold_poke" or self.hold_port is None:
                raise self.fail("state poked_in is not preceded by hold_poke")
            self.long_pokes.append(self.hold_port)
        elif name == "ITI":
            self.enter_iti()
        elif name == "wait_choice" and previous == "wait_init":
            self.initiate()

    def enter_iti(self) -> None:
        if self.awaiting_window:
            if self.window is not None:
                raise self.fail("a second ITI state before the next initiation")
            self.window = []
        elif self.initiated:  # initiated, then the ITI with no print: a timeout or abort
            self.trials.append(_Trial(None, None, (), None, NO_PRINT, has_print=False))
            self.reset_trial()

    def reset_trial(self) -> None:
        self.initiated = False
        self.long_pokes = []

    def initiate(self) -> None:
        if self.last_in != self.initiation_tower:
            raise self.fail(
                f"initiation (wait_init -> wait_choice) was triggered by {self.last_in!r}, "
                f"not the initiation tower {self.initiation_tower!r}"
            )
        if self.awaiting_window:
            self.close_window()
        self.initiated = True
        self.long_pokes = []

    def close_window(self) -> None:
        last = self.trials[-1]
        if self.window is None:
            last.dropped("went back to wait_init with no ITI before the next initiation")
        elif not self.window:
            raise self.fail("initiating poke missing from the after-choice window")
        else:
            self.window.pop()  # the initiating poke itself
            last.window = tuple(self.window)
        self.window, self.awaiting_window = None, False

    def poke_in(self, port: str) -> None:
        self.last_in = port
        if self.window is not None:
            self.window.append(port)

    def trial_printed(self, printed: dict) -> None:
        if self.awaiting_window:  # a print with no initiation after the previous trial
            self.trials[-1].dropped("next trial printed without an initiation")
            self.window, self.awaiting_window = None, False
        chosen = _first_alias(printed, ALIASES["choice"])
        if chosen is None:
            self.trials.append(_Trial(None, None, (), None, NO_PRINT))
            self.reset_trial()
            return
        chosen = str(chosen)
        printed_list = _first_alias(printed, ALIASES["long_pokes"])
        trial = _Trial(
            chosen,
            None if printed_list is None else tuple(str(p) for p in printed_list),
            tuple(self.long_pokes),
        )
        if not self.initiated or not self.long_pokes:
            trial.dropped("printed without an initiation and a choice poke")
        elif self.long_pokes[-1] != chosen:
            trial.dropped(
                f"choice poke was held at {self.long_pokes[-1]} but the print says {chosen}"
            )
        else:
            self.awaiting_window = True
        self.trials.append(trial)
        self.reset_trial()


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
        One :class:`TrialPokes` per trial (a trial with a print row, or one that ended in an
        ITI without any), in order; ``[]`` for a recording without bandit trials (port finding,
        pretraining). Untrustworthy trials carry a ``drop_reason`` (see the module doc).

    Raises:
        PokeParseError: If the log is not in time order, a state sequence is impossible
            (``poked_in`` without ``hold_poke``, two ITIs in a row), or a trial is initiated by
            a port other than *initiation_tower*.
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
        TrialPokes(t.chosen, t.printed, t.long_pokes, t.window, t.drop_reason, t.has_print)
        for t in position.trials
    ]


def _recordings(session: dict) -> list[pd.DataFrame]:
    if "df" not in session:
        raise PokeParseError(
            "session has no raw 'df' recordings; the poke analysis reads the raw event log"
        )
    frames = session["df"]
    return [frames] if isinstance(frames, pd.DataFrame) else list(frames)


def _check_against_session_choices(trials: list[TrialPokes], choices: Sequence, label: str) -> None:
    printed = [t for t in trials if t.has_print]
    if len(printed) != len(choices):
        raise PokeParseError(
            f"{label}: parsed {len(printed)} trials from the raw events but the session lists "
            f"{len(choices)} choices"
        )
    for k, (trial, choice) in enumerate(zip(printed, choices, strict=True), start=1):
        if trial.chosen != choice:
            raise PokeParseError(
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

    Warns once with the number of dropped trials (see the module doc).

    Raises:
        PokeParseError: As :func:`parse_recording`, if the session has no ``"df"``, or if the
            parsed trials do not match the session's ``"choice"`` list.
    """
    initiation = initiation_tower or session.get("initiation_tower") or DEFAULT_INITIATION_TOWER
    trials: list[TrialPokes] = []
    for i, frame in enumerate(_recordings(session)):
        trials.extend(parse_recording(frame, initiation, label=f"{label} recording {i}"))
    if "choice" in session:
        _check_against_session_choices(trials, session["choice"], label)
    _warn_about_dropped(trials, label)
    return trials


def _warn_about_dropped(trials: Sequence[TrialPokes], label: str) -> None:
    reasons = Counter(t.drop_reason for t in trials if t.drop_reason is not None)
    if reasons:
        total = sum(reasons.values())
        listed = "; ".join(f"{n} x {reason}" for reason, n in reasons.items())
        warnings.warn(
            f"{label}: dropped {total} trial{'s' if total > 1 else ''} from the poke analysis "
            f"({listed})",
            UserWarning,
            stacklevel=3,
        )


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
    chosen tower; trials whose print has no list disagree. Dropped trials are not compared.
    """
    trials = [t for t in trials if t.drop_reason is None]
    mismatches = []
    for k, trial in enumerate(trials, start=1):
        printed = (
            None if trial.printed_long_pokes is None else (*trial.printed_long_pokes, trial.chosen)
        )
        if printed != trial.long_pokes:
            mismatches.append((k, trial.long_pokes, printed))
    return LongPokeCheck(len(trials), len(trials) - len(mismatches), mismatches)
