"""Pokes around each trial, read from the raw pyControl event log (no plotting, no file access).

Two kinds of pokes are of interest, both attributed to the trial whose ``print`` row ends it:

- **Long pokes after initiation.** A poke held until ``poke_hold_timer`` fires goes
  ``hold_poke`` -> ``poked_in``. Every such poke between the initiating poke and the choice is
  listed, the choice itself being the last one (a long poke at a choice tower is the choice).
- **Pokes after the choice.** Every ``<port>_in`` event from the trial's ``ITI`` state until the
  next trial's initiation, which is the ``wait_init`` -> ``wait_choice`` transition. The
  initiating poke is the ``_in`` event that triggered that transition; it is the last poke of
  the window, and it counts as a poke at the initiation port. Pokes during the rewarded
  ``choice`` state (which precedes the ``ITI``) are outside the window; an unrewarded trial goes
  straight to the ``ITI``.

The log is read in file order, from the ``state`` rows and the ``<port>_in`` input events only.
Timer events are ignored: pyControl can log a timer a few milliseconds before the state row that
precedes it, so the log is not strictly time-ordered, and nothing here depends on it. Input events
can be out of order too: a poke logged a millisecond after the ``ITI`` state row, although it
happened just before, is still read after it. Reading in file order puts such a poke in the
after-choice window, where a leading poke at the chosen tower is dropped anyway.

The parsers keep everything they saw; the counting rules are separate small functions
(:func:`pokes_after_initiation`, :func:`pokes_after_choice`) so each can be tested alone.

Trials are identified by the recording's ``print`` rows, as the importer does, so the k-th
printed trial here is the k-th trial of the session's merged trial variables (checked in
:func:`parse_session`).

A trial whose pokes cannot be trusted is kept in the list with a ``drop_reason`` and no pokes, and
:func:`parse_session` warns once with the count: a trial that ended without a summary print (a
timeout or aborted trial, or a plain-text print), a held port that differs from the printed
chosen tower, a trial that went back to ``wait_init`` without an ITI, a print with no
initiation since the previous trial (that trial and the previous one are both dropped). Only a
recording's last printed trial may have an open after-choice window. Structurally impossible
logs raise :class:`PokeParseError`.

Port finding. A port-finding print has ``num_t_found``, ``found_ports`` false and no ``num_t``;
a print with ``num_t_found`` that breaks the rest of that rule raises :class:`PokeParseError`.
Port-finding trials go through the same state machine as task trials (initiation, choice,
``ITI``), with the print's ``chosen_tow`` as the chosen tower, so they get the same two kinds of
pokes. They stay in the list at their place in the file, flagged ``port_finding``
(:func:`is_port_finding`, :func:`count_port_finding`), and are not task trials: they have no
``has_print``, are never mixed into the task-trial counts, and an untrustworthy one (same causes
as above, or a print without ``chosen_tow``) is dropped with a warning like a task trial. A
port-finding trial that times out (an ``ITI`` with no print) is only recognised by the
port-finding print that follows it, so it counts as port finding when the next print is a
port-finding one, and as a dropped task trial otherwise; it has no chosen tower, so it carries
no pokes and is not warned about. A recording that holds only port finding (or pretraining) and
cannot be read as trials does not stop the session: its port-finding trials are dropped with the
reason. Trials that do not line up with the session's imported trials raise
:class:`PokeAlignmentError`, which is not a :class:`PokeParseError` because it points to a bug in
the code rather than to an odd recording.
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
from src.behavior_import.import_data import drop_non_numeric_times, parse_content_json
from src.behavior_import.problem_numbering import DEFAULT_INITIATION_TOWER

ARMS9 = ("A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3")

_POKE_IN_RE = re.compile(r"^([ABC][123])_in$")

NO_PRINT = "no summary print row"
NO_CHOSEN_TOWER = "port-finding print has no chosen tower"


class PokeParseError(ValueError):
    """The raw event log cannot be read as trials (impossible state sequence)."""


class PokeAlignmentError(ValueError):
    """The parsed trials do not line up with the session's imported trials (count or choices).

    Deliberately not a :class:`PokeParseError`: callers that skip unreadable recordings must not
    skip this.
    """


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
    ``has_print`` is False for a trial that ended without any print row and for a port-finding
    trial (neither is one of the session's trials); ``chosen`` is ``None`` when there is no
    usable print. ``port_finding`` marks a port-finding trial: its ``drop_reason`` and pokes mean
    the same, but it must never be mixed with the task trials, so a caller that wants task trials
    leaves it out first.

    Attributes:
        chosen (str | None): Chosen tower from the trial print, or ``None`` without a usable print.
        printed_long_pokes (tuple[str, ...] | None): The print's non-choice ``long_pokes`` list,
            or ``None`` if absent.
        long_pokes (tuple[str, ...]): Ports of all long pokes from initiation to the choice,
            the choice last; empty for a dropped trial.
        window_pokes (tuple[str, ...] | None): Ports of the ``_in`` events of the after-choice
            window, or ``None`` if the window never closed (or the trial was dropped).
        drop_reason (str | None): ``None`` for a trial to analyse, else why it is not.
        has_print (bool): False for a trial that ended without any print row, and for a
            port-finding trial. Default True.
        port_finding (bool): True for a port-finding trial. Default False.
    """

    chosen: str | None
    printed_long_pokes: tuple[str, ...] | None
    long_pokes: tuple[str, ...]
    window_pokes: tuple[str, ...] | None
    drop_reason: str | None = None
    has_print: bool = True
    port_finding: bool = False


class LongPokeCheck(NamedTuple):
    """Raw-event long pokes against the printed ``long_pokes`` lists.

    ``mismatches`` holds ``(trial, raw, printed)`` for each disagreeing trial (1-based within the
    compared trials); ``raw`` is the raw long pokes including the choice poke and ``printed`` the
    printed non-choice list followed by the chosen tower.

    Attributes:
        n_trials (int): Trials compared (the non-dropped ones).
        n_agree (int): Trials where raw and printed long pokes agree.
        mismatches (list[tuple[int, tuple[str, ...], tuple[str, ...] | None]]):
            ``(trial, raw, printed)`` per disagreeing trial; *printed* is ``None`` when the
            print has no list.
    """

    n_trials: int
    n_agree: int
    mismatches: list[tuple[int, tuple[str, ...], tuple[str, ...] | None]]


def _first_alias(printed: dict, names: Sequence[str]):
    """Value of the first of *names* present in the trial print *printed*, else ``None``."""
    return next((printed[n] for n in names if n in printed), None)


def _is_port_finding_print(printed, label: str) -> bool:
    """Whether a parsed print row is a port-finding trial.

    A port-finding print has ``num_t_found``, ``found_ports`` false and no ``num_t``. The importer
    discards every print with ``num_t_found``, so a print with ``num_t_found`` that breaks the
    rest of the rule would be a task trial here and a discarded one there; it raises instead.

    Raises:
        PokeParseError: If the print has ``num_t_found`` but not ``found_ports`` false, or has
            ``num_t`` as well.
    """
    if not isinstance(printed, dict) or "num_t_found" not in printed:
        return False
    if printed.get("found_ports") is not False or "num_t" in printed:
        raise PokeParseError(
            f"{label}: a print has num_t_found but is not a port-finding print "
            f"(found_ports={printed.get('found_ports')!r}, num_t present: {'num_t' in printed}); "
            "port finding needs num_t_found, found_ports false and no num_t"
        )
    return True


def is_port_finding(trial: TrialPokes) -> bool:
    """Whether *trial* is a port-finding trial rather than a task trial.

    Args:
        trial (TrialPokes): One trial as returned by :func:`parse_session`.

    Returns:
        bool: ``True`` if the trial was skipped as port finding.
    """
    return trial.port_finding


def count_port_finding(trials: Sequence[TrialPokes]) -> int:
    """Number of port-finding trials in *trials* (as from :func:`parse_session`).

    Args:
        trials (Sequence[TrialPokes]): The parsed trials of one session.

    Returns:
        int: How many of them are port-finding trials.
    """
    return sum(is_port_finding(t) for t in trials)


def _check_state_times(df: pd.DataFrame, label: str) -> None:
    """Raise if the ``state`` rows are not in time order (other rows may be, see the module doc)."""
    state_times = df.loc[df["type"] == "state", "time"].to_numpy(dtype=float)
    if np.any(np.diff(state_times) < 0):
        raise PokeParseError(f"{label}: state rows are not in increasing time order")


@dataclass
class _Trial:
    chosen: str | None
    printed: tuple[str, ...] | None
    long_pokes: tuple[str, ...]
    window: tuple[str, ...] | None = None
    drop_reason: str | None = None
    has_print: bool = True
    port_finding: bool = False

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
        self.unprinted = 0  # timed-out trials since the last print, which may be port finding
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
            self.unprinted += 1
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
            last.window = tuple(self.window)  # ends with the initiating poke
        self.window, self.awaiting_window = None, False

    def poke_in(self, port: str) -> None:
        self.last_in = port
        if self.window is not None:
            self.window.append(port)

    def trial_printed(self, printed: dict, *, port_finding: bool = False) -> None:
        """A trial's print row: close the trial it ends, task or port finding.

        Trials that timed out since the previous print were port-finding trials too when this
        print is a port-finding one; they have no chosen tower, so they carry no pokes.
        """
        if port_finding:
            for timed_out in self.trials[len(self.trials) - self.unprinted :]:
                timed_out.port_finding = True
        self.unprinted = 0
        if self.awaiting_window:  # a print with no initiation after the previous trial
            self.trials[-1].dropped("next trial printed without an initiation")
            self.window, self.awaiting_window = None, False
        chosen = _first_alias(printed, ALIASES["choice"])
        if chosen is None:
            reason = NO_CHOSEN_TOWER if port_finding else NO_PRINT
            self.trials.append(_Trial(None, None, (), None, reason, not port_finding, port_finding))
            self.reset_trial()
            return
        chosen = str(chosen)
        printed_list = _first_alias(printed, ALIASES["long_pokes"])
        trial = _Trial(
            chosen,
            None if printed_list is None else tuple(str(p) for p in printed_list),
            tuple(self.long_pokes),
            has_print=not port_finding,
            port_finding=port_finding,
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
    """The pokes of every trial printed in one raw recording, task and port finding.

    Args:
        df (pd.DataFrame): The raw four-column pyControl DataFrame (``time``, ``type``,
            ``subtype``, ``content``), in log order.
        initiation_tower (str): Port whose ``_in`` starts a trial, e.g. ``"B2"``. Default is
            the protocol default.
        label (str): Names the recording in error messages. Default ``"recording"``.

    Returns:
        list[TrialPokes]: One :class:`TrialPokes` per trial (a trial with a print row, or one
        that ended in an ITI without any), in the order of the file; port-finding trials are
        flagged ``port_finding``. A recording without bandit trials (port finding, pretraining)
        gives only its port-finding trials; if it cannot be read as trials, those are returned
        dropped, with the reason, instead of raising. Untrustworthy trials carry a
        ``drop_reason`` (see the module doc).

    Raises:
        PokeParseError: If the ``state`` rows are not in time order, a state sequence is impossible
            (``poked_in`` without ``hold_poke``, two ITIs in a row), or a trial is initiated by
            a port other than *initiation_tower*; for a recording without bandit trials only if
            a print breaks the port-finding rule.
    """
    df = drop_non_numeric_times(df, label)
    has_bandit_trials = bool(bandit_trial_info(df, label, label))
    try:
        trials = _read_trials(df, initiation_tower, label)
    except PokeParseError as error:
        if has_bandit_trials:
            raise
        return _unreadable_port_finding(df, label, error)
    return trials if has_bandit_trials else [t for t in trials if t.port_finding]


def _read_trials(df: pd.DataFrame, initiation_tower: str, label: str) -> list[TrialPokes]:
    """Run the state machine over the rows of one recording, in file order."""
    _check_state_times(df, label)
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
            if isinstance(printed, dict):
                position.trial_printed(printed, port_finding=_is_port_finding_print(printed, label))
    return [
        TrialPokes(
            t.chosen, t.printed, t.long_pokes, t.window, t.drop_reason, t.has_print, t.port_finding
        )
        for t in position.trials
    ]


def _unreadable_port_finding(
    df: pd.DataFrame, label: str, error: PokeParseError
) -> list[TrialPokes]:
    """The port-finding trials of a recording without task trials, dropped because of *error*.

    A recording that holds no task trials cannot take the session's task trials down with it, so
    its failure to parse is reported through the dropped-trial warning rather than raised. A
    print that breaks the port-finding rule still raises.
    """
    reason = f"recording could not be read ({error})"
    prints = (
        parse_content_json(content)
        for row_type, subtype, content in zip(df["type"], df["subtype"], df["content"], strict=True)
        if row_type == "variable" and subtype == "print"
    )
    return [
        TrialPokes(None, None, (), None, reason, has_print=False, port_finding=True)
        for printed in prints
        if _is_port_finding_print(printed, label)
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
        raise PokeAlignmentError(
            f"{label}: parsed {len(printed)} trials from the raw events but the session lists "
            f"{len(choices)} choices"
        )
    for k, (trial, choice) in enumerate(zip(printed, choices, strict=True), start=1):
        if trial.chosen != choice:
            raise PokeAlignmentError(
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
        session (dict): A session record with the importer's ``"df"`` (a DataFrame or a list of
            them).
        initiation_tower (str | None): Defaults to the session's ``"initiation_tower"``, or the
            protocol default ``"B2"`` when the session has none.
        label (str): Names the session in error messages. Default ``"session"``.

    Returns:
        list[TrialPokes]: The trials of all recordings, in recording order.

    Warns once with the number of dropped trials (see the module doc).

    Raises:
        PokeParseError: As :func:`parse_recording`, or if the session has no ``"df"``.
        PokeAlignmentError: If the parsed trials do not match the session's ``"choice"`` list.
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
    # A port-finding trial that timed out is expected and has no pokes to lose: not a drop.
    reasons = Counter(
        f"port finding: {t.drop_reason}" if t.port_finding else t.drop_reason
        for t in trials
        if t.drop_reason is not None and not (t.port_finding and t.drop_reason == NO_PRINT)
    )
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

    Args:
        long_pokes (Sequence[str]): Ports of a trial's long pokes in order
            (``TrialPokes.long_pokes``).
        initiation_tower (str): The port that initiates trials, e.g. ``"B2"``.

    Returns:
        list[str]: *long_pokes* from the first poke at a port other than *initiation_tower*
        onward; empty if there is none.
    """
    start = next((i for i, port in enumerate(long_pokes) if port != initiation_tower), None)
    return [] if start is None else list(long_pokes[start:])


def pokes_after_choice(window_pokes: Sequence[str] | None, chosen: str) -> list[str]:
    """Port visits to count after the choice: leading visits to the chosen tower are dropped.

    The raw ``_in`` events are first merged into visits: consecutive events at the same port,
    with no other port in between, are one visit. A mouse sitting at a port breaks and re-makes
    the beam many times (tens of ms apart), and each of those is an ``_in`` event, not a new
    poke. The mouse may be drinking at the chosen port, so visits there count only after a
    visit to some other port.

    Args:
        window_pokes (Sequence[str] | None): Ports of the after-choice window
            (``TrialPokes.window_pokes``); ``None`` if the window never closed.
        chosen (str): The trial's chosen tower.

    Returns:
        list[str]: The visits (one port per visit) from the first visit to a port other than
        *chosen* onward; empty if there is none.

    Raises:
        ValueError: If *window_pokes* is ``None`` (the window was never closed).
    """
    if window_pokes is None:
        raise ValueError("the after-choice window was never closed; it has no pokes to count")
    visits = port_visits(window_pokes)
    start = next((i for i, port in enumerate(visits) if port != chosen), None)
    return [] if start is None else visits[start:]


def port_visits(pokes: Sequence[str]) -> list[str]:
    """Merge consecutive pokes at the same port into one visit.

    Args:
        pokes (Sequence[str]): Ports of successive ``_in`` events, in order.

    Returns:
        list[str]: One port per visit: *pokes* with runs of the same port collapsed to one.
    """
    return [port for i, port in enumerate(pokes) if i == 0 or port != pokes[i - 1]]


def cross_check_long_pokes(trials: Sequence[TrialPokes]) -> LongPokeCheck:
    """Compare the raw long pokes of each trial with its printed ``long_pokes`` list.

    Agreement means the raw long pokes equal the printed non-choice long pokes followed by the
    chosen tower; trials whose print has no list disagree. Dropped trials are not compared.

    Args:
        trials (Sequence[TrialPokes]): The trials of a session, as from :func:`parse_session`.

    Returns:
        LongPokeCheck: The number of compared trials, how many agree, and the disagreeing ones.
    """
    trials = [t for t in trials if t.drop_reason is None and not t.port_finding]
    mismatches = []
    for k, trial in enumerate(trials, start=1):
        printed = (
            None if trial.printed_long_pokes is None else (*trial.printed_long_pokes, trial.chosen)
        )
        if printed != trial.long_pokes:
            mismatches.append((k, trial.long_pokes, printed))
    return LongPokeCheck(len(trials), len(trials) - len(mismatches), mismatches)
