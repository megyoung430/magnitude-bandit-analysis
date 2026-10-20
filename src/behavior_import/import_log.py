"""A structured record of what the importer skipped or flagged, returned next to the data.

The importer warns (:class:`ImportSkipWarning`) wherever it leaves data out, and the sessions
it leaves out of the problems are not in the dict it returns. An :class:`ImportLog` is filled
as the import runs, so a table can list every skip and flag afterwards. Passing a log changes
nothing that is loaded or printed.

Skips (:class:`SkipRecord`) come at three levels:

- ``"file"``: a ``.tsv`` that never became a recording: unreadable, no ``sub-<N>_id-<mouse>``
  folder in its path, or in a session folder without a number (``ses-test``, ``ses-baseplating``).
- ``"recording"``: one recording of a session without trial information while the session's other
  recordings have trials.
- ``"session"``: every recording of the session lacks trial information, so the session is left
  out of every analysis.

Flags (:class:`SessionFlag`) are the problem-numbering outcomes of
:mod:`src.behavior_import.problem_numbering` for the sessions of the final data.

The log also carries the raw recordings of the sessions that hold only port finding
(:attr:`ImportLog.port_finding_sessions`), so the poke analysis can count their trials without
those sessions entering the trials by problem.
"""

from dataclasses import dataclass, field
from typing import NamedTuple

from src.behavior_import.warning_records import WarningRecord

FILE, RECORDING, SESSION = "file", "recording", "session"


class ImportSkipWarning(UserWarning):
    """The importer left a file, recording or session out of the data.

    Raised next to the matching :class:`SkipRecord`, so the skip reaches the console and the
    warnings log as well as ``no_trial_sessions.csv``.
    """


UNREADABLE_FILE = "unreadable_file"
NO_SUBJECT = "no_subject"
UNNUMBERED_SESSION = "unnumbered_session"
NO_BANDIT_FIELDS = "no_bandit_fields"
ONLY_PORT_FINDING = "only_port_finding"
NO_TRIAL_PRINTS = "no_trial_prints"


class SkipRecord(NamedTuple):
    """Something the importer left out.

    Attributes:
        mouse (str | None): Subject id read from the path, ``None`` if there is none.
        session (str | None): Session folder name (``"ses-<N>_date-<YYYYMMDD>"``, or e.g.
            ``"ses-test_date-20260102"``), ``None`` if the path has none.
        file (str | None): The ``.tsv`` path; ``None`` when a whole session has no file.
        level (str): ``"file"``, ``"recording"`` or ``"session"`` (see the module doc).
        kind (str): One of the module's kind constants.
        reason (str): The reason in plain words.
        task_name (str | None): The recording's ``task_name`` info row when it was read.
    """

    mouse: str | None
    session: str | None
    file: str | None
    level: str
    kind: str
    reason: str
    task_name: str | None = None


class SessionFlag(NamedTuple):
    """A session the problem numbering flagged, or whose number is provisional.

    Attributes:
        mouse (str): Subject id.
        session (str): Session key (a ``_part-NN`` key for a split session).
        files (tuple[str, ...]): The session's recordings, earliest first.
        flag (str | None): The numbering flag, ``None`` for a provisional number without a flag.
        problem (int | None): The overall problem number, ``None`` if the session is left out.
        provisional (bool): Whether a later session can still change the number.
        reason (str): What ``flag`` means for this session, in plain words; empty without a flag.
        task_name (str | None): The session's ``task_name``.
    """

    mouse: str
    session: str
    files: tuple[str, ...]
    flag: str | None
    problem: int | None
    provisional: bool
    reason: str
    task_name: str | None


@dataclass
class ImportLog:
    """Everything the importer recorded; filled in place and returned with the trials.

    Attributes:
        skips (list[SkipRecord]): Files, recordings and sessions left out, in import order.
        flags (list[SessionFlag]): The flagged or provisional sessions of the final data. Set
            once the sessions are numbered.
        warnings (list[WarningRecord]): The Python warnings raised during the import. Filled by
            the loader, so that a cached load still has them.
        port_finding_sessions (dict[str, dict[str, dict]]): ``{mouse: {session_key: record}}``
            for the sessions that hold only port finding (they are also in ``skips``). Each
            record has ``"df"`` (the raw recordings, one DataFrame or a list), ``"choice_towers"``
            (a frozenset: the ``c_tows`` of the port-finding prints) and ``"initiation_tower"``.
            They stay out of the trials by problem, which every task analysis assumes have bandit
            trials; the poke analysis counts their trials in the port-finding maps. Not part of
            the log's equality (it holds recordings). Set once the sessions are extracted.
    """

    skips: list[SkipRecord] = field(default_factory=list)
    flags: list[SessionFlag] = field(default_factory=list)
    warnings: list[WarningRecord] = field(default_factory=list)
    port_finding_sessions: dict[str, dict[str, dict]] = field(default_factory=dict, compare=False)

    def add_skip(self, skip: SkipRecord) -> None:
        """Append *skip* to :attr:`skips`."""
        self.skips.append(skip)
