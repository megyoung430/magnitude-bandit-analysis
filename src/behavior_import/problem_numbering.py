"""One problem-numbering rule, shared by ``import_data`` and ``group_sessions_by_problem``.

A *problem* is a tower layout (the set of choice towers plus the initiation tower) run under
one task type (blocked or drifting, from the task name; see :func:`classify_task_name`).
Sessions are taken per subject in :func:`src.behavior_import.session_order.session_sort_key`
order (date, then session number; the one ordering rule of the package). Sessions that are not
bandit problems are skipped first (below); the rest are cut into *runs* of consecutive sessions
with the same layout and task type, and each run is compared with the **last unflagged
layout and type**, not with the session before it. A change of task type with an unchanged
layout is a change like any other: it starts a new problem, and when it lasts one session
before the next change it is a settings error. Sessions that carry no task type
(``task_type`` is ``None``, the importer's default) are numbered by layout alone:

- Same layout and type as the last unflagged one: the run continues that problem (so
  A, flagged B, A is problems 1, None, 1, whatever the evidence of the resumed A).
- A new layout or type that lasts exactly one session before the layout changes again is a
  *suspected settings error*: problem ``None``, flag :data:`SUSPECTED_SETTINGS_ERROR`.
  It is neither a new problem nor pooled with another, and does not become the last
  unflagged layout.
- A new layout or type that lasts two or more sessions (or is the final run of the data, whose
  duration is not yet known) is a real problem, numbered from its first session, even
  after a return to an earlier layout (A, B, B, A is 1, 2, 2, 3). If its first session
  fails the evidence checks, it keeps the number but gets flag :data:`REVIEW_SETTINGS`
  and a warning for manual review. Checks, on the first ``run_start`` and printed trials:

  c. the session starts with ``found_ports`` false;
  b. port finding happened: a printed trial dict carries ``num_t_found`` (the
     ``num_t_found`` key of ``run_start`` is always present and says nothing). Not
     checked when (c) already failed.

- A first run that lasts one session before the layout changes (B, A, A) is a settings
  error like any other one-session layout, and the next layout is problem 1. A first run
  that is the only layout in the data, or the last one, is problem 1.
- The first problem of a subject is not evidence-checked: its first session is often
  mid-problem.
- The number of a final run of one session is provisional (``ProblemAssignment.provisional``):
  a later session can turn it into a settings error. ``import_data`` and
  ``group_sessions_by_problem`` write it onto the session as ``"problem_provisional"``; a
  warning names it when the session is also flagged.

Sessions that are not bandit problems get problem ``None``, are ignored when comparing the
sessions around them (A, skipped, A is 1, None, 1) and stay in the data:

1. *Pretraining* (task name, see :func:`is_pretraining`): flag :data:`PRETRAINING`, no
   warning.
2. No choice towers and port-finding trials printed: flag :data:`PORT_FINDING`, warning.
3. No choice towers otherwise: flag :data:`NO_CHOICE_TOWERS`, warning, rather than guessing.
"""

import re
import warnings
from typing import NamedTuple

from src.behavior_import.session_order import session_sort_key  # noqa: F401  (re-exported)

SUSPECTED_SETTINGS_ERROR = "suspected_settings_error"
REVIEW_SETTINGS = "review_settings"
PRETRAINING = "pretraining"
PORT_FINDING = "port_finding"
NO_CHOICE_TOWERS = "no_choice_towers"

# Pretraining tasks, recognised from the ``task_name`` info row (e.g.
# ``Meg\3x3_field_drifting_value_bandit``) by its last path component, case-insensitive.
# The lab's cohort-fitting notebook excludes task names containing "mag" or "pretraining";
# "mag" is matched as the whole name here so that a task such as "..._magnitude_bandit" is
# not mistaken for pretraining.
BLOCKED = "blocked"
DRIFTING = "drifting"
BLOCKED_NAME_PARTS = ("blocked", "structured")
DRIFTING_NAME_PARTS = ("drifting",)
PRETRAINING_TASK_NAMES = frozenset({"mag"})
PRETRAINING_TASK_NAME_PARTS = ("pretraining",)
DEFAULT_INITIATION_TOWER = "B2"


class ProblemNumberingWarning(UserWarning):
    """A session was flagged or skipped by the numbering rule."""


# Always shown, so rerunning an import in a notebook warns again.
warnings.simplefilter("always", ProblemNumberingWarning)

TOWER_TOKEN_RE = re.compile(r"^\s*([A-Za-z]+)\s*(\d+)\s*$")


class SessionFacts(NamedTuple):
    """What the numbering rule needs to know about one session.

    Attributes:
        layout (tuple | None): Canonical tower layout from :func:`tower_layout`; ``None`` when
            the session has no choice towers.
        starts_found_ports_false (bool): Whether the first ``run_start`` has ``found_ports``
            equal to false.
        has_port_finding_trials (bool): Whether a printed trial dict carries ``num_t_found``.
        task_name (str | None): The ``task_name`` info row, or ``None``. Default ``None``.
        task_type (str | None): ``"blocked"`` or ``"drifting"`` when the importer classified
            the task name, else ``None``. Default ``None``.
    """

    layout: tuple | None  # from :func:`tower_layout`; ``None`` without choice towers
    starts_found_ports_false: bool
    has_port_finding_trials: bool  # a printed trial dict carries ``num_t_found``
    task_name: str | None = None
    # "blocked" or "drifting" when the importer classified the task name, else ``None``
    task_type: str | None = None


class ProblemAssignment(NamedTuple):
    """The outcome of the numbering rule for one session.

    Attributes:
        problem (int | None): 1-based problem number, ``None`` for a skipped or
            settings-error session.
        flag (str | None): One of :data:`SUSPECTED_SETTINGS_ERROR`, :data:`REVIEW_SETTINGS`,
            :data:`PRETRAINING`, :data:`PORT_FINDING`, :data:`NO_CHOICE_TOWERS`, or ``None``.
        reasons (tuple): Which checks fired, as strings for the warning text.
        provisional (bool): ``True`` for a final run of one session, whose number a later
            session can turn into a settings error. Default ``False``.
    """

    problem: int | None  # ``None`` for a skipped or settings-error session
    flag: str | None
    reasons: tuple  # which of the checks fired, for the warning text
    # True for a final run of one session: a later session can turn it into a settings error.
    provisional: bool = False


def normalize_tower_token(x):
    """Normalise a single tower token into a sortable, comparable tuple.

    Converts letter+digit tokens (e.g. ``"A1"``, ``" b02 "``) to
    ``(LETTER, int)`` tuples for robust comparison.  Falls back gracefully
    for raw strings, numbers, and nested tuples.

    Args:
        x (str | int | float | tuple): A tower token — typically a string like ``"A1"`` or
            ``"C3"``, a number, or a tuple of such values.

    Returns:
        tuple: A normalised ``(prefix, value)`` tuple:

        - String matching ``[A-Za-z]+[0-9]+`` → ``(UPPER_LETTER, int)``.
        - Other strings → ``("RAW", stripped_string)``.
        - Integers/floats → ``("NUM", x)``.
        - Tuples → recursively normalised tuple.

    Raises:
        TypeError: If *x* is an unsupported type.
    """
    if isinstance(x, str):
        s = x.strip()
        m = TOWER_TOKEN_RE.match(s)
        if m:
            return (m.group(1).upper(), int(m.group(2)))
        return ("RAW", s)

    # If your signature can include richer token types, extend this section.
    # For now, keep numbers comparable as-is.
    if isinstance(x, (int, float)):
        return ("NUM", x)

    # If you already produce normalized tuples in choice_towers_signature,
    # keep them comparable by normalizing elements recursively.
    if isinstance(x, tuple):
        return tuple(normalize_tower_token(e) for e in x)

    # Fail fast for unsupported objects so you don't silently create bad grouping.
    raise TypeError(f"Unsupported tower token type: {type(x)!r} (value={x!r})")


def permutation_invariant_signature(raw_sig):
    """Canonicalise a tower-set signature to be permutation-invariant.

    Normalises each token with :func:`normalize_tower_token` and sorts the
    result so that two tower sets with the same members but different orderings
    produce identical signatures.

    Rule of thumb:

    - Sequence-like (list/tuple/set/frozenset) → unordered multiset,
      canonicalised by normalising each token and sorting.
    - Dict → sort normalised ``(key, value)`` pairs.
    - Single string → normalise it.
    - ``None`` → kept as ``None``.
    - A ``(towers_tuple, initiation_tower_str)`` pair, as built by
      ``choice_towers_signature`` → ``(sorted normalised towers, normalised initiation tower)``.

    Args:
        raw_sig (tuple | list | set | frozenset | dict | str | None): A raw signature, such as
            ``choice_towers_signature`` returns: a sequence, dict, string, a
            ``(towers, initiation_tower)`` pair, or ``None``.

    Returns:
        tuple | None: A sorted, normalised tuple suitable for equality comparison, or
        ``None`` if *raw_sig* is ``None``. A single string gives its normalised token; any
        other type is returned unchanged.

    Raises:
        TypeError: If a token inside *raw_sig* is of an unsupported type.
    """
    if raw_sig is None:
        return None

    if (
        isinstance(raw_sig, tuple)
        and len(raw_sig) == 2
        and isinstance(raw_sig[0], tuple)
        and isinstance(raw_sig[1], str)
    ):
        towers_norm = tuple(sorted(normalize_tower_token(x) for x in raw_sig[0]))
        init_norm = normalize_tower_token(raw_sig[1])
        return (towers_norm, init_norm)

    if isinstance(raw_sig, (list, tuple, set, frozenset)):
        normalized = [normalize_tower_token(x) for x in raw_sig]
        return tuple(sorted(normalized))

    if isinstance(raw_sig, dict):
        items = [(normalize_tower_token(k), normalize_tower_token(v)) for k, v in raw_sig.items()]
        return tuple(sorted(items))

    if isinstance(raw_sig, str):
        return normalize_tower_token(raw_sig)

    return raw_sig


def tower_layout(choice_towers, initiation_tower):
    """Canonical, hashable layout of a session, or ``None`` if it has no choice towers.

    Tower order and letter case do not matter. A missing initiation tower means the
    protocol default, ``"B2"``.

    Args:
        choice_towers (Iterable | None): The session's choice-tower identifiers (e.g. a set of
            ``"A1"``-style strings); empty or ``None`` means none were logged.
        initiation_tower (str | None): The initiation tower, or ``None`` for the default.

    Returns:
        tuple | None: ``(sorted normalised towers, normalised initiation tower)``, or ``None``
        if there are no choice towers.
    """
    if not choice_towers:
        return None
    towers = tuple(sorted(choice_towers, key=str))
    return permutation_invariant_signature((towers, initiation_tower or DEFAULT_INITIATION_TOWER))


def is_pretraining(task_name):
    """True if *task_name* (the ``task_name`` info row, or ``None``) is a pretraining task.

    Args:
        task_name (str | None): The ``task_name`` info row; its last path component is matched
            case-insensitively.

    Returns:
        bool: ``True`` for a pretraining task; ``False`` for any other name, including a
        missing one.
    """
    if not task_name:
        return False
    name = re.split(r"[\\/]", task_name.strip())[-1].casefold()
    return name in PRETRAINING_TASK_NAMES or any(
        part in name for part in PRETRAINING_TASK_NAME_PARTS
    )


def classify_task_name(task_name):
    """``"blocked"`` or ``"drifting"`` for a pyControl task name.

    The last path component is matched case-insensitively: ``blocked`` or ``structured`` is
    blocked, ``drifting`` is drifting.

    Args:
        task_name (str | None): The ``task_name`` info row.

    Returns:
        str: ``"blocked"`` or ``"drifting"``.

    Raises:
        ValueError: If the name is missing, matches neither type, or matches both.
    """
    if not task_name:
        raise ValueError(
            f"no task name ({task_name!r}), so the task type cannot be told; "
            "the run stops here until the recording has a task_name"
        )
    last = re.split(r"[\\/]", task_name.strip())[-1].casefold()
    is_blocked = any(part in last for part in BLOCKED_NAME_PARTS)
    is_drifting = any(part in last for part in DRIFTING_NAME_PARTS)
    if is_blocked and is_drifting:
        raise ValueError(f"task name {task_name!r} matches both blocked and drifting")
    if is_blocked:
        return BLOCKED
    if is_drifting:
        return DRIFTING
    raise ValueError(
        f"unrecognised task name {task_name!r}: expected a name containing "
        f"{' / '.join(BLOCKED_NAME_PARTS)} (blocked) or "
        f"{' / '.join(DRIFTING_NAME_PARTS)} (drifting)"
    )


def known_task_type(session):
    """The task type of a session dict: its ``task_type``, else the type its task name gives.

    Args:
        session (dict): A session dict with the importer's ``task_type`` (set when the import
            classified task types) and ``task_name``.

    Returns:
        str | None: ``"blocked"`` or ``"drifting"``; ``None`` when the session carries neither
        (hand-built data), so that the caller applies its default deliberately.

    Raises:
        ValueError: If there is a task name that is unrecognised or matches both types.
    """
    if session.get("task_type") is not None:
        return session["task_type"]
    name = session.get("task_name")
    return classify_task_name(name) if name else None


def numbering_signature(session):
    """What must stay equal for sessions to belong to one problem: layout and task type.

    Args:
        session (SessionFacts): One session's facts.

    Returns:
        tuple: ``(session.layout, session.task_type)``.
    """
    return (session.layout, session.task_type)


def _skip_assignment(session):
    """The assignment for a session that is not a bandit problem, or ``None`` if it is one."""
    if is_pretraining(session.task_name):
        return ProblemAssignment(None, PRETRAINING, ())
    if session.layout is not None:
        return None
    if session.has_port_finding_trials:
        return ProblemAssignment(None, PORT_FINDING, ("no choice towers; port-finding trials",))
    return ProblemAssignment(None, NO_CHOICE_TOWERS, ("no choice towers",))


def _evidence_reasons(session):
    """Why the first session of a new problem needs review (empty if it passes)."""
    if not session.starts_found_ports_false:
        return ("does not start with found_ports false",)
    if not session.has_port_finding_trials:
        return ("no port-finding trials (num_t_found) printed",)
    return ()


def _layout_runs(facts, indices):
    """Group *indices* (session positions, in order) into runs of equal layout."""
    runs = []
    for index in indices:
        if runs and numbering_signature(facts[runs[-1][0]]) == numbering_signature(facts[index]):
            runs[-1].append(index)
        else:
            runs.append([index])
    return runs


def assign_problems(facts):
    """Problem number and flag for one subject's sessions, given in session order.

    See the module docstring for the rule.

    Args:
        facts (list[SessionFacts]): One :class:`SessionFacts` per session of one subject, in
            session order.

    Returns:
        list[ProblemAssignment]: One assignment per session, in the order of *facts*.
    """
    assignments = {i: a for i, f in enumerate(facts) if (a := _skip_assignment(f))}
    runs = _layout_runs(facts, [i for i in range(len(facts)) if i not in assignments])
    changed = "layout or task type" if any(f.task_type for f in facts) else "layout"
    problem, last_signature = 0, None
    # Each run is judged against the last unflagged layout, so this is a loop.
    for position, run in enumerate(runs):
        signature = numbering_signature(facts[run[0]])
        if signature == last_signature:
            first = ProblemAssignment(problem, None, ())
        elif len(run) == 1 and position < len(runs) - 1:
            reasons = (f"{changed} lasts one session before changing again",)
            assignments[run[0]] = ProblemAssignment(None, SUSPECTED_SETTINGS_ERROR, reasons)
            continue
        else:
            problem, last_signature = problem + 1, signature
            reasons = _evidence_reasons(facts[run[0]]) if problem > 1 else ()
            first = ProblemAssignment(problem, REVIEW_SETTINGS if reasons else None, reasons)
            first = first._replace(provisional=len(run) == 1 and position == len(runs) - 1)
        assignments[run[0]] = first
        for index in run[1:]:
            assignments[index] = ProblemAssignment(problem, None, ())
    return [assignments[i] for i in range(len(facts))]


def warn_flagged(subject, session_ids, assignments, *, report=None, stacklevel=2):
    """One :class:`ProblemNumberingWarning` per flagged session (pretraining is silent).

    Args:
        subject (str): Subject id, named in the warnings.
        session_ids (list[str]): Session keys, in the same order as *assignments*.
        assignments (list[ProblemAssignment]): As returned by :func:`assign_problems`.
        report (list[bool] | None): One boolean per session; only sessions marked ``True``
            warn. Default ``None``: all sessions. Keyword-only.
        stacklevel (int): Passed to ``warnings.warn``; 2 (default) points at this function's
            caller. Keyword-only.

    Returns:
        None: Returns nothing; the warnings are the output.
    """
    report = [True] * len(assignments) if report is None else report
    for session_id, assignment, wanted in zip(session_ids, assignments, report, strict=True):
        if not (wanted and assignment.flag and assignment.flag != PRETRAINING):
            continue
        outcome = (
            "excluded from problem numbering"
            if assignment.problem is None
            else f"kept as problem {assignment.problem}"
        )
        if assignment.provisional:
            outcome += " (provisional: a later session can turn it into a settings error)"
        warnings.warn(
            f"{subject} {session_id}: {assignment.flag.replace('_', ' ')} ("
            + "; ".join(assignment.reasons)
            + f"); {outcome}.",
            ProblemNumberingWarning,
            stacklevel=stacklevel,
        )


def number_subject_sessions(subject, session_ids, facts, *, warn=True, stacklevel=2):
    """:func:`assign_problems` plus a warning (see :func:`warn_flagged`) per flagged session.

    Args:
        subject (str): Subject id, named in the warnings.
        session_ids (list[str]): Session keys, in the same order as *facts*.
        facts (list[SessionFacts]): One :class:`SessionFacts` per session.
        warn (bool): ``False`` to stay silent. Default ``True``. Keyword-only.
        stacklevel (int): Passed to ``warnings.warn``; 2 (default) points at this function's
            caller. Keyword-only.

    Returns:
        list[ProblemAssignment]: One assignment per session, in the order of *facts*.
    """
    assignments = assign_problems(facts)
    if warn:
        warn_flagged(subject, session_ids, assignments, stacklevel=stacklevel + 1)
    return assignments
