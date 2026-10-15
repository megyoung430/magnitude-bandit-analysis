"""One problem-numbering rule, shared by ``import_data`` and ``group_sessions_by_problem``.

A *problem* is a tower layout: the set of choice towers plus the initiation tower.
Per subject, in session order:

- A session whose layout equals the previous unflagged session's layout belongs to
  the same problem.
- A different layout starts a new problem, including a return to an earlier layout
  (A, B, A -> 1, 2, 3 when B is real).
- A session whose layout differs from the session before it is a *suspected settings
  error* (problem ``None``, flag :data:`SUSPECTED_SETTINGS_ERROR`) if any of these hold:

  a. the new layout lasts only that one session before changing again;
  b. the session has no ``num_t_found`` (neither in its first ``run_start`` nor in a
     printed trial);
  c. the session does not start with ``found_ports`` false (first ``run_start``).

  A flagged session is neither a new problem nor pooled with another one, and it is
  ignored when deciding whether the next session changed problem: A, flagged B, A is
  problems 1, None, 1.
- Sessions that are not bandit problems are skipped (problem ``None``) and are ignored
  when comparing the sessions around them, so they never break pooling
  (A, skipped, A is problem 1, 1). In this order:

  1. *Pretraining* (task name, see :func:`is_pretraining`): flag :data:`PRETRAINING`,
     no warning.
  2. No choice towers and port-finding trials printed: flag :data:`PORT_FINDING`, with
     a warning.
  3. No choice towers otherwise: flag :data:`NO_CHOICE_TOWERS`, with a warning, rather
     than guessing what the session was.
- The first session with a layout is never flagged.
"""

import re
import warnings
from typing import NamedTuple

SUSPECTED_SETTINGS_ERROR = "suspected_settings_error"
PRETRAINING = "pretraining"
PORT_FINDING = "port_finding"
NO_CHOICE_TOWERS = "no_choice_towers"

# Pretraining tasks, recognised from the ``task_name`` info row (e.g.
# ``Meg\3x3_field_drifting_value_bandit``) by its last path component, case-insensitive.
# The lab's cohort-fitting notebook excludes task names containing "mag" or "pretraining";
# "mag" is matched as the whole name here so that a task such as "..._magnitude_bandit" is
# not mistaken for pretraining.
PRETRAINING_TASK_NAMES = frozenset({"mag"})
PRETRAINING_TASK_NAME_PARTS = ("pretraining",)
DEFAULT_INITIATION_TOWER = "B2"

TOWER_TOKEN_RE = re.compile(r"^\s*([A-Za-z]+)\s*(\d+)\s*$")


class SessionFacts(NamedTuple):
    """What the numbering rule needs to know about one session."""

    layout: tuple | None  # from :func:`tower_layout`; ``None`` without choice towers
    has_num_t_found: bool
    starts_found_ports_false: bool
    task_name: str | None = None
    has_port_finding_trials: bool = False  # a printed trial dict carries ``num_t_found``


class ProblemAssignment(NamedTuple):
    problem: int | None  # ``None`` for a flagged session
    flag: str | None
    reasons: tuple  # which of the checks fired, for the warning text


def normalize_tower_token(x):
    """Normalise a single tower token into a sortable, comparable tuple.

    Converts letter+digit tokens (e.g. ``"A1"``, ``" b02 "``) to
    ``(LETTER, int)`` tuples for robust comparison.  Falls back gracefully
    for raw strings, numbers, and nested tuples.

    Args:
        x: A tower token — typically a string like ``"A1"`` or ``"C3"``, a
            number, or a tuple of such values.

    Returns:
        A normalised ``(prefix, value)`` tuple:

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

    Args:
        raw_sig: A raw signature as returned by :func:`choice_towers_signature`:
            a sequence, dict, string, or ``None``.

    Returns:
        A sorted, normalised tuple suitable for equality comparison, or
        ``None`` if *raw_sig* is ``None``.
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
    """
    if not choice_towers:
        return None
    towers = tuple(sorted(choice_towers, key=str))
    return permutation_invariant_signature((towers, initiation_tower or DEFAULT_INITIATION_TOWER))


def is_pretraining(task_name):
    """True if *task_name* (the ``task_name`` info row, or ``None``) is a pretraining task."""
    if not task_name:
        return False
    name = re.split(r"[\\/]", task_name.strip())[-1].casefold()
    return name in PRETRAINING_TASK_NAMES or any(
        part in name for part in PRETRAINING_TASK_NAME_PARTS
    )


def _skip_assignment(session):
    """The assignment for a session that is not a bandit problem, or ``None`` if it is one."""
    if is_pretraining(session.task_name):
        return ProblemAssignment(None, PRETRAINING, ())
    if session.layout is not None:
        return None
    if session.has_port_finding_trials:
        return ProblemAssignment(None, PORT_FINDING, ("no choice towers; port-finding trials",))
    return ProblemAssignment(None, NO_CHOICE_TOWERS, ("no choice towers",))


def _settings_error_reasons(facts, position, layouts):
    """Why the session at *position* of *layouts* is a suspected settings error (maybe empty).

    *layouts* is ``[(index, layout)]`` for the sessions that have a layout; the session at
    *position* must differ from the one before it.
    """
    reasons = []
    following = layouts[position + 1][1] if position + 1 < len(layouts) else None
    if following is not None and following != layouts[position][1]:
        reasons.append("layout lasts one session before changing again")
    if not facts.has_num_t_found:
        reasons.append("no num_t_found")
    if not facts.starts_found_ports_false:
        reasons.append("does not start with found_ports false")
    return tuple(reasons)


def _flagged_reasons(facts, skipped):
    """``{session index: reasons}`` for sessions that changed layout and look like errors."""
    layouts = [(i, f.layout) for i, f in enumerate(facts) if i not in skipped]
    flagged = {}
    for position in range(1, len(layouts)):
        index, layout = layouts[position]
        if layout == layouts[position - 1][1]:
            continue
        reasons = _settings_error_reasons(facts[index], position, layouts)
        if reasons:
            flagged[index] = reasons
    return flagged


def assign_problems(facts):
    """Problem number and flag for one subject's sessions, given in session order.

    See the module docstring for the rule. Returns one :class:`ProblemAssignment` per
    session.
    """
    skipped = {i: a for i, f in enumerate(facts) if (a := _skip_assignment(f))}
    flagged = _flagged_reasons(facts, skipped)
    assignments = []
    problem, last_layout = 0, None
    # Each number depends on the last unflagged layout, so this is a loop.
    for index, session in enumerate(facts):
        if index in skipped:
            assignments.append(skipped[index])
        elif index in flagged:
            assignments.append(ProblemAssignment(None, SUSPECTED_SETTINGS_ERROR, flagged[index]))
        else:
            if session.layout != last_layout:
                problem, last_layout = problem + 1, session.layout
            assignments.append(ProblemAssignment(problem, None, ()))
    return assignments


def number_subject_sessions(subject, session_ids, facts):
    """:func:`assign_problems` plus one warning per flagged session, naming it."""
    assignments = assign_problems(facts)
    for session_id, assignment in zip(session_ids, assignments, strict=True):
        if assignment.flag and assignment.flag != PRETRAINING:
            warnings.warn(
                f"{subject} {session_id}: {assignment.flag.replace('_', ' ')} ("
                + "; ".join(assignment.reasons)
                + "); excluded from problem numbering.",
                stacklevel=2,
            )
    return assignments
