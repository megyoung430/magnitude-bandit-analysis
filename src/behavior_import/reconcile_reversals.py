"""Reconcile trial counters with run snapshots, without inventing trials.

Per-session arrays contain incoming boundary events. ``reversal_totals`` also
includes events after the last trial, which must be carried into the next
session by the merger. Unclassified state changes are retained as diagnostics.

A recording restarts the per-recording counters (``num_blocks``, ``num_good_rev``,
``num_bad_rev``), but the session totals (``tot_num_*``) keep counting. A change that the task
logs while nothing is recording (in a ``run_end`` after the last trial, or in the next
``run_start``) is therefore read from the totals: a block change in the drifting task, a good or
bad reversal in the blocked task. Files without totals fall back to the reward magnitudes of the
trials on either side of the boundary (see :mod:`src.behavior_import.run_boundaries`).

Every session is reconciled from its raw recordings first; the checks that need the next session
(:func:`_check_session_ends`) then run on those results, and :func:`_write_session` is the only
place that changes a session dict.
"""

import json
from typing import NamedTuple

from src.behavior_import.problem_numbering import BLOCKED, UnrecognisedTaskName, known_task_type
from src.behavior_import.run_boundaries import (
    describe_boundary,
    event_matches,
    events_match,
    magnitude_change,
)
from src.behavior_import.session_order import ordered_session_keys

KINDS = ("good", "bad")
BLOCK = "block"
# Trial-variable name of each counter. "block" is the only counter of a session that logs no
# good or bad reversals (the drifting task); elsewhere it is derived from the other two.
_SERIES = {"good": "good_reversals", "bad": "bad_reversals", BLOCK: "blocks"}
_ALL_KINDS = (BLOCK, *KINDS)


class _SnapshotError(ValueError):
    """A run snapshot that must be parsed could not be; the caller adds the session."""


def _state(raw):
    aliases = {
        "good": ("num_good_reversals", "num_good_rev"),
        "bad": ("num_bad_reversals", "num_bad_rev"),
        "total_good": ("tot_num_good_reversals", "tot_num_good_rev"),
        "total_bad": ("tot_num_bad_reversals", "tot_num_bad_rev"),
        "block": ("num_blocks",),
        "total_block": ("tot_num_blocks",),
        "magnitudes": ("current_reward_magnitudes", "curr_rew_mag"),
        "problem": ("num_problem",),
        "found_ports": ("found_ports",),
    }
    return {
        field: next((raw[name] for name in names if name in raw), None)
        for field, names in aliases.items()
    }


def _snapshots(df):
    start, end = {}, {}
    for row in df.loc[df["type"] == "variable"].to_dict("records"):
        if row["subtype"] not in ("run_start", "run_end"):
            continue
        content = row["content"]
        try:
            raw = content if isinstance(content, dict) else json.loads(content)
        except (ValueError, TypeError) as err:
            if row["subtype"] == "run_start":
                # Without the start snapshot the baseline would be 0 and counters that
                # persist across recording restarts would be counted again.
                raise _SnapshotError(f"run_start snapshot is not parseable: {content!r}") from err
            continue
        if row["subtype"] == "run_start":
            start = _state(raw)
        else:
            end = _state(raw)
    return start, end


def _same_problem(first, second):
    # Require positive evidence of continuity; never bridge different arm sets.
    first_mags, second_mags = first.get("magnitudes"), second.get("magnitudes")
    if not first_mags or not second_mags or set(first_mags) != set(second_mags):
        return False
    return (
        first.get("problem") is None
        or second.get("problem") is None
        or first["problem"] == second["problem"]
    )


class _Run(NamedTuple):
    """Reconciled result for one recording run of a session."""

    n_trials: int
    corrected: dict  # kind -> per-trial cumulative counts, already offset by earlier runs
    added: dict  # kind -> events to add to the session total: between runs plus inside this run
    observed: dict  # kind -> whether any counter was recorded for this kind
    events: list
    diagnostics: list
    end_state: dict  # snapshot handed to the next run as ``previous``
    first_side: dict | None = None  # the run's first trial, as a side of a boundary
    last_side: dict | None = None  # the run's last trial, as a side of a boundary
    boundary: dict | None = None  # description of the gap before this run, if there is one


def _session_frames(sess):
    # A session without ``data`` has zero runs, so it contributes no trials.
    frames = sess.get("data", [])
    return frames if isinstance(frames, list) else [frames]


def _run_variables(sess, run_index):
    # ``extract_trials`` writes ``trial_variables = {}`` for sessions with no bandit
    # trials, so the empty fallback occurs on real data: that run has zero trials.
    # If the frames and trial variables differ in length, the surplus runs are
    # treated the same way and their trials are dropped.
    variables = sess.get("trial_variables", [])
    if isinstance(variables, list) and run_index < len(variables):
        return variables[run_index]
    return {}


def _totals_logged(previous, start, kinds):
    """True if the session totals of some counter in *kinds* are logged on both sides."""
    return any(
        previous.get("total_" + kind) is not None and start.get("total_" + kind) is not None
        for kind in kinds
    )


def _unexplained_change(before, after, change):
    """True if the magnitudes moved in a way no logged counter explains and the task allows.

    *change* is :func:`magnitude_change` of the two sides: ``False`` (nothing a block change
    would move, such as the drift of the other arms) is never flagged, and when the task type
    is unknown (``None``) any difference is.
    """
    if change is not None:
        return change
    return before is not None and after is not None and before != after


def _between_runs(
    previous, start, trial_offset, run_index, kinds=KINDS, mode=None, first_side=None
):
    """Events and diagnostics for counter changes between the previous run and this one.

    Returns ``(incoming, events, diagnostics)`` where ``incoming`` is the number of
    events per kind in *kinds* that happened while nothing was recording.

    Without any logged total, a change of the magnitudes that *mode* counts as a block change is
    one event (``"block"``, source ``"magnitudes"``). The magnitudes are those of the last trial
    before the boundary (``previous["last_side"]``) and the first trial after it (*first_side*):
    the ``run_start`` snapshot of a recording that is still finding the ports holds placeholder
    magnitudes. No such event is added when the previous run's own end-of-run events
    (``previous["end_events"]``) already account for the boundary.
    """
    incoming = {k: 0 for k in kinds}
    events, diagnostics = [], []
    if not (previous and _same_problem(previous, start)):
        return incoming, events, diagnostics
    for kind in kinds:
        before, after = previous.get("total_" + kind), start.get("total_" + kind)
        if before is None or after is None:
            continue
        delta = int(after - before)
        if delta > 0:
            incoming[kind] = delta
            events.append(dict(kind=kind, index=trial_offset, count=delta, source="between_runs"))
        elif delta < 0:
            diagnostics.append(
                dict(kind=kind, source="counter_rollback", delta=delta, run=run_index)
            )
    if previous.get("end_events"):
        return incoming, events, diagnostics
    before = (previous.get("last_side") or {}).get("magnitudes")
    after = (first_side or {}).get("magnitudes")
    change = magnitude_change(before, after, mode)
    if change and not _totals_logged(previous, start, kinds):
        events.append(dict(kind=BLOCK, index=trial_offset, count=1, source="magnitudes"))
        if BLOCK in incoming:
            incoming[BLOCK] = 1
    elif (
        _unexplained_change(before, after, change)
        and not any(incoming.values())
        and not previous.get("end_flagged")
    ):
        diagnostics.append(
            dict(source="unclassified_magnitude_change", run=run_index, before=before, after=after)
        )
    return incoming, events, diagnostics


def _counter_observed(values, start, kind):
    """True if the run recorded this counter in its trials or its start snapshot."""
    return (
        any(v is not None for v in values)
        or start.get(kind) is not None
        or start.get("total_" + kind) is not None
    )


def _running_counts(values, n_trials, baseline):
    """Per-trial event count within a run, relative to the run-start ``baseline``.

    Missing samples (``None`` or past the end of ``values``) repeat the previous
    count, and the count never decreases.
    """
    counts, count = [], 0
    for i in range(n_trials):
        value = values[i] if i < len(values) else None
        if value is not None:
            count = max(count, int(value) - baseline)
        counts.append(count)
    return counts


def _run_total(last_count, start, end, kind, baseline):
    """Events in a run: the larger of what trials, end snapshot and total counters show."""
    final = end.get(kind)
    total = max(last_count, int(final) - baseline) if final is not None else last_count
    # Total counters are independent evidence; take whichever source shows more events.
    start_total, end_total = start.get("total_" + kind), end.get("total_" + kind)
    if start_total is not None and end_total is not None and end_total >= start_total:
        total = max(total, int(end_total - start_total))
    return total


def _last_magnitudes(variables, n_trials):
    return variables.get("reward_magnitudes", [None] * n_trials)[-1]


def _terminal_magnitude_diagnostics(
    last_magnitudes, end, n_trials, trial_offset, events, run_index, mode=None
):
    """Flag a magnitude change after the last trial that no terminal reversal explains.

    With a known *mode*, drift of the magnitudes that is not a block change (the drifting task's
    other arms move on every trial) is not flagged.
    """
    explained = any(e["index"] == trial_offset + n_trials for e in events)
    if end.get("magnitudes") is None or last_magnitudes == end["magnitudes"] or explained:
        return []
    if magnitude_change(last_magnitudes, end["magnitudes"], mode) is False:
        return []
    return [
        dict(
            source="unclassified_terminal_magnitude_change",
            run=run_index,
            before=last_magnitudes,
            after=end["magnitudes"],
        )
    ]


def _terminal_from_magnitudes(last_magnitudes, start, end, kinds, mode, events, index):
    """A block change after the last trial, from the magnitudes of a ``run_end`` without counters.

    Only when the snapshot logs no counter of *kinds* and nothing else explains the change.
    """
    logged = any(end.get(k) is not None or end.get("total_" + k) is not None for k in kinds)
    explained = any(e["index"] == index for e in events)
    if logged or explained:
        return []
    if magnitude_change(last_magnitudes, end.get("magnitudes"), mode) is not True:
        return []
    return [dict(kind=BLOCK, index=index, count=1, source="magnitudes")]


def _end_state(start, end, last_magnitudes, run_totals):
    """The run's end snapshot with missing totals, magnitudes and problem filled in.

    ``last_magnitudes`` is ``None`` for a run without trials, which then shows the magnitudes
    it started from, so that a recording that crashed before its first trial still links to the
    next one.
    """
    state = dict(end)
    for kind in run_totals:
        start_total = start.get("total_" + kind)
        if state.get("total_" + kind) is None and start_total is not None:
            state["total_" + kind] = start_total + run_totals[kind]
    fallback = last_magnitudes if last_magnitudes is not None else start.get("magnitudes")
    if state.get("magnitudes") is None and fallback is not None:
        state["magnitudes"] = fallback
    if state.get("problem") is None:
        state["problem"] = start.get("problem")
    return state


def _reconcile_counter(values, start, end, kind, n_trials, offset):
    """Reconcile one counter kind within one run.

    ``offset`` is the number of events of this kind before the run's first trial.
    Returns ``(corrected, total, terminal)``: the per-trial cumulative counts, the
    events inside the run, and how many of those happen after the last trial.
    """
    # Counters can persist across recording restarts: subtract the run-start
    # baseline rather than counting historical events again.
    baseline = start.get(kind) or 0
    counts = _running_counts(values, n_trials, baseline)
    last_count = counts[-1] if counts else 0
    total = _run_total(last_count, start, end, kind, baseline)
    return [offset + c for c in counts], total, total - last_count


def _first_value(values):
    return next((v for v in values if v is not None), None)


def _edge_counters(kind, start, variables, n_trials):
    """``(first counter, last counter, first total, last total)`` of one kind in a run.

    The totals are the ``run_start`` total plus the counter's progress up to that trial, so
    they are the totals as the trial prints show them; ``None`` where the run logs no total.
    """
    values = variables.get(_SERIES[kind], [])[:n_trials]
    first, last = _first_value(values), _first_value(reversed(values))
    baseline, start_total = start.get(kind), start.get("total_" + kind)

    def total_at(value):
        if start_total is None:
            return None
        if value is None:
            return start_total
        return start_total + max(0, value - (baseline or 0))

    # A run without a printed value shows the counter it started from.
    shown_first = baseline if first is None else first
    shown_last = baseline if last is None else last
    return shown_first, shown_last, total_at(first), total_at(last)


def _trial_free_magnitudes(start):
    """The magnitudes a run without trials shows, or ``None`` if its snapshot only has placeholders.

    A recording that is still finding the ports (``found_ports`` false) logs placeholder
    magnitudes in ``run_start``.
    """
    return None if start.get("found_ports") is False else start.get("magnitudes")


def _edge_sides(start, end, variables, n_trials, inherited=None):
    """``(first_side, last_side)`` of a run in the layout of :func:`describe_boundary`.

    The magnitudes are those of the run's first and last trial. A run without trials shows the
    state it started from, before anything logged in ``run_end``; when that is only placeholders
    it shows no magnitudes as its first side and the *inherited* ones (the last trial before the
    run) as its last side, so that the next run is compared with a real trial.
    """
    magnitudes = variables.get("reward_magnitudes") or []
    if n_trials:
        first_mags = magnitudes[0] if magnitudes else None
        last_mags = magnitudes[n_trials - 1] if len(magnitudes) >= n_trials else None
    else:
        first_mags = _trial_free_magnitudes(start)
        last_mags = first_mags if first_mags is not None else inherited
    first = {"magnitudes": first_mags, "totals": {}, "counters": {}}
    last = {"magnitudes": last_mags, "totals": {}, "counters": {}}
    for kind in _ALL_KINDS:
        first_c, last_c, first_t, last_t = _edge_counters(kind, start, variables, n_trials)
        first["counters"][kind], last["counters"][kind] = first_c, last_c
        first["totals"][kind], last["totals"][kind] = first_t, last_t
    return first, last


def _reconcile_run(
    df,
    variables,
    previous,
    carried,
    trial_offset,
    run_index,
    *,
    snapshots=None,
    kinds=KINDS,
    mode=None,
):
    """Reconcile one run given the previous run's end state and totals so far.

    Args:
        df: Event frame of this run.
        variables: Trial variables of this run (may be empty).
        previous: End state of the previous run, or ``None`` for the first, with its
            ``"last_side"`` (the last trial before this run) and ``"end_events"`` (the events at
            the end of the previous run, which already account for the boundary) and
            ``"end_flagged"``.
        carried: ``{kind: events counted in earlier runs of this session}``.
        trial_offset: Number of trials in earlier runs of this session.
        run_index: Position of this run in the session (for diagnostics only).
        snapshots: ``(start, end)`` already parsed from *df*; parsed here when ``None``.
        kinds: The counters that count in this session: ``("good", "bad")`` for a session that
            logs reversal counters, ``("block",)`` for one that logs only blocks.
        mode: ``"drifting"``, ``"blocked"`` or ``None`` (task unknown), to read the magnitudes.
    """
    try:
        start, end = snapshots if snapshots is not None else _snapshots(df)
    except _SnapshotError as err:
        raise _SnapshotError(f"run {run_index}: {err}") from err
    n_trials = len(variables.get("trial", []))
    linked = bool(previous) and _same_problem(previous, start)
    inherited = ((previous.get("last_side") or {}).get("magnitudes")) if linked else None
    first_side, last_side = _edge_sides(start, end, variables, n_trials, inherited)
    incoming, events, diagnostics = _between_runs(
        previous, start, trial_offset, run_index, kinds=kinds, mode=mode, first_side=first_side
    )
    corrected, run_totals, observed = {}, {}, {}
    for kind in kinds:
        values = variables.get(_SERIES[kind], [])
        observed[kind] = _counter_observed(values, start, kind)
        corrected[kind], run_totals[kind], terminal = _reconcile_counter(
            values, start, end, kind, n_trials, carried[kind] + incoming[kind]
        )
        if terminal:
            events.append(
                dict(kind=kind, index=trial_offset + n_trials, count=terminal, source="run_end")
            )
    last_magnitudes = _last_magnitudes(variables, n_trials) if n_trials else None
    added = {kind: incoming[kind] + run_totals[kind] for kind in kinds}
    if n_trials:
        found = _terminal_from_magnitudes(
            last_magnitudes, start, end, kinds, mode, events, trial_offset + n_trials
        )
        events += found
        if BLOCK in added:
            added[BLOCK] += sum(e["count"] for e in found)
        diagnostics += _terminal_magnitude_diagnostics(
            last_magnitudes, end, n_trials, trial_offset, events, run_index, mode
        )
    boundary = None
    if linked and previous.get("last_side"):
        boundary = describe_boundary(previous["last_side"], first_side, mode)
    return _Run(
        n_trials=n_trials,
        corrected=corrected,
        added=added,
        observed=observed,
        events=events,
        diagnostics=diagnostics,
        end_state=_end_state(start, end, last_magnitudes, run_totals),
        first_side=first_side,
        last_side=last_side,
        boundary=boundary,
    )


def _unmatched_reason(kind, mode):
    if kind == "good":
        return "the best tower is the same at the start of the next session"
    if kind == BLOCK and mode == "drifting":
        return "the zero arm has not moved at the start of the next session"
    return "the magnitudes are the same at the start of the next session"


def _end_events(run, previous, n_trials_before):
    """The events at the end of *run*: they already account for the boundary that follows.

    A run without trials adds to the end events of the run before it.
    """
    end = n_trials_before + run.n_trials
    events = [e for e in run.events if e["index"] == end]
    return events if run.n_trials else [*(previous or {}).get("end_events", []), *events]


def _end_flagged(run, previous):
    """Whether *run* flagged a magnitude change after its last trial (see :func:`_end_events`)."""
    flagged = any(d["source"] == "unclassified_terminal_magnitude_change" for d in run.diagnostics)
    return flagged or (not run.n_trials and bool((previous or {}).get("end_flagged")))


def _reconcile_runs(frames, snapshots, variables, previous, kinds, mode):
    """Reconcile every run of a session in order; return ``(runs, last end state)``.

    The state handed from run to run (and on to the next session) is the run's end snapshot plus
    ``"last_side"`` (its last trial, or the last trial before it for a run without trials),
    ``"end_events"`` and ``"end_flagged"``.
    """
    runs = []
    carried = {kind: 0 for kind in kinds}
    if BLOCK in carried and snapshots:
        # The session's first block number is the one the first recording starts from.
        carried[BLOCK] = snapshots[0][0].get(BLOCK) or 0
    trial_offset = 0
    for run_index, df in enumerate(frames):
        run = _reconcile_run(
            df,
            variables[run_index],
            previous,
            carried,
            trial_offset,
            run_index,
            snapshots=snapshots[run_index],
            kinds=kinds,
            mode=mode,
        )
        for kind in kinds:
            carried[kind] += run.added[kind]
        previous = {
            **run.end_state,
            "last_side": run.last_side,
            "end_events": _end_events(run, previous, trial_offset),
            "end_flagged": _end_flagged(run, previous),
        }
        trial_offset += run.n_trials
        runs.append(run)
    return runs, previous


class _Session(NamedTuple):
    """Reconciled result for one session, before the next session has checked its end."""

    arrays: dict
    totals: dict
    observed: dict
    events: list
    diagnostics: list
    run_boundaries: list
    start: dict | None  # the first run_start; None for a session with no recording
    end_state: dict | None  # state after the last recording, as :func:`_reconcile_runs` hands on
    mode: str | None
    n_trials: int
    first_side: dict | None  # the session's first trial, as a side of a boundary
    session_edges: dict | None  # sides of the first and last trial (see _session_edges)


def _session_changes_check(refs, events, diagnostics, where, before, after, mode):
    """Mark the events in *refs* with whether the trials *before* and *after* show them.

    *refs* are ``(session, event)`` positions into *events*; *where* is the position of the
    session whose first trial is *after*. The events of one kind at the boundary are judged
    together (:func:`~src.behavior_import.run_boundaries.events_match`). Events of an earlier
    session that are not shown are also recorded in that session's *diagnostics*; the
    lists are the caller's own copies.
    """
    by_kind = {}
    for ref in refs:
        by_kind.setdefault(events[ref[0]][ref[1]]["kind"], []).append(ref)
    for kind, group in by_kind.items():
        count = sum(events[s][j]["count"] for s, j in group)
        matches = events_match(kind, count, before, after, mode)
        if matches is None:
            continue
        if count >= 2 and event_matches(kind, before, after, mode) is False:
            diagnostics[where].append(
                dict(source="stacked_changes_net_unchanged", kind=kind, count=count)
            )
        for s, j in group:
            event = events[s][j]
            events[s][j] = {**event, "session_check": "consistent" if matches else "inconsistent"}
            if not matches and s != where:
                diagnostics[s].append(
                    dict(
                        source="session_end_change_unmatched",
                        kind=event["kind"],
                        count=event["count"],
                        reason=_unmatched_reason(event["kind"], mode),
                    )
                )


def _check_session_ends(sessions):
    """Events and diagnostics of every session, with the ``session_check`` of incoming changes.

    Two kinds of change enter a session: the events logged after the previous session's last
    trial (``index`` equal to its trial count) and the ones in this session's first ``run_start``
    (``between_runs`` at trial 0). The first trial of the session that follows with trials, which
    may be several sessions later (a session with no trial has none to check against), and the
    last trial before the change show whether the magnitudes agree: ``session_check`` is
    ``"consistent"`` or ``"inconsistent"``. Nothing is marked when the sessions are not the same
    problem or the magnitudes cannot tell. An inconsistent change after a session's last trial is
    also recorded in that session's diagnostics.

    Args:
        sessions (list[_Session]): The subject's sessions in session order.

    Returns:
        list[tuple[list, list]]: ``(events, diagnostics)`` per session. These are new lists, and the
        marked events are new dicts; *sessions* is not changed.
    """
    events = [list(s.events) for s in sessions]
    diagnostics = [list(s.diagnostics) for s in sessions]
    # ``pending`` holds the (session, event) positions of changes awaiting a session with trials.
    pending, previous = [], None
    for pos, sess in enumerate(sessions):
        if sess.start is None:
            continue
        linked = previous is not None and _same_problem(previous.end_state, sess.start)
        if not linked:
            pending = []
        if sess.n_trials:
            incoming = [
                (pos, j)
                for j, e in enumerate(sess.events)
                if e["source"] == "between_runs" and e["index"] == 0
            ]
            if linked and (pending or incoming):
                before = (previous.end_state.get("last_side") or {}).get("magnitudes")
                after = sess.first_side["magnitudes"]
                _session_changes_check(
                    [*pending, *incoming], events, diagnostics, pos, before, after, sess.mode
                )
            pending = [(pos, j) for j, e in enumerate(sess.events) if e["index"] == sess.n_trials]
        else:
            pending = [*pending, *((pos, j) for j in range(len(sess.events)))]
        previous = sess
    return list(zip(events, diagnostics, strict=True))


def _combine_runs(runs, kinds=KINDS):
    """Merge per-run results into session-level values.

    Returns ``(arrays, totals, observed, events, diagnostics)``, each dict keyed by
    kind (of *kinds*) except the two lists, which keep run order.
    """
    arrays = {kind: [c for run in runs for c in run.corrected[kind]] for kind in kinds}
    totals = {kind: sum(run.added[kind] for run in runs) for kind in kinds}
    observed = {kind: any(run.observed[kind] for run in runs) for kind in kinds}
    events = [e for run in runs for e in run.events]
    diagnostics = [d for run in runs for d in run.diagnostics]
    return arrays, totals, observed, events, diagnostics


def _correct_blocks(original_blocks, events):
    """Add the boundary events at or before each trial to the block counter."""
    # Any None block raises TypeError (None + int); the extractor is expected to
    # supply integer blocks, so a missing one is not handled here.
    return [
        block + sum(e["count"] for e in events if e["index"] <= i)
        for i, block in enumerate(original_blocks)
    ]


def _write_session(sess, result, events, diagnostics):
    """Store a session's reconciled values on its dict, with the checked *events* and *diagnostics*.

    This is the only place that changes a session dict.
    """
    arrays, totals, observed = result.arrays, result.totals, result.observed
    # Rebuild block corrections from the original extraction on repeat calls.
    original_blocks = sess.setdefault(
        "blocks_before_boundary_reconciliation", list(sess.get("blocks", []))
    )
    if observed.get(BLOCK):
        # Block numbers from the counters and totals, not from the per-recording values, which
        # restart at 1 whether or not a block change happened in between.
        sess["blocks"] = arrays[BLOCK]
    else:
        sess["blocks"] = _correct_blocks(original_blocks, events)
    sess["reversal_totals"] = {k: totals[k] for k in KINDS if observed.get(k)}
    sess["reversal_boundary_events"] = events
    sess["reversal_diagnostics"] = diagnostics
    sess["run_boundaries"] = result.run_boundaries
    sess["session_edges"] = result.session_edges
    for kind in KINDS:
        if observed.get(kind):
            sess[kind + "_reversals"] = arrays[kind]
            sess["has_" + kind] = True


def _session_snapshots(frames):
    """``[(start, end)]`` of each recording; a bad ``run_start`` names its run."""
    parsed = []
    for run_index, df in enumerate(frames):
        try:
            parsed.append(_snapshots(df))
        except _SnapshotError as err:
            raise _SnapshotError(f"run {run_index}: {err}") from err
    return parsed


def _session_kinds(snapshots, variables):
    """``("good", "bad")`` if any recording logs a reversal counter, else ``("block",)``."""
    logs_reversals = any(
        _counter_observed(run_variables.get(_SERIES[kind], []), start, kind)
        for (start, _), run_variables in zip(snapshots, variables, strict=True)
        for kind in KINDS
    )
    return KINDS if logs_reversals else (BLOCK,)


def _session_mode(sess, kinds):
    """``"blocked"`` for a session with reversal counters, else the task type of the session.

    The type is that of :func:`~src.behavior_import.problem_numbering.known_task_type`: the
    session's ``task_type``, else the one its task name gives. ``None`` when the session has
    neither or the name is unrecognised (a pretraining task): the magnitudes are then not read as
    block changes, so only the counters count.

    Raises:
        ValueError: If the task name matches both types.
    """
    if kinds == KINDS:
        return BLOCKED
    try:
        return known_task_type(sess)
    except UnrecognisedTaskName:
        return None


def _session_edges(runs):
    """The sides of the session's first and last trial, ``None`` if it has no trial.

    A recording without trials is skipped: it has no trial to read the magnitudes or counters of.
    """
    with_trials = [run for run in runs if run.n_trials]
    if not with_trials:
        return None
    return {"first": with_trials[0].first_side, "last": with_trials[-1].last_side}


def _first_trial_side(runs):
    """The first trial of a session as a side of a boundary; ``None`` if it has no trial."""
    return next((run.first_side for run in runs if run.n_trials), None)


def _reconcile_session(sess, previous):
    """Reconcile one session; return ``(result, state after its last recording)``.

    Reads *sess* and changes nothing; :func:`_write_session` stores the result.
    """
    frames = _session_frames(sess)
    snapshots = _session_snapshots(frames)
    variables = [_run_variables(sess, i) for i in range(len(frames))]
    kinds = _session_kinds(snapshots, variables)
    mode = _session_mode(sess, kinds)
    runs, last_end_state = _reconcile_runs(frames, snapshots, variables, previous, kinds, mode)
    arrays, totals, observed, events, diagnostics = _combine_runs(runs, kinds)
    result = _Session(
        arrays=arrays,
        totals=totals,
        observed=observed,
        events=events,
        diagnostics=diagnostics,
        run_boundaries=[run.boundary for run in runs],
        start=snapshots[0][0] if runs else None,
        end_state=last_end_state if runs else None,
        mode=mode,
        n_trials=sum(run.n_trials for run in runs),
        first_side=_first_trial_side(runs),
        session_edges=_session_edges(runs),
    )
    return result, last_end_state


def uncarried_counts(sess):
    """``{kind: events}`` logged after the session's last trial that the next session's start
    does not show.

    The logged totals keep these events, but the next session does not start a new block (or
    reversal) because of them, so a merger must not carry them into its first trial.

    Args:
        sess (dict): A session dict, optionally with ``reversal_boundary_events``.

    Returns:
        dict[str, int]: Events per kind whose ``session_check`` is ``"inconsistent"`` and that sit
        after the last trial; empty if there are none.
    """
    counts = {}
    n_trials = len(sess.get("trial", []))
    for event in sess.get("reversal_boundary_events", []):
        if event.get("session_check") == "inconsistent" and event["index"] == n_trials:
            counts[event["kind"]] = counts.get(event["kind"], 0) + event["count"]
    return counts


def reconcile_reversals(data):
    """Attach corrected counters, totals, boundary events and diagnostics in place.

    Call after extracting *all* sessions and before selecting problems or early/
    late subsets. Re-running is safe: raw per-run trial variables are the source.
    Positive cumulative-total changes between runs identify additional events;
    negative changes and magnitude-only changes are diagnostics, not reversals.

    Sessions run in date-then-number order (:mod:`src.behavior_import.session_order`), so
    each session's counters start from the right previous session.

    A session that logs good/bad counters counts good and bad events. A session that logs only
    blocks (the drifting task) counts ``"block"`` events from ``tot_num_blocks`` and
    ``num_blocks``, and its ``blocks`` come from them. A change logged while nothing was
    recording is an event at the first trial of the next recording, whether it is in the
    ``run_end`` after the last trial or in the next ``run_start``. Where the session totals are
    absent, a change of the magnitudes between the last trial before the boundary and the first
    trial after it is an event of kind ``"block"`` and source ``"magnitudes"``: the zero arm
    moved (drifting), or the magnitudes permuted (blocked). The ``run_start`` snapshot is not
    used, because a recording that is still finding the ports logs placeholder magnitudes in it,
    and no such event is added where the previous recording's own end-of-recording events already
    account for the boundary.

    Each session dict gains ``"reversal_totals"`` (``{"good": int, "bad": int}`` for the kinds
    that were recorded, including events after the last trial),
    ``"reversal_boundary_events"`` (list of ``{"kind", "index", "count", "source"}`` dicts, with
    ``kind`` ``"good"``, ``"bad"`` or ``"block"`` and ``source`` ``"between_runs"``,
    ``"run_end"`` or ``"magnitudes"``), ``"reversal_diagnostics"`` (list of dicts,
    each with a ``"source"`` such as ``"counter_rollback"``), ``"run_boundaries"`` (one
    :func:`~src.behavior_import.run_boundaries.describe_boundary` description per recording: the
    gap between the previous recording's last trial and this one's first, ``None`` for the very
    first recording or when the previous recording is another problem),
    ``"session_edges"`` (``{"first", "last"}``: the first trial of the session's first recording
    with trials and the last trial of its last one, each as a side of a boundary in the layout
    of :func:`~src.behavior_import.run_boundaries.describe_boundary`; ``None`` without trials,
    so that two sessions of a problem can be compared whatever lies between them in time) and
    ``"blocks_before_boundary_reconciliation"``. It overwrites ``"blocks"``, and
    ``"good_reversals"`` / ``"bad_reversals"`` (with ``"has_good"`` / ``"has_bad"`` set) for
    recorded kinds, with the corrected per-trial cumulative counts.

    An event after a session's last trial, and an event in the next session's first
    ``run_start``, also get ``"session_check"``: ``"consistent"`` if the last trial before the
    event and the first trial after it show it, ``"inconsistent"`` if they do not (the best tower
    for a good reversal, a permutation for a bad one, the zero arm for a drifting block change).
    The first trial after it is the first trial of the next session that has trials. The events
    of one kind at one boundary are judged together: when two or more are logged and the net
    result shows no change (the zero arm or the best tower went away and came back), they count
    as shown and a ``"stacked_changes_net_unchanged"`` diagnostic is added to the session that
    has the first trial. An inconsistent event after the last trial is also recorded in
    ``"reversal_diagnostics"`` as ``"session_end_change_unmatched"``; see
    :func:`uncarried_counts`. Every session is reconciled first and the checks are made on the
    results, so no session is changed while another is read; each session dict is written once.

    Args:
        data (dict): ``{subject: {session_key: session_dict}}`` after trial extraction, so each
            session carries ``"data"`` and ``"trial_variables"``. Mutated in place.

    Returns:
        dict: The same *data* object.

    Raises:
        ValueError: If a run's ``run_start`` snapshot cannot be parsed or a session's task name
            matches both task types (the message names the subject and session), or if a
            subject's session numbers and dates disagree.
    """
    for subject, sessions in data.items():
        keys = ordered_session_keys(subject, sessions)
        results, previous = [], None
        for key in keys:
            try:
                result, previous = _reconcile_session(sessions[key], previous)
            except ValueError as err:
                raise ValueError(f"{subject} {key}: {err}") from err
            results.append(result)
        checked = _check_session_ends(results)
        for key, result, (events, diagnostics) in zip(keys, results, checked, strict=True):
            _write_session(sessions[key], result, events, diagnostics)
    return data
