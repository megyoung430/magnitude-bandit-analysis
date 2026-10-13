"""Reconcile trial counters with run snapshots, without inventing trials.

Per-session arrays contain incoming boundary events. ``reversal_totals`` also
includes events after the last trial, which must be carried into the next
session by the merger. Unclassified state changes are retained as diagnostics.
"""

import json
import re
from typing import NamedTuple

KINDS = ("good", "bad")


def _state(d):
    aliases = {
        "good": ("num_good_reversals", "num_good_rev"),
        "bad": ("num_bad_reversals", "num_bad_rev"),
        "total_good": ("tot_num_good_reversals", "tot_num_good_rev"),
        "total_bad": ("tot_num_bad_reversals", "tot_num_bad_rev"),
        "magnitudes": ("current_reward_magnitudes", "curr_rew_mag"),
        "problem": ("num_problem",),
    }
    return {k: next((d[a] for a in names if a in d), None) for k, names in aliases.items()}


def _snapshots(df):
    start, end = {}, {}
    for row in df.loc[df["type"] == "variable"].to_dict("records"):
        if row["subtype"] not in ("run_start", "run_end"):
            continue
        content = row["content"]
        try:
            d = content if isinstance(content, dict) else json.loads(content)
        except (ValueError, TypeError):
            continue
        if row["subtype"] == "run_start":
            start = _state(d)
        else:
            end = _state(d)
    return start, end


def _same_problem(a, b):
    # Require positive evidence of continuity; never bridge different arm sets.
    am, bm = a.get("magnitudes"), b.get("magnitudes")
    if not am or not bm or set(am) != set(bm):
        return False
    return a.get("problem") is None or b.get("problem") is None or a["problem"] == b["problem"]


class _Run(NamedTuple):
    """Reconciled result for one recording run of a session."""

    n_trials: int
    corrected: dict  # kind -> per-trial cumulative counts, already offset by earlier runs
    added: dict  # kind -> events to add to the session total: between runs plus inside this run
    observed: dict  # kind -> whether any counter was recorded for this kind
    events: list
    diagnostics: list
    end_state: dict  # snapshot handed to the next run as ``previous``


def _session_order(key):
    """Sort by session number, then name; keys without ``ses-<n>`` go last."""
    match = re.search(r"ses-(\d+)", key)
    return (int(match[1]) if match else float("inf"), key)


def _session_frames(sess):
    frames = sess.get("data", [])
    return frames if isinstance(frames, list) else [frames]


def _run_variables(sess, run_index):
    variables = sess.get("trial_variables", [])
    if isinstance(variables, list) and run_index < len(variables):
        return variables[run_index]
    return {}


def _between_runs(previous, start, trial_offset, run_index):
    """Events and diagnostics for counter changes between the previous run and this one.

    Returns ``(incoming, events, diagnostics)`` where ``incoming`` is the number of
    events per kind that happened while nothing was recording.
    """
    incoming = {k: 0 for k in KINDS}
    events, diagnostics = [], []
    if not (previous and _same_problem(previous, start)):
        return incoming, events, diagnostics
    for kind in KINDS:
        a, b = previous.get("total_" + kind), start.get("total_" + kind)
        if a is None or b is None:
            continue
        delta = int(b - a)
        if delta > 0:
            incoming[kind] = delta
            events.append(dict(kind=kind, index=trial_offset, count=delta, source="between_runs"))
        elif delta < 0:
            diagnostics.append(
                dict(kind=kind, source="counter_rollback", delta=delta, run=run_index)
            )
    if previous.get("magnitudes") != start.get("magnitudes") and not any(incoming.values()):
        diagnostics.append(
            dict(
                source="unclassified_magnitude_change",
                run=run_index,
                before=previous.get("magnitudes"),
                after=start.get("magnitudes"),
            )
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
    # Total counters are an independent fallback for end snapshots.
    a, b = start.get("total_" + kind), end.get("total_" + kind)
    if a is not None and b is not None and b >= a:
        total = max(total, int(b - a))
    return total


def _last_magnitudes(tv, n_trials):
    return tv.get("reward_magnitudes", [None] * n_trials)[-1]


def _terminal_magnitude_diagnostics(tv, end, n_trials, trial_offset, events, run_index):
    """Flag a magnitude change after the last trial that no terminal reversal explains."""
    last_mags = _last_magnitudes(tv, n_trials)
    explained = any(
        e["source"] == "run_end" and e["index"] == trial_offset + n_trials for e in events
    )
    if end.get("magnitudes") is None or last_mags == end["magnitudes"] or explained:
        return []
    return [
        dict(
            source="unclassified_terminal_magnitude_change",
            run=run_index,
            before=last_mags,
            after=end["magnitudes"],
        )
    ]


def _end_state(start, end, tv, n_trials, run_totals):
    """The run's end snapshot with missing totals, magnitudes and problem filled in."""
    state = dict(end)
    for kind in KINDS:
        a = start.get("total_" + kind)
        if state.get("total_" + kind) is None and a is not None:
            state["total_" + kind] = a + run_totals[kind]
    if n_trials and state.get("magnitudes") is None:
        state["magnitudes"] = _last_magnitudes(tv, n_trials)
    if state.get("problem") is None:
        state["problem"] = start.get("problem")
    return state


def _reconcile_run(df, tv, previous, carried, trial_offset, run_index):
    """Reconcile one run given the previous run's end state and totals so far.

    Args:
        df: Event frame of this run.
        tv: Trial variables of this run (may be empty).
        previous: End state of the previous run, or ``None`` for the first.
        carried: ``{kind: events counted in earlier runs of this session}``.
        trial_offset: Number of trials in earlier runs of this session.
        run_index: Position of this run in the session (for diagnostics only).
    """
    start, end = _snapshots(df)
    n_trials = len(tv.get("trial", []))
    incoming, events, diagnostics = _between_runs(previous, start, trial_offset, run_index)
    corrected, run_totals, observed = {}, {}, {}
    for kind in KINDS:
        values = tv.get(kind + "_reversals", [])
        observed[kind] = _counter_observed(values, start, kind)
        # Counters can persist across recording restarts: subtract the run-start
        # baseline rather than counting historical events again.
        baseline = start.get(kind) or 0
        counts = _running_counts(values, n_trials, baseline)
        last_count = counts[-1] if counts else 0
        corrected[kind] = [carried[kind] + incoming[kind] + c for c in counts]
        run_totals[kind] = _run_total(last_count, start, end, kind, baseline)
        terminal = run_totals[kind] - last_count
        if terminal:
            events.append(
                dict(kind=kind, index=trial_offset + n_trials, count=terminal, source="run_end")
            )
    if n_trials:
        diagnostics += _terminal_magnitude_diagnostics(
            tv, end, n_trials, trial_offset, events, run_index
        )
    added = {k: incoming[k] + run_totals[k] for k in KINDS}
    return _Run(
        n_trials,
        corrected,
        added,
        observed,
        events,
        diagnostics,
        _end_state(start, end, tv, n_trials, run_totals),
    )


def _correct_blocks(original_blocks, events):
    """Add the boundary events at or before each trial to the block counter."""
    # A session whose blocks are all None raises TypeError here (None + int); the
    # extractor is expected to supply integer blocks, so this is left to fail.
    return [
        b + sum(e["count"] for e in events if e["index"] <= i)
        for i, b in enumerate(original_blocks)
    ]


def _reconcile_session(sess, previous):
    """Reconcile every run of one session in place; return the last run's end state."""
    frames = _session_frames(sess)
    arrays = {k: [] for k in KINDS}
    totals = {k: 0 for k in KINDS}
    available = {k: False for k in KINDS}
    events, diagnostics = [], []
    trial_offset = 0
    for run_index, df in enumerate(frames):
        run = _reconcile_run(
            df, _run_variables(sess, run_index), previous, totals, trial_offset, run_index
        )
        for kind in KINDS:
            arrays[kind].extend(run.corrected[kind])
            totals[kind] += run.added[kind]
            available[kind] |= run.observed[kind]
        events += run.events
        diagnostics += run.diagnostics
        previous = run.end_state
        trial_offset += run.n_trials
    # Rebuild block corrections from the original extraction on repeat calls.
    original_blocks = sess.setdefault(
        "blocks_before_boundary_reconciliation", list(sess.get("blocks", []))
    )
    sess["blocks"] = _correct_blocks(original_blocks, events)
    sess["reversal_totals"] = {k: totals[k] for k in KINDS if available[k]}
    sess["reversal_boundary_events"] = events
    sess["reversal_diagnostics"] = diagnostics
    for kind in KINDS:
        if available[kind]:
            sess[kind + "_reversals"] = arrays[kind]
            sess["has_" + kind] = True
    return previous


def reconcile_reversals(data):
    """Attach corrected counters, totals, boundary events and diagnostics in place.

    Call after extracting *all* sessions and before selecting problems or early/
    late subsets. Re-running is safe: raw per-run trial variables are the source.
    Positive cumulative-total changes between runs identify additional events;
    negative changes and magnitude-only changes are diagnostics, not reversals.
    """
    for sessions in data.values():
        previous = None
        for key in sorted(sessions, key=_session_order):
            previous = _reconcile_session(sessions[key], previous)
    return data
