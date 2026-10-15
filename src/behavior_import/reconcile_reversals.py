"""Reconcile trial counters with run snapshots, without inventing trials.

Per-session arrays contain incoming boundary events. ``reversal_totals`` also
includes events after the last trial, which must be carried into the next
session by the merger. Unclassified state changes are retained as diagnostics.
"""

import json
import re
from typing import NamedTuple

KINDS = ("good", "bad")


class _SnapshotError(ValueError):
    """A run snapshot that must be parsed could not be; the caller adds the session."""


def _state(raw):
    aliases = {
        "good": ("num_good_reversals", "num_good_rev"),
        "bad": ("num_bad_reversals", "num_bad_rev"),
        "total_good": ("tot_num_good_reversals", "tot_num_good_rev"),
        "total_bad": ("tot_num_bad_reversals", "tot_num_bad_rev"),
        "magnitudes": ("current_reward_magnitudes", "curr_rew_mag"),
        "problem": ("num_problem",),
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


def _session_order(key):
    """Sort by session number, then name; keys without ``ses-<n>`` go last."""
    match = re.search(r"ses-(\d+)", key)
    return (int(match[1]) if match else float("inf"), key)


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
    # Total counters are independent evidence; take whichever source shows more events.
    start_total, end_total = start.get("total_" + kind), end.get("total_" + kind)
    if start_total is not None and end_total is not None and end_total >= start_total:
        total = max(total, int(end_total - start_total))
    return total


def _last_magnitudes(variables, n_trials):
    return variables.get("reward_magnitudes", [None] * n_trials)[-1]


def _terminal_magnitude_diagnostics(
    last_magnitudes, end, n_trials, trial_offset, events, run_index
):
    """Flag a magnitude change after the last trial that no terminal reversal explains."""
    explained = any(
        e["source"] == "run_end" and e["index"] == trial_offset + n_trials for e in events
    )
    if end.get("magnitudes") is None or last_magnitudes == end["magnitudes"] or explained:
        return []
    return [
        dict(
            source="unclassified_terminal_magnitude_change",
            run=run_index,
            before=last_magnitudes,
            after=end["magnitudes"],
        )
    ]


def _end_state(start, end, last_magnitudes, run_totals):
    """The run's end snapshot with missing totals, magnitudes and problem filled in.

    ``last_magnitudes`` is ``None`` for a run without trials.
    """
    state = dict(end)
    for kind in KINDS:
        start_total = start.get("total_" + kind)
        if state.get("total_" + kind) is None and start_total is not None:
            state["total_" + kind] = start_total + run_totals[kind]
    if last_magnitudes is not None and state.get("magnitudes") is None:
        state["magnitudes"] = last_magnitudes
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


def _reconcile_run(df, variables, previous, carried, trial_offset, run_index):
    """Reconcile one run given the previous run's end state and totals so far.

    Args:
        df: Event frame of this run.
        variables: Trial variables of this run (may be empty).
        previous: End state of the previous run, or ``None`` for the first.
        carried: ``{kind: events counted in earlier runs of this session}``.
        trial_offset: Number of trials in earlier runs of this session.
        run_index: Position of this run in the session (for diagnostics only).
    """
    try:
        start, end = _snapshots(df)
    except _SnapshotError as err:
        raise _SnapshotError(f"run {run_index}: {err}") from err
    n_trials = len(variables.get("trial", []))
    incoming, events, diagnostics = _between_runs(previous, start, trial_offset, run_index)
    corrected, run_totals, observed = {}, {}, {}
    for kind in KINDS:
        values = variables.get(kind + "_reversals", [])
        observed[kind] = _counter_observed(values, start, kind)
        corrected[kind], run_totals[kind], terminal = _reconcile_counter(
            values, start, end, kind, n_trials, carried[kind] + incoming[kind]
        )
        if terminal:
            events.append(
                dict(kind=kind, index=trial_offset + n_trials, count=terminal, source="run_end")
            )
    last_magnitudes = _last_magnitudes(variables, n_trials) if n_trials else None
    if n_trials:
        diagnostics += _terminal_magnitude_diagnostics(
            last_magnitudes, end, n_trials, trial_offset, events, run_index
        )
    return _Run(
        n_trials=n_trials,
        corrected=corrected,
        added={kind: incoming[kind] + run_totals[kind] for kind in KINDS},
        observed=observed,
        events=events,
        diagnostics=diagnostics,
        end_state=_end_state(start, end, last_magnitudes, run_totals),
    )


def _reconcile_runs(sess, previous):
    """Reconcile every run of a session in order; return ``(runs, last end state)``."""
    runs = []
    carried = {kind: 0 for kind in KINDS}
    trial_offset = 0
    for run_index, df in enumerate(_session_frames(sess)):
        run = _reconcile_run(
            df, _run_variables(sess, run_index), previous, carried, trial_offset, run_index
        )
        for kind in KINDS:
            carried[kind] += run.added[kind]
        previous = run.end_state
        trial_offset += run.n_trials
        runs.append(run)
    return runs, previous


def _combine_runs(runs):
    """Merge per-run results into session-level values.

    Returns ``(arrays, totals, observed, events, diagnostics)``, each dict keyed by
    kind except the two lists, which keep run order.
    """
    arrays = {kind: [c for run in runs for c in run.corrected[kind]] for kind in KINDS}
    totals = {kind: sum(run.added[kind] for run in runs) for kind in KINDS}
    observed = {kind: any(run.observed[kind] for run in runs) for kind in KINDS}
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


def _write_session(sess, arrays, totals, observed, events, diagnostics):
    """Store the reconciled values on the session dict (the only mutation)."""
    # Rebuild block corrections from the original extraction on repeat calls.
    original_blocks = sess.setdefault(
        "blocks_before_boundary_reconciliation", list(sess.get("blocks", []))
    )
    sess["blocks"] = _correct_blocks(original_blocks, events)
    sess["reversal_totals"] = {k: totals[k] for k in KINDS if observed[k]}
    sess["reversal_boundary_events"] = events
    sess["reversal_diagnostics"] = diagnostics
    for kind in KINDS:
        if observed[kind]:
            sess[kind + "_reversals"] = arrays[kind]
            sess["has_" + kind] = True


def _reconcile_session(sess, previous):
    """Reconcile one session in place; return the last run's end state."""
    runs, last_end_state = _reconcile_runs(sess, previous)
    _write_session(sess, *_combine_runs(runs))
    return last_end_state


def reconcile_reversals(data):
    """Attach corrected counters, totals, boundary events and diagnostics in place.

    Call after extracting *all* sessions and before selecting problems or early/
    late subsets. Re-running is safe: raw per-run trial variables are the source.
    Positive cumulative-total changes between runs identify additional events;
    negative changes and magnitude-only changes are diagnostics, not reversals.

    Raises:
        ValueError: If a run's ``run_start`` snapshot cannot be parsed; the message
            names the subject and session.
    """
    for subject, sessions in data.items():
        previous = None
        for key in sorted(sessions, key=_session_order):
            try:
                previous = _reconcile_session(sessions[key], previous)
            except _SnapshotError as err:
                raise ValueError(f"{subject} {key}: {err}") from err
    return data
