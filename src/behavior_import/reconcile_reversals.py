"""Reconcile trial counters with run snapshots, without inventing trials.

Per-session arrays contain incoming boundary events. ``reversal_totals`` also
includes events after the last trial, which must be carried into the next
session by the merger. Unclassified state changes are retained as diagnostics.
"""
import json
import re

KINDS = ('good', 'bad')


def _state(d):
    aliases = {
        'good': ('num_good_reversals', 'num_good_rev'),
        'bad': ('num_bad_reversals', 'num_bad_rev'),
        'total_good': ('tot_num_good_reversals', 'tot_num_good_rev'),
        'total_bad': ('tot_num_bad_reversals', 'tot_num_bad_rev'),
        'magnitudes': ('current_reward_magnitudes', 'curr_rew_mag'),
        'problem': ('num_problem',),
    }
    return {k: next((d[a] for a in names if a in d), None)
            for k, names in aliases.items()}


def _snapshots(df):
    start, end = {}, {}
    for row in df.loc[df['type'] == 'variable'].to_dict('records'):
        if row['subtype'] not in ('run_start', 'run_end'):
            continue
        content = row['content']
        try:
            d = content if isinstance(content, dict) else json.loads(content)
        except (ValueError, TypeError):
            continue
        if row['subtype'] == 'run_start':
            start = _state(d)
        else:
            end = _state(d)
    return start, end


def _same_problem(a, b):
    # Require positive evidence of continuity; never bridge different arm sets.
    am, bm = a.get('magnitudes'), b.get('magnitudes')
    if not am or not bm or set(am) != set(bm):
        return False
    return (a.get('problem') is None or b.get('problem') is None
            or a['problem'] == b['problem'])


def reconcile_reversals(data):
    """Attach corrected counters, totals, boundary events and diagnostics in place.

    Call after extracting *all* sessions and before selecting problems or early/
    late subsets. Re-running is safe: raw per-run trial variables are the source.
    Positive cumulative-total changes between runs identify additional events;
    negative changes and magnitude-only changes are diagnostics, not reversals.
    """
    for sessions in data.values():
        previous = None
        for key in sorted(sessions, key=lambda k: (int(re.search(r'ses-(\d+)', k)[1])
                                                   if re.search(r'ses-(\d+)', k) else float('inf'), k)):
            sess = sessions[key]
            frames = sess.get('data', [])
            frames = frames if isinstance(frames, list) else [frames]
            variables = sess.get('trial_variables', [])
            arrays = {k: [] for k in KINDS}
            totals = {k: 0 for k in KINDS}
            available = {k: False for k in KINDS}
            events, diagnostics = [], []
            trial_offset = 0
            for run_index, df in enumerate(frames):
                start, end = _snapshots(df)
                tv = variables[run_index] if isinstance(variables, list) and run_index < len(variables) else {}
                n = len(tv.get('trial', []))
                incoming = {k: 0 for k in KINDS}
                if previous and _same_problem(previous, start):
                    for kind in KINDS:
                        a, b = previous.get('total_' + kind), start.get('total_' + kind)
                        if a is not None and b is not None:
                            delta = int(b - a)
                            if delta > 0:
                                incoming[kind] = delta
                                events.append(dict(kind=kind, index=trial_offset, count=delta,
                                                   source='between_runs'))
                            elif delta < 0:
                                diagnostics.append(dict(kind=kind, source='counter_rollback', delta=delta,
                                                        run=run_index))
                    if previous.get('magnitudes') != start.get('magnitudes') and not any(incoming.values()):
                        diagnostics.append(dict(source='unclassified_magnitude_change', run=run_index,
                                                before=previous.get('magnitudes'), after=start.get('magnitudes')))
                effective_end = dict(end)
                for kind in KINDS:
                    vals = tv.get(kind + '_reversals', [])
                    available[kind] |= (any(v is not None for v in vals) or start.get(kind) is not None
                                        or start.get('total_' + kind) is not None)
                    # Counters can persist across recording restarts: subtract the
                    # run-start baseline rather than counting historical events again.
                    baseline = start.get(kind) or 0
                    count = 0
                    corrected = []
                    for i in range(n):
                        value = vals[i] if i < len(vals) else None
                        if value is not None:
                            count = max(count, int(value) - baseline)
                        corrected.append(totals[kind] + incoming[kind] + count)
                    arrays[kind].extend(corrected)
                    final = end.get(kind)
                    run_total = max(count, int(final) - baseline) if final is not None else count
                    # Total counters are an independent fallback for end snapshots.
                    a, b = start.get('total_' + kind), end.get('total_' + kind)
                    if a is not None and b is not None and b >= a:
                        run_total = max(run_total, int(b - a))
                    terminal = run_total - count
                    if terminal:
                        events.append(dict(kind=kind, index=trial_offset + n, count=terminal,
                                           source='run_end'))
                    totals[kind] += incoming[kind] + run_total
                    if effective_end.get('total_' + kind) is None and a is not None:
                        effective_end['total_' + kind] = a + run_total
                if n:
                    last_mags = tv.get('reward_magnitudes', [None] * n)[-1]
                    if end.get('magnitudes') is not None and last_mags != end['magnitudes'] and not any(
                            e['source'] == 'run_end' and e['index'] == trial_offset + n for e in events):
                        diagnostics.append(dict(source='unclassified_terminal_magnitude_change', run=run_index,
                                                before=last_mags, after=end['magnitudes']))
                    if effective_end.get('magnitudes') is None:
                        effective_end['magnitudes'] = last_mags
                if effective_end.get('problem') is None:
                    effective_end['problem'] = start.get('problem')
                previous = effective_end
                trial_offset += n
            # Rebuild block corrections from the original extraction on repeat calls.
            original_blocks = sess.setdefault('blocks_before_boundary_reconciliation', list(sess.get('blocks', [])))
            sess['blocks'] = [b + sum(e['count'] for e in events if e['index'] <= i)
                              for i, b in enumerate(original_blocks)]
            sess['reversal_totals'] = {k: totals[k] for k in KINDS if available[k]}
            sess['reversal_boundary_events'] = events
            sess['reversal_diagnostics'] = diagnostics
            for kind in KINDS:
                if available[kind]:
                    sess[kind + '_reversals'] = arrays[kind]
                    sess['has_' + kind] = True
    return data
