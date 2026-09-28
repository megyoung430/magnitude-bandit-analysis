"""Audit trial/end/start reversal state without inferring event type from magnitudes.

Usage: python scripts/audit_session_reversals.py RAW_ROOT OUTPUT_JSON
Includes terminal trials, multiple runs per session, counter resets and problem changes.
"""
import csv
import json
import re
import sys
from pathlib import Path

ALIASES = {
    'magnitudes': ('current_reward_magnitudes', 'curr_rew_mag'),
    'good': ('num_good_reversals', 'num_good_rev'),
    'bad': ('num_bad_reversals', 'num_bad_rev'),
    'total_good': ('tot_num_good_reversals', 'tot_num_good_rev'),
    'total_bad': ('tot_num_bad_reversals', 'tot_num_bad_rev'),
    'problem': ('num_problem',),
}


def state(d):
    return {name: next((d[k] for k in keys if k in d), None)
            for name, keys in ALIASES.items()}


def compare(before, after):
    changes = {}
    for key in ALIASES:
        a, b = before.get(key), after.get(key)
        if a is not None and b is not None and a != b:
            changes[key] = {'before': a, 'after': b}
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                changes[key]['delta'] = b - a
    return changes


def audit(root):
    runs = []
    for path in sorted(Path(root).rglob('*.tsv')):
        match = re.search(r'(sub-[^/]+)/(ses-(\d+)_date-\d+)', str(path))
        if not match:
            continue
        start, end, last = {}, {}, {}
        trial_count = 0
        with path.open() as handle:
            for row in csv.DictReader(handle, delimiter='\t'):
                if row['type'] != 'variable':
                    continue
                try:
                    d = json.loads(row['content'])
                except (ValueError, TypeError):
                    continue
                if not isinstance(d, dict):
                    continue
                if row['subtype'] == 'run_start':
                    start = state(d)
                elif row['subtype'] == 'run_end':
                    end = state(d)
                elif row['subtype'] == 'print' and 'num_t_found' not in d:
                    s = state(d)
                    if s['magnitudes'] is not None:
                        last = s
                        trial_count += 1
        if start.get('total_good') is None and start.get('good') is None:
            continue
        runs.append(dict(subject=match[1], session=match[2], session_number=int(match[3]),
                         path=str(path), start=start, end=end, last_trial=last,
                         trial_count=trial_count))
    runs.sort(key=lambda r: (r['subject'], r['session_number'], r['path']))
    boundaries = []
    previous = {}
    for run in runs:
        run['last_trial_to_end'] = compare(run['last_trial'], run['end'])
        prev = previous.get(run['subject'])
        if prev:
            boundaries.append(dict(subject=run['subject'], previous_session=prev['session'],
                next_session=run['session'], previous_path=prev['path'], next_path=run['path'],
                same_problem=prev['start'].get('problem') == run['start'].get('problem'),
                end_to_start=compare(prev['end'], run['start']),
                last_trial_to_start=compare(prev['last_trial'], run['start'])))
        previous[run['subject']] = run
    return {'runs': runs, 'boundaries': boundaries}


if __name__ == '__main__':
    result = audit(sys.argv[1])
    target = Path(sys.argv[2])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(f"Audited {len(result['runs'])} runs and {len(result['boundaries'])} boundaries; saved {target}")
