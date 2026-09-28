"""Observed session and cross-session block lengths for cohort summaries."""
import re
import pandas as pd
from src.behavior_analysis.get_variables_across_sessions import get_vars_across_all_sessions


def subject_numbers_from_paths(paths):
    numbers = {}
    for path in paths:
        match = re.search(r'sub-(\d+)_id-([^/]+)', str(path))
        if match:
            number, mouse = int(match[1]), match[2]
            if mouse in numbers and numbers[mouse] != number:
                raise ValueError(f'Conflicting subject numbers for {mouse}')
            numbers[mouse] = number
    return numbers


def build_trial_block_tables(problems, subject_numbers):
    """Return session and block tables; blocks continue across session boundaries.

    Counts include observed portions of first/last blocks. ``edge_block`` flags
    these potentially incomplete blocks. No missing sessions/trials are invented.
    Calendar time for a block is the date of its last observed trial.
    """
    sessions, blocks = [], []
    for problem, subjects in sorted(problems.items()):
        for mouse, mouse_sessions in subjects.items():
            number = subject_numbers[mouse]
            group = 'sub-01–04' if 1 <= number <= 4 else 'sub-05–08' if 5 <= number <= 8 else 'Other'
            base = dict(mouse=mouse, subject_number=number, group=group, problem=problem)
            keys = sorted(mouse_sessions, key=lambda k: (int(re.search(r'ses-(\d+)', k)[1]), k))
            dates = []
            for key in keys:
                count = len(mouse_sessions[key].get('trial', []))
                if not count:
                    continue
                date = pd.to_datetime(re.search(r'date-(\d{8})', key)[1], format='%Y%m%d')
                sessions.append(dict(base, session=key, date=date, trials=count))
                dates.extend([date] * count)
            merged, _ = get_vars_across_all_sessions({mouse: mouse_sessions})
            ids = merged[mouse]['blocks']
            if len(ids) != len(dates):
                raise ValueError(f'{mouse}, problem {problem}: blocks and trials are misaligned')
            starts = [i for i in range(len(ids)) if i == 0 or ids[i] != ids[i-1]]
            for j, start in enumerate(starts):
                stop = starts[j+1] if j+1 < len(starts) else len(ids)
                blocks.append(dict(base, block=j+1, block_id=ids[start], trials=stop-start,
                                   start_date=dates[start], date=dates[stop-1],
                                   edge_block=j == 0 or j == len(starts)-1))
    return pd.DataFrame(sessions), pd.DataFrame(blocks)
