"""Functions for merging and aligning behavioral variables across multiple recording sessions."""

import re
from datetime import datetime

def get_vars_across_all_sessions(data):
    """Merge aligned trials, carrying terminal reversals into the next session.

    End-of-session events affect totals even without a following trial. They
    enter trial-aligned counters only when another trial is available. Sessions
    retain their incoming boundary corrections when selected as a subset.
    """
    merged, unmerged = {}, {}
    for subject, subject_sessions in data.items():
        def order(key):
            match = re.search(r"ses-(\d+)", key)
            return (int(match[1]) if match else float("inf"), key)

        sessions = [subject_sessions[k] for k in sorted(subject_sessions, key=order)]
        fields = ('trial', 'blocks', 'trials_in_block', 'reward_magnitudes_by_tower',
                  'choices_by_tower', 'choices_by_rank')
        parts = {key: [s.get(key, []) for s in sessions if s.get('trial')]
                 for key in fields}
        result = {'trial': merge_trials_across_sessions(parts['trial'])}
        for key in ('reward_magnitudes_by_tower', 'choices_by_tower', 'choices_by_rank'):
            result[key] = merge_list_of_dicts_of_lists(parts[key])
        for kind in ('good', 'bad'):
            key = kind + '_reversals'
            if not any(key in s for s in sessions):
                continue
            offset, values = 0, []
            parts[key] = []
            for sess in sessions:
                arr = sess.get(key, [])
                n = len(sess.get('trial', []))
                if arr and len(arr) != n:
                    raise ValueError(f"{subject}: {key} length does not match trial count")
                filled, last = [], 0
                for i in range(n):
                    v = arr[i] if arr else None
                    if v is not None:
                        last = int(v)
                    filled.append(last)
                parts[key].append(filled)
                values.extend(offset + v for v in filled)
                offset += sess.get('reversal_totals', {}).get(kind, last)
            result[key] = values
        # Preserve existing block numbering, adding terminal events that have
        # no trial in their own session (within-session additions are extracted).
        block_values, offset = [], 0
        for sess in sessions:
            blocks = sess.get('blocks', [])
            block_values.extend(b + offset for b in blocks)
            if blocks:
                offset = block_values[-1] - 1
            n = len(sess.get('trial', []))
            offset += sum(e['count'] for e in sess.get('reversal_boundary_events', [])
                          if e['index'] == n)
        result['blocks'] = block_values
        result['trials_in_block'] = compute_merged_num_trials_in_block(block_values)
        merged[subject], unmerged[subject] = result, parts
    return merged, unmerged

# ========== Merging Rules for Variables of Interest ==========
def merge_trials_across_sessions(trials_by_session):
    """Concatenate per-session trial index lists into a single continuous sequence.

    Each session's trial indices are offset so the final merged list is
    monotonically increasing (1 … n1, n1+1 … n1+n2, …).

    Args:
        trials_by_session: List of lists, e.g.
            ``[[1, 2, …, n1], [1, 2, …, n2], …]``.

    Returns:
        A single flat list of re-indexed trial numbers spanning all sessions.
    """
    merged = []
    offset = 0
    for sess_trials in trials_by_session:
        merged.extend([t + offset for t in sess_trials])
        offset += len(sess_trials)
    return merged

def merge_list_of_dicts_of_lists(dict_list):
    """Concatenate a list of ``{key: list}`` dicts into a single merged dict.

    Each per-session dict maps variable names to per-trial value lists.  The
    merged dict concatenates the lists in order so the result covers all
    sessions for each key.

    Args:
        dict_list: List of dicts, each of the form ``{key: [values, …]}``.
            ``None`` dicts and ``None`` value lists are silently skipped.

    Returns:
        A single dict ``{key: [merged_values, …]}`` covering all sessions.
    """
    merged = {}
    for d in dict_list:
        if d is None:
            continue
        for k, v in d.items():
            if v is None:
                continue
            merged.setdefault(k, []).extend(v)
    return merged

def merge_reversals_across_sessions(list_of_lists, start_offset=0):
    """Merge per-session cumulative reversal count lists across sessions.

    Each session list contains cumulative reversal counts that reset to zero
    at the start of the session.  This function offsets each session by the
    last value of the previously merged output so that the final list is
    monotonically non-decreasing across all sessions.

    The offset rule is: ``offset = last value of merged so far`` (not last+1),
    so the carry-over is seamless.

    Args:
        list_of_lists: List of lists, e.g.
            ``[[0, 0, 1, 1], [0, 0, 0, 1, 2], …]``.
            ``None`` values within sublists are skipped.
        start_offset: Initial offset to add to the first session's values
            (default: ``0``).

    Returns:
        A single flat list of monotonically non-decreasing reversal counts.
    """
    merged = []
    offset = start_offset

    for lst in list_of_lists:
        if not lst:
            continue

        shifted = []
        for x in lst:
            if x is None:
                continue
            shifted.append(int(x) + offset)

        if not shifted:
            continue

        merged.extend(shifted)
        offset = merged[-1]
    return merged

def merge_blocks_across_sessions(blocks_by_session):
    """Merge per-session block-ID lists so IDs are continuous across sessions.

    Each session's block IDs start at 1.  The merged list offsets each session
    by the last block ID of the previous session, so block numbering is
    continuous across all sessions.

    Args:
        blocks_by_session: List of lists, e.g.
            ``[[1, 1, 2, 2], [1, 1, 2, 2, 3], …]``.
            Empty sublists are skipped.

    Returns:
        A single flat list of block IDs with continuous numbering.
    """
    merged = []
    offset = 0
    for blocks in blocks_by_session:
        if not blocks:
            continue
        shifted = [b + offset for b in blocks]
        merged.extend(shifted)
        offset = merged[-1] - 1
    return merged

def compute_merged_num_trials_in_block(merged_num_blocks):
    """Compute per-trial within-block trial index from a merged block-ID list.

    For each position in *merged_num_blocks*, returns the 1-based trial index
    within the current block (i.e. resets to 1 at every block transition).

    Args:
        merged_num_blocks: List of block IDs, e.g. ``[1, 1, 1, 2, 2, 3, …]``,
            as returned by :func:`merge_blocks_across_sessions`.

    Returns:
        List of within-block trial counts, e.g. ``[1, 2, 3, 1, 2, 1, …]``.
    """
    out = []
    prev_block = None
    count = 0
    for b in merged_num_blocks:
        if b != prev_block:
            count = 1
            prev_block = b
        else:
            count += 1
        out.append(count)
    return out