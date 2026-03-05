"""Functions for merging and aligning behavioral variables across multiple recording sessions."""

import re
from datetime import datetime

def get_vars_across_all_sessions(data):
    """Merge per-session trial data across all sessions for each subject.

    Args:
        data: Nested dict of the form
            ``data[subject][session_key] = session_dict``,
            where each ``session_dict`` must contain at least a ``"trial_info"``
            flag and the standard trial variables (``"trial"``, ``"blocks"``, etc.).

    Returns:
        A tuple ``(merged, unmerged)`` where:

        - ``merged[subject]`` – a single dict of concatenated arrays, with
          reversal counts shifted so they are monotonically increasing across
          sessions.
        - ``unmerged[subject]`` – the same data in per-session list form,
          before merging.
    """

    def sort_ses_date(session_id: str):
        """
        Sort by session number (ses-XX) then date (date-YYYYMMDD).
        Falls back to session_id string if parsing fails.
        """
        m_ses = re.search(r"ses-(\d+)", session_id)
        m_date = re.search(r"date-(\d{8})", session_id)

        ses_num = int(m_ses.group(1)) if m_ses else float("inf")
        dt = datetime.strptime(m_date.group(1), "%Y%m%d") if m_date else datetime.max

        return (ses_num, dt, session_id)
    
    def clean_reversal_list(lst):
        """Return list of ints with None removed. If lst is None, return None."""
        if lst is None:
            return None
        out = []
        for x in lst:
            if x is None:
                continue
            try:
                out.append(int(x))
            except Exception:
                continue
        return out

    subject_data_across_all_sessions = {}
    merged_subject_data_across_sessions = {}

    all_subjects = list(data.keys())
    for current_subject in all_subjects:
        data_across_sessions = {}
        subject_sessions = data[current_subject]  
        all_sessions = sorted(subject_sessions.keys(), key=sort_ses_date)
        trials_across_sessions = []
        good_reversals_across_sessions = []
        bad_reversals_across_sessions = []
        blocks_across_sessions = []
        trials_in_block_across_sessions = []
        reward_magnitudes_by_tower_across_sessions = []
        choices_by_tower_across_sessions = []
        choices_by_rank_across_sessions = []
        for current_session in all_sessions:
            current_data = data[current_subject][current_session]
            if current_data["trial_info"]:
                trials_across_sessions.append(current_data['trial'])
                gr = clean_reversal_list(current_data.get("good_reversals", None))
                br = clean_reversal_list(current_data.get("bad_reversals", None))
                if gr and len(gr) > 0:
                    good_reversals_across_sessions.append(gr)
                if br and len(br) > 0:
                    bad_reversals_across_sessions.append(br)
                blocks_across_sessions.append(current_data['blocks'])
                trials_in_block_across_sessions.append(current_data['trials_in_block'])
                reward_magnitudes_by_tower_across_sessions.append(current_data['reward_magnitudes_by_tower'])
                choices_by_tower_across_sessions.append(current_data['choices_by_tower'])
                choices_by_rank_across_sessions.append(current_data['choices_by_rank'])
            else:
                print(f"[WARNING] No trial information found for subject {current_subject}, session {current_session}")
                continue
        have_good = len(good_reversals_across_sessions) > 0
        have_bad  = len(bad_reversals_across_sessions) > 0
        if have_good and have_bad:
            data_across_sessions = {
                'trial': trials_across_sessions,
                'good_reversals': good_reversals_across_sessions,
                'bad_reversals': bad_reversals_across_sessions,
                'blocks': blocks_across_sessions,
                'trials_in_block': trials_in_block_across_sessions,
                'reward_magnitudes_by_tower': reward_magnitudes_by_tower_across_sessions,
                'choices_by_tower': choices_by_tower_across_sessions,
                'choices_by_rank': choices_by_rank_across_sessions
            }
            merged_data_across_sessions = {
                'trial': merge_trials_across_sessions(data_across_sessions['trial']),
                'good_reversals': merge_reversals_across_sessions(data_across_sessions['good_reversals']),
                'bad_reversals': merge_reversals_across_sessions(data_across_sessions['bad_reversals']),
                'blocks': merge_blocks_across_sessions(data_across_sessions['blocks']),
                'trials_in_block': compute_merged_num_trials_in_block(merge_blocks_across_sessions(data_across_sessions['blocks'])),
                'reward_magnitudes_by_tower': merge_list_of_dicts_of_lists(data_across_sessions['reward_magnitudes_by_tower']),
                'choices_by_tower': merge_list_of_dicts_of_lists(data_across_sessions['choices_by_tower']),
                'choices_by_rank': merge_list_of_dicts_of_lists(data_across_sessions['choices_by_rank'])
            }
        else:
            data_across_sessions = {
                'trial': trials_across_sessions,
                'blocks': blocks_across_sessions,
                'trials_in_block': trials_in_block_across_sessions,
                'reward_magnitudes_by_tower': reward_magnitudes_by_tower_across_sessions,
                'choices_by_tower': choices_by_tower_across_sessions,
                'choices_by_rank': choices_by_rank_across_sessions
            }
            merged_data_across_sessions = {
                'trial': merge_trials_across_sessions(data_across_sessions['trial']),
                'blocks': merge_blocks_across_sessions(data_across_sessions['blocks']),
                'trials_in_block': compute_merged_num_trials_in_block(merge_blocks_across_sessions(data_across_sessions['blocks'])),
                'reward_magnitudes_by_tower': merge_list_of_dicts_of_lists(data_across_sessions['reward_magnitudes_by_tower']),
                'choices_by_tower': merge_list_of_dicts_of_lists(data_across_sessions['choices_by_tower']),
                'choices_by_rank': merge_list_of_dicts_of_lists(data_across_sessions['choices_by_rank'])
            }
        subject_data_across_all_sessions[current_subject] = data_across_sessions
        merged_subject_data_across_sessions[current_subject] = merged_data_across_sessions
    return merged_subject_data_across_sessions, subject_data_across_all_sessions

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