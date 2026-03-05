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
    """
    trials_by_session: list of lists like [[1..n1], [1..n2], ...]
    returns one list with continuous numbering: [1..n1, n1+1..n1+n2, ...]
    """
    merged = []
    offset = 0
    for sess_trials in trials_by_session:
        merged.extend([t + offset for t in sess_trials])
        offset += len(sess_trials)
    return merged

def merge_list_of_dicts_of_lists(dict_list):
    """
    dict_list: [ {k: [..], ...}, {k: [..], ...}, ... ]
    returns:   {k: [..merged..], ...}
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
    """
    list_of_lists: e.g. [[0,0,1,1], [0,0,0,1,1,2,2], ...]
    Returns one list where each subsequent list is offset so the cumulative
    count carries over across sessions.

    The offset rule is: offset = last value of merged so far (not last+1).
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
    """
    blocks_by_session: list of lists like
      [[1,1,1,2,2], [1,1,2,2,3,3,3], ...]

    Returns one list where block IDs continue across sessions.
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
    """
    merged_num_blocks: list like [1,1,1,2,2,2,3,3,...]
    returns:           [1,2,3,1,2,3,1,2,...]
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