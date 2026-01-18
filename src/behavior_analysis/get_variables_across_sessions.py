def get_vars_across_all_sessions(data):
    subject_data_across_all_sessions = {}
    merged_subject_data_across_sessions = {}

    all_subjects = list(data.keys())
    for current_subject in all_subjects:
        data_across_sessions = {}
        subject_sessions = data[current_subject]  
        all_sessions = list(subject_sessions.keys())
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
                good_reversals_across_sessions.append(current_data['good_reversals'])
                bad_reversals_across_sessions.append(current_data['bad_reversals'])
                blocks_across_sessions.append(current_data['blocks'])
                trials_in_block_across_sessions.append(current_data['trials_in_block'])
                reward_magnitudes_by_tower_across_sessions.append(current_data['reward_magnitudes_by_tower'])
                choices_by_tower_across_sessions.append(current_data['choices_by_tower'])
                choices_by_rank_across_sessions.append(current_data['choices_by_rank'])
            else:
                print(f"[WARNING] No trial information found for subject {current_subject}, session {current_session}")
                continue
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
    for i, lst in enumerate(list_of_lists):
        if not lst:
            continue
        shifted = [int(x) + offset for x in lst]
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
