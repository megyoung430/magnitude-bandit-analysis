import json
import pandas as pd
from copy import deepcopy
from collections import defaultdict

# Across some pyControl files, different variable names are used for the same trial information.
# The following dictionary maps standardized variable names to their possible aliases.
ALIASES = {
    # trial counters
    "trial": ("num_trials", "num_t"),
    "blocks": ("num_blocks",),
    "good_reversals": ("num_good_reversals", "num_good_rev"),
    "bad_reversals": ("num_bad_reversals", "num_bad_rev"),
    "trials_in_block": ("num_trials_in_block", "num_t_in_block"),
    "trials_since_last_reversal": ("num_trials_since_last_reversal", "num_t_since_last_reversal"),

    # chosen tower and rank
    "choice": ("chosen_tower", "chosen_tow"),
    "chosen_rank": ("chosen_rank",),

    # long pokes
    "num_long_pokes": ("num_long_pokes_before_choice", "num_long_pokes"),
    "long_pokes": ("long_pokes_before_choice","long_pokes"),

    # reward
    "reward_pulses": ("num_reward_pulses", "num_rew_pulses"),
    "reward_magnitudes": ("current_reward_magnitudes", "curr_rew_mag"),
    "total_reward_amount": ("total_reward_amount", "tot_rew_amt"),

    # best arm + available choice towers
    "best_arm": ("best_arm",),
    "choice_towers": ("current_choice_towers", "curr_choice_tows"),

    # counts
    "choice_counts": ("choice_counts",),
    "rank_counts": ("rank_counts",),

    # moving averages
    "ema_best_arm_choices": ("ema_best_arm_choices", "ema_best_choices"),
    "ema_second_arm_choices": ("ema_second_arm_choices", "ema_second_choices"),

    # reversal triggering
    "can_switch": ("can_switch",),
    "num_trials_until_switch": ("num_trials_until_switch", "num_t_switch"),
    "total_num_trials_until_switch": ("total_num_trials_until_switch", "tot_num_t_switch"),
}

# Sometimes, due to errors, we need to restart the pyControl recording midway through. These are 
# variables that should be treated as continuous across such restarts.
SERIAL_VARS = {"trial"}
CUMULATIVE_NUMERIC_VARS = {"good_reversals", "bad_reversals", "blocks"}
CUMULATIVE_DICTLIST_VARS = {"rank_counts", "choice_counts"}
SPECIAL_VARS = {"trials_in_block"}

def extract_trials(data):
    def safe_json_load(x):
        """Parse JSON safely, returning empty dict on failure."""
        if pd.isna(x):
            return {}
        if isinstance(x, (dict, list)):
            return x
        try:
            return json.loads(x)
        except Exception:
            return {}
    
    # Extract trials across subjects
    all_subjects = list(data.keys())
    for current_subject in all_subjects:
        subject_sessions = data[current_subject]

        # Extract trials across sessions
        all_sessions = list(subject_sessions.keys())
        for current_session in all_sessions:
            df_list = subject_sessions[current_session]["data"]
            
            # Check if there are multiple dataframes within a particular session
            # If not, turn the dataframe into a list of one dataframe for consistency
            if isinstance(df_list, pd.DataFrame):
                df_list = [df_list]
            
            trial_info_list = []
            trial_vars_list = []
            for df in df_list:
                trials_df = df[(df.get("type", "") == "variable") & (df.get("subtype", "").isin(["print"]))].sort_values("time")
                trial_info = trials_df["content"].map(safe_json_load).tolist()
                trial_info_list.append(trial_info)
                trial_vars = transpose_trials(trial_info, ALIASES)
                trial_vars_list.append(trial_vars)
            if len(trial_info_list[0]) == 0:
                print(f"[WARNING] No trial information found for subject {current_subject}, session {current_session}")
                data[current_subject][current_session]["trial_variables"] = {}
                data[current_subject][current_session]["trial_info"] = []
                continue
            data[current_subject][current_session]["trial_variables"] = trial_vars_list
            data[current_subject][current_session]["trial_info"] = trial_info_list
            if len(trial_vars_list) == 1:
                for var_name, values in trial_vars_list[0].items():
                    data[current_subject][current_session][var_name] = values
            else:
                print(f"[INFO] Merging multiple files for subject {current_subject}, session {current_session}")
                combined = {}
                all_var_names = set().union(*[tv.keys() for tv in trial_vars_list])
                for var_name in all_var_names:
                    segments = [tv.get(var_name, []) for tv in trial_vars_list]
                    if var_name in SERIAL_VARS:
                        combined[var_name] = concat_serial_numeric(segments)
                    elif var_name in CUMULATIVE_NUMERIC_VARS:
                        combined[var_name] = concat_cumulative_numeric(segments)
                    elif var_name in CUMULATIVE_DICTLIST_VARS:
                        combined[var_name] = offset_dict_list_cumulative(segments)
                    elif var_name in SPECIAL_VARS:
                        combined[var_name] = concat_trials_in_block(segments)
                    else:
                        out = []
                        for seg in segments:
                            if seg:
                                out.extend(seg)
                        combined[var_name] = out
                for var_name, values in combined.items():
                    data[current_subject][current_session][var_name] = values
            data[current_subject][current_session] = unpack_reward_magnitudes(data[current_subject][current_session])
            data[current_subject][current_session] = unpack_choices(data[current_subject][current_session])
            data[current_subject][current_session] = unpack_chosen_rank(data[current_subject][current_session])
    return data

def standardize_variables(dictionary, aliases):
    """Returns a dictionary with the trial names standardized according to the aliases, e.g., "num_good_reversals" and
    "num_good_rev" will now both become "good_reversals" for pyControl files with different variable names."""
    out = {}
    for variable, names in aliases.items():
        if isinstance(names, str):
            names = (names,)
        val = None
        for name in names:
            if name in dictionary:
                val = dictionary[name]
                break
        out[variable] = val
    return out

def transpose_trials(trial_dicts, aliases, keep=None):
    """Convert list[dict] (trials) -> dict[standardized_variable] = list[values] Keeps trial alignment; missing values become None."""
    standardized_keys = list(aliases.keys()) if keep is None else list(keep)
    out = {k: [] for k in standardized_keys}

    for d in trial_dicts:
        canon = standardize_variables(d, aliases)
        for k in standardized_keys:
            out[k].append(canon.get(k))
    return out

# --- Data Unpacking Helper Functions ---
def unpack_reward_magnitudes(session_data):
    choice_towers = session_data['reward_magnitudes'][0].keys()
    magnitude_by_tower = {k: [] for k in choice_towers}
    for i in range(len(session_data['reward_magnitudes'])):
        current_reward_magnitudes = session_data['reward_magnitudes'][i]
        for tower in choice_towers:
            magnitude_by_tower[tower].append(current_reward_magnitudes[tower])
    session_data['reward_magnitudes_by_tower'] = magnitude_by_tower
    return session_data

def unpack_choices(session_data):
    choice_towers = session_data['choice_towers'][0]
    choices_by_tower = {k: [] for k in choice_towers}
    for i in range(len(session_data['choice'])):
        current_choice = session_data['choice'][i]
        for tower in choice_towers:
            if current_choice == tower:
                choices_by_tower[tower].append(1)
            else:
                choices_by_tower[tower].append(0)
    session_data['choices_by_tower'] = choices_by_tower
    return session_data

def unpack_chosen_rank(session_data):
    # Map inconsistent labels to standardized labels
    rank_map = {
        "second_best": "second",
        "third_best": "third",
        "second": "second",
        "third": "third",
        "best": "best",
    }

    # Standardize chosen_rank values
    chosen_std = [rank_map.get(r, r) for r in session_data["chosen_rank"]]
    session_data["chosen_rank"] = chosen_std

    # Standardize rank keys from rank_counts (won't vary within session)
    possible_ranks = []
    for k in session_data["rank_counts"][0].keys():
        k2 = rank_map.get(k, k)
        if k2 not in possible_ranks:
            possible_ranks.append(k2)

    # Build one-hot choices_by_rank with standardized keys
    choices_by_rank = {k: [] for k in possible_ranks}
    for r in chosen_std:
        for k in possible_ranks:
            choices_by_rank[k].append(1 if r == k else 0)

    session_data["choices_by_rank"] = choices_by_rank
    return session_data

# ========== Merging Rules for Multiple Files within a Session ==========
def concat_serial_numeric(segments):
    """1..N serial renumbering across segments (23 then 1 -> 24...)."""
    out = []
    offset = 0
    for seg in segments:
        if not seg:
            continue
        seg_vals = [x for x in seg if x is not None]
        if not seg_vals:
            continue
        out.extend([x + offset for x in seg_vals])
        offset = out[-1]  # keep continuity
    return out

def concat_cumulative_numeric(segments):
    """Cumulative counters: offset each segment by previous segment's last value."""
    out = []
    prev_last = None
    for seg in segments:
        if not seg:
            continue
        seg_vals = [x for x in seg if x is not None]
        if not seg_vals:
            continue
        if prev_last is None:
            out.extend(seg_vals)
            prev_last = out[-1]
            continue
        start = seg_vals[0]
        # Only offset if the counter reset
        if start < prev_last:
            seg_vals = [x + prev_last for x in seg_vals]
        out.extend(seg_vals)
        prev_last = out[-1]
    return out

def add_dicts(a, b):
    """Elementwise add dict b into dict a (returns new dict). Missing keys treated as 0."""
    out = dict(a) if a else {}
    if b:
        for k, v in b.items():
            try:
                out[k] = int(out.get(k, 0)) + int(v)
            except Exception:
                out[k] = out.get(k, 0)
    return out

def offset_dict_list_cumulative(segments):
    """segments: list of lists-of-dicts (per trial). Each later segment gets + last dict of previous cumulative output."""
    out = []
    offset = {}
    for seg in segments:
        if not seg:
            continue
        adjusted = []
        for d in seg:
            if not isinstance(d, dict):
                adjusted.append(d)
                continue
            adjusted.append(add_dicts(d, offset))
        out.extend(adjusted)
        for d in reversed(out):
            if isinstance(d, dict):
                offset = d
                break
    return out

def concat_trials_in_block(segments):
    """Continuous numeric until it resets to 1 (new block).
    For each segment after the first:
      - find first index where value == 1
      - offset values before that by previous segment's last value
      - keep values from the first reset onward unchanged
    """
    out = []
    last_val = 0

    first = True
    for seg in segments:
        if not seg:
            continue
        seg_vals = [x for x in seg if x is not None]
        if not seg_vals:
            continue

        if first:
            out.extend(seg_vals)
            last_val = out[-1]
            first = False
            continue

        # Locate first reset-to-1
        try:
            reset_idx = seg_vals.index(1)
        except ValueError:
            reset_idx = None

        if reset_idx is None:
            # If there is no reset, treat as continuation
            adjusted = [x + last_val for x in seg_vals]
        else:
            before = [x + last_val for x in seg_vals[:reset_idx]]
            after = seg_vals[reset_idx:]  # keep as-is from the reset
            adjusted = before + after

        out.extend(adjusted)
        last_val = out[-1]
    return out