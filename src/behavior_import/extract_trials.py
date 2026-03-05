"""Extract and process trial-by-trial data from behavioral experiment recordings."""

import re
import json
import pandas as pd
from copy import deepcopy
from collections import defaultdict
from datetime import datetime

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

TOWER_TOKEN_RE = re.compile(r"^\s*([A-Za-z]+)\s*(\d+)\s*$")

def extract_trials_grouped_by_problem(data):
    data = extract_trials(data) 
    problems = group_sessions_by_problem(data)
    return problems

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
                raw_trial_info = trials_df["content"].map(safe_json_load).tolist()

                # Drop any trials where mice are finding ports, i.e., ones that contain "num_t_found"
                trial_info = [
                    d for d in raw_trial_info
                    if not (isinstance(d, dict) and "num_t_found" in d)
                ]

                trial_info_list.append(trial_info)
                trial_vars = transpose_trials(trial_info, ALIASES)
                trial_vars_list.append(trial_vars)
            if len(trial_info_list[0]) == 0:
                print(f"[WARNING] No trial information found for subject {current_subject}, session {current_session}")
                data[current_subject][current_session]["trial_variables"] = {}
                data[current_subject][current_session]["trial_info"] = []
                continue

            # Check if the session had good or bad reversals (i.e., was performance dependent)
            flat_trial_info = []
            for ti in trial_info_list:
                flat_trial_info.extend(ti)
            has_good = session_has_any_key(flat_trial_info, ALIASES["good_reversals"])
            has_bad  = session_has_any_key(flat_trial_info, ALIASES["bad_reversals"])
            data[current_subject][current_session]["has_good"] = has_good
            data[current_subject][current_session]["has_bad"] = has_bad

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
            data[current_subject][current_session] = add_reward_magnitude_features(data[current_subject][current_session])
            data[current_subject][current_session] = unpack_choices(data[current_subject][current_session])
            data[current_subject][current_session] = fill_missing_chosen_rank_from_rank_counts(data[current_subject][current_session])
            data[current_subject][current_session] = unpack_chosen_rank(data[current_subject][current_session])
            
            # If a session never printed good/bad reversal vars, remove them entirely
            # so downstream code doesn't try to merge int(None).
            sess = data[current_subject][current_session]
            if not sess.get("has_good", False):
                sess.pop("good_reversals", None)

            if not sess.get("has_bad", False):
                sess.pop("bad_reversals", None)
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

def fill_missing_chosen_rank_from_rank_counts(session_data):
    """
    If session_data['chosen_rank'][i] is None, infer it from rank_counts increments.
    Assumes rank_counts is a list of dicts with cumulative counts per rank per trial.
    Does NOT change existing non-None chosen_rank entries.
    """
    if "chosen_rank" not in session_data or "rank_counts" not in session_data:
        return session_data

    chosen = session_data.get("chosen_rank", [])
    rank_counts = session_data.get("rank_counts", [])

    if not chosen or not rank_counts or len(chosen) != len(rank_counts):
        return session_data

    key_map = {
        "second_best": "second",
        "third_best": "third",
        "second best": "second",
        "third best": "third",
        "best": "best",
        "second": "second",
        "third": "third",
    }

    def standardized_key(k):
        if k is None:
            return k
        if isinstance(k, str):
            return key_map.get(k.strip().lower(), key_map.get(k, k.strip().lower()))
        return k

    standardized_rank_counts = []
    for d in rank_counts:
        if not isinstance(d, dict):
            standardized_rank_counts.append({})
            continue
        dd = {}
        for k, v in d.items():
            kk = standardized_key(k)
            if kk in ("best", "second", "third"):
                dd[kk] = v
        standardized_rank_counts.append(dd)

    filled = list(chosen)

    for i in range(len(filled)):
        if filled[i] is not None:
            continue

        curr = standardized_rank_counts[i]
        prev = standardized_rank_counts[i - 1] if i > 0 else {}

        deltas = {rk: (curr.get(rk, 0) or 0) - (prev.get(rk, 0) or 0) for rk in ("best", "second", "third")}

        inc = [rk for rk, dv in deltas.items() if dv > 0]
        if len(inc) == 1:
            filled[i] = inc[0]
        elif i == 0:
            rk0 = max(("best", "second", "third"), key=lambda rk: curr.get(rk, 0) or 0)
            if (curr.get(rk0, 0) or 0) > 0:
                filled[i] = rk0
        else:
            pass

    session_data["chosen_rank"] = filled
    return session_data

def session_has_any_key(trial_info, keys):
    """
    trial_info: list[dict]
    keys: tuple[str]
    Returns True if any trial dict contains any of the keys. This is mainly used to identify problems that are performance
    independent v. performance dependent.
    """
    if not trial_info:
        return False
    for d in trial_info:
        if isinstance(d, dict) and any(k in d for k in keys):
            return True
    return False

# --- Merging Rules for Multiple Files within a Session ---
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

# --- Grouping Sessions into Problems Based on Choice Towers ---
def choice_towers_signature(session_data):
    """
    Returns a hashable signature for the session's current choice towers.
    Uses the first non-None entry in session_data['choice_towers'].
    """
    towers_seq = session_data.get("choice_towers", None)
    if not towers_seq:
        return None

    first = None
    for x in towers_seq:
        if x is not None:
            first = x
            break
    if first is None:
        return None

    # Make it hashable + stable
    if isinstance(first, (list, tuple)):
        return tuple(first)
    if isinstance(first, dict):
        # if stored as dict, treat the keys as the set/order-defining items
        return tuple(sorted(first.keys()))
    if isinstance(first, str):
        return (first,)
    return (repr(first),)

def normalize_tower_token(x):
    """
    Normalize a single tower token into a sortable, comparable tuple.

    Examples:
      "A1"  -> ("A", 1)
      " b02 " -> ("B", 2)
      "C1"  -> ("C", 1)
      other strings -> ("RAW", "<stripped>")
    """
    if isinstance(x, str):
        s = x.strip()
        m = TOWER_TOKEN_RE.match(s)
        if m:
            return (m.group(1).upper(), int(m.group(2)))
        return ("RAW", s)

    # If your signature can include richer token types, extend this section.
    # For now, keep numbers comparable as-is.
    if isinstance(x, (int, float)):
        return ("NUM", x)

    # If you already produce normalized tuples in choice_towers_signature,
    # keep them comparable by normalizing elements recursively.
    if isinstance(x, tuple):
        return tuple(normalize_tower_token(e) for e in x)

    # Fail fast for unsupported objects so you don't silently create bad grouping.
    raise TypeError(f"Unsupported tower token type: {type(x)!r} (value={x!r})")

def permutation_invariant_signature(raw_sig):
    """
    Make the signature permutation-invariant.

    Rule of thumb:
      - If it's sequence-like (list/tuple/set/frozenset): treat as an unordered multiset,
        canonicalize by normalizing each token and sorting.
      - If it's a dict: canonicalize order by sorting normalized (key, value) pairs.
      - If it's a single string token: normalize it.
      - If None: keep None.
      - Otherwise: return as-is (or raise, depending on your preference).
    """
    if raw_sig is None:
        return None

    if isinstance(raw_sig, (list, tuple, set, frozenset)):
        normalized = [normalize_tower_token(x) for x in raw_sig]
        return tuple(sorted(normalized))

    if isinstance(raw_sig, dict):
        # dict equality is order-independent, but we canonicalize anyway for determinism
        items = [(normalize_tower_token(k), normalize_tower_token(v)) for k, v in raw_sig.items()]
        return tuple(sorted(items))

    if isinstance(raw_sig, str):
        return normalize_tower_token(raw_sig)

    return raw_sig

def group_sessions_by_problem(data, copy_sessions=True):
    """
    Returns dict-of-dicts:
      problems[problem_id][subject_id][session_id] = session_data

    Problem increments *within each subject* when choice towers change from one session to the next.

    Now permutation-invariant with respect to the tower signature returned by choice_towers_signature().
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

    problems = defaultdict(lambda: defaultdict(dict))

    for subject_id, subject_sessions in data.items():
        sessions_sorted = sorted(subject_sessions.keys(), key=sort_ses_date)

        num_problem = 1
        prev_sig = None

        for session_id in sessions_sorted:
            sess = subject_sessions[session_id]

            # Skip sessions that have no trial information
            if not sess.get("trial_info"):
                print(f"[WARNING] Skipping session {session_id} for subject {subject_id} due to missing trial information.")
                continue

            raw_sig = choice_towers_signature(sess)
            sig = permutation_invariant_signature(raw_sig)

            # Only increment when we have two comparable signatures and they differ
            if prev_sig is not None and sig is not None and sig != prev_sig:
                num_problem += 1

            if sig is not None:
                prev_sig = sig

            problems[num_problem][subject_id][session_id] = deepcopy(sess) if copy_sessions else sess

    return {p: {s: dict(ss) for s, ss in subj.items()} for p, subj in problems.items()}

# --- Adding Variables to Analyze Value --- 
def add_reward_magnitude_features(session_data):
    """
    Adds:
      - chosen_magnitude: list[float|int|None]
      - reward_magnitude_by_arm: dict[tower -> list[magnitude|None]]
      - reward_magnitude_by_rank: dict['best'|'second'|'third' -> list[magnitude|None]]

    Assumptions:
      - session_data['reward_magnitudes'] is a list of dicts per trial: {tower: magnitude, ...}
      - session_data['choice'] is a list of chosen tower tokens per trial (same length)
      - session_data['choice_towers'] is a list per trial (optional but supported). If missing/None,
        we fall back to reward_magnitudes[t].keys().
    """
    mags_seq = session_data.get("reward_magnitudes", [])
    choice_seq = session_data.get("choice", [])
    towers_seq = session_data.get("choice_towers", None)

    if not mags_seq or not isinstance(mags_seq, list):
        return session_data

    n = len(mags_seq)

    # Make lengths consistent where possible
    if not choice_seq or len(choice_seq) != n:
        choice_seq = [None] * n

    # Build per-trial available tower list
    def towers_for_trial(t):
        if towers_seq and isinstance(towers_seq, list) and t < len(towers_seq) and towers_seq[t] is not None:
            # could be list/tuple/dict/etc
            x = towers_seq[t]
            if isinstance(x, dict):
                return list(x.keys())
            if isinstance(x, (list, tuple, set)):
                return list(x)
            if isinstance(x, str):
                return [x]
        d = mags_seq[t]
        return list(d.keys()) if isinstance(d, dict) else []

    # Union of all towers seen across trials (stable-ish order)
    all_towers = []
    seen = set()
    for t in range(n):
        for tw in towers_for_trial(t):
            if tw not in seen:
                seen.add(tw)
                all_towers.append(tw)

    # reward_magnitude_by_arm: tower -> per-trial magnitude (None if tower not present)
    mag_by_arm = {tw: [None] * n for tw in all_towers}
    for t in range(n):
        d = mags_seq[t] if isinstance(mags_seq[t], dict) else {}
        for tw in all_towers:
            if tw in d:
                mag_by_arm[tw][t] = d[tw]

    # chosen_magnitude
    chosen_mag = [None] * n
    for t in range(n):
        ch = choice_seq[t]
        d = mags_seq[t] if isinstance(mags_seq[t], dict) else {}
        if ch in d:
            chosen_mag[t] = d[ch]

    # reward_magnitude_by_rank (computed per trial from available towers)
    rank_keys = ("best", "second", "third")
    mag_by_rank = {rk: [None] * n for rk in rank_keys}

    for t in range(n):
        d = mags_seq[t] if isinstance(mags_seq[t], dict) else {}
        avail = towers_for_trial(t)
        # Keep only towers that have a magnitude defined
        pairs = [(tw, d.get(tw, None)) for tw in avail]
        pairs = [(tw, m) for tw, m in pairs if m is not None]

        if not pairs:
            continue

        # Sort by magnitude desc; tie-break deterministically by tower token string
        pairs_sorted = sorted(pairs, key=lambda x: (-x[1], str(x[0])))

        # Assign top-3 magnitudes
        for i, rk in enumerate(rank_keys):
            if i < len(pairs_sorted):
                mag_by_rank[rk][t] = pairs_sorted[i][1]

    session_data["chosen_magnitude"] = chosen_mag
    session_data["reward_magnitude_by_arm"] = mag_by_arm
    session_data["reward_magnitude_by_rank"] = mag_by_rank

    # (Optional) keep/alias your older field name if downstream expects it:
    session_data["reward_magnitudes_by_tower"] = mag_by_arm

    return session_data