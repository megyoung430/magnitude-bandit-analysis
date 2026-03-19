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
    "choice_towers": ("current_choice_towers", "curr_choice_tows", "c_tows"),

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
    """Extract trial variables and group sessions into problems by choice tower set.

    A convenience wrapper that calls :func:`extract_trials` to parse all
    trial-level variables, then :func:`group_sessions_by_problem` to split
    sessions into contiguous problems whenever the set of available choice
    towers changes.

    Args:
        data: Nested dict ``{subject: {session_key: session_dict}}`` as returned
            by :func:`src.behavior_import.import_data.import_data`.

    Returns:
        Dict ``{problem_id: {subject: {session_key: session_dict}}}`` where
        *problem_id* is a 1-based integer that increments each time the choice
        tower signature changes within a subject.
    """
    data = extract_trials(data)
    problems = group_sessions_by_problem(data)
    return problems

def extract_trials(data):
    """Parse and attach trial-by-trial variables to every session in *data*.

    For each subject and session, reads the raw ``"data"`` DataFrame(s),
    filters to variable-print rows, applies alias normalisation, handles
    multi-file sessions by concatenating segments with the appropriate merging
    rules, and adds derived fields (reward magnitude features, choice one-hots,
    rank one-hots).

    Args:
        data: Nested dict ``{subject: {session_key: session_dict}}`` where each
            ``session_dict`` contains at least a ``"data"`` key holding a
            ``pd.DataFrame`` or list of ``pd.DataFrame`` objects.

    Returns:
        The same *data* dict, mutated in-place, with each ``session_dict``
        extended by the following keys (when trial information is available):

        - ``"trial"``, ``"blocks"``, ``"good_reversals"``, ``"bad_reversals"``,
          ``"trials_in_block"``, ``"choice"``, ``"chosen_rank"``, etc. —
          standardised trial variables.
        - ``"reward_magnitudes_by_tower"``, ``"choices_by_tower"``,
          ``"choices_by_rank"`` — derived per-trial one-hot / magnitude arrays.
        - ``"has_good"`` / ``"has_bad"`` (bool) — whether performance-dependent
          reversal variables were logged.
        - ``"trial_info"`` and ``"trial_variables"`` — raw parsed lists retained
          for debugging.
    """
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
    """Return a copy of *dictionary* with variable names normalised to canonical keys.

    Looks up each canonical key (from *aliases*) against its possible alias
    names and returns the first matching value found, or ``None`` if none of
    the aliases appear in *dictionary*.  For example, both
    ``"num_good_reversals"`` and ``"num_good_rev"`` are mapped to the
    canonical name ``"good_reversals"``.

    Args:
        dictionary: A single-trial dict of raw variable names and values.
        aliases: Dict mapping canonical name → tuple of alias strings, as
            defined in :data:`ALIASES`.

    Returns:
        A new dict with exactly the canonical keys from *aliases*, each mapped
        to the first matching value found in *dictionary* (or ``None``).
    """
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
    """Transpose a list of per-trial dicts into a dict of per-variable lists.

    Applies alias normalisation to each trial dict and accumulates values into
    parallel lists, one per canonical variable name.  Missing values (where no
    alias key is present in a trial dict) become ``None``.

    Args:
        trial_dicts: List of per-trial raw dicts.
        aliases: Dict mapping canonical name → tuple of alias strings, as
            defined in :data:`ALIASES`.
        keep: Optional iterable of canonical key names to retain.  If ``None``
            (default), all keys in *aliases* are included.

    Returns:
        Dict ``{canonical_key: [per_trial_values]}`` with one entry per
        retained key.
    """
    standardized_keys = list(aliases.keys()) if keep is None else list(keep)
    out = {k: [] for k in standardized_keys}

    for d in trial_dicts:
        canon = standardize_variables(d, aliases)
        for k in standardized_keys:
            out[k].append(canon.get(k))
    return out

# --- Data Unpacking Helper Functions ---
def unpack_reward_magnitudes(session_data):
    """Reformat per-trial reward magnitude dicts into per-tower value lists.

    Reads ``session_data["reward_magnitudes"]`` (a list of ``{tower: magnitude}``
    dicts, one per trial) and transposes it into a dict of per-tower lists stored
    under ``session_data["reward_magnitudes_by_tower"]``.

    Args:
        session_data: Session dict containing a ``"reward_magnitudes"`` key whose
            value is a non-empty list of dicts mapping tower names to magnitudes.

    Returns:
        The same *session_data* dict with ``"reward_magnitudes_by_tower"`` added.
    """
    choice_towers = session_data['reward_magnitudes'][0].keys()
    magnitude_by_tower = {k: [] for k in choice_towers}
    for i in range(len(session_data['reward_magnitudes'])):
        current_reward_magnitudes = session_data['reward_magnitudes'][i]
        for tower in choice_towers:
            magnitude_by_tower[tower].append(current_reward_magnitudes[tower])
    session_data['reward_magnitudes_by_tower'] = magnitude_by_tower
    return session_data

def unpack_choices(session_data):
    """Build per-tower binary choice indicator lists from the raw choice sequence.

    For each trial, sets the chosen tower's indicator to 1 and all others to 0.
    Stores the result under ``session_data["choices_by_tower"]``.

    Args:
        session_data: Session dict containing ``"choice_towers"`` (a list whose
            first element gives the available tower identifiers) and ``"choice"``
            (a per-trial list of the tower chosen on each trial).

    Returns:
        The same *session_data* dict with ``"choices_by_tower"`` added.
    """
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
    """Standardise rank labels and build per-rank choice indicator lists.

    Normalises ``session_data["chosen_rank"]`` values (e.g. ``"second_best"`` →
    ``"second"``) and creates a ``"choices_by_rank"`` dict of one-hot lists for
    each rank (``"best"``, ``"second"``, ``"third"``).

    Args:
        session_data: Session dict containing ``"chosen_rank"`` (list of raw rank
            strings) and ``"rank_counts"`` (list of per-trial cumulative count
            dicts used to derive the canonical rank keys).

    Returns:
        The same *session_data* dict with ``"chosen_rank"`` normalised in-place
        and ``"choices_by_rank"`` added.
    """
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
    """Infer missing ``chosen_rank`` entries from ``rank_counts`` increments.

    When ``session_data["chosen_rank"][i]`` is ``None``, computes the delta
    between consecutive ``rank_counts`` dicts to determine which rank
    incremented and fills in that rank label.  Existing non-``None`` entries
    are never modified.

    Args:
        session_data: Session dict containing ``"chosen_rank"`` (list, may have
            ``None`` entries) and ``"rank_counts"`` (list of cumulative count
            dicts, same length as ``"chosen_rank"``).

    Returns:
        The same *session_data* dict with ``"chosen_rank"`` filled in where
        possible.
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
        """Normalise rank key *k* to ``"best"``, ``"second"``, or ``"third"``."""
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
    """Return ``True`` if any trial dict in *trial_info* contains at least one of *keys*.

    Used to detect whether a session logged performance-dependent reversal
    variables (e.g. ``"num_good_reversals"``), distinguishing performance-
    independent from performance-dependent task variants.

    Args:
        trial_info: List of per-trial dicts.
        keys: Tuple of key strings to search for.

    Returns:
        ``True`` if at least one dict in *trial_info* contains any key from
        *keys*; ``False`` otherwise (including when *trial_info* is empty).
    """
    if not trial_info:
        return False
    for d in trial_info:
        if isinstance(d, dict) and any(k in d for k in keys):
            return True
    return False

# --- Merging Rules for Multiple Files within a Session ---
def concat_serial_numeric(segments):
    """Renumber serial trial-index segments into a single continuous sequence.

    When a session spans multiple recording files, trial indices restart from
    the beginning of each file.  This function offsets each subsequent segment
    so the merged list is monotonically increasing (e.g. ``23`` then ``1`` →
    ``24``).

    Args:
        segments: List of lists of integers (or ``None`` values) representing
            per-file trial index sequences.

    Returns:
        A single flat list with continuous trial numbering.
    """
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
    """Merge cumulative counter segments across recording files.

    Cumulative counters (e.g. reversal counts) may restart at the beginning
    of a later file if the recording was interrupted.  This function offsets
    each segment by the last value of the previous segment when a reset is
    detected (i.e. the current segment starts below the running maximum).

    Args:
        segments: List of lists of numbers (or ``None`` values) representing
            per-file cumulative counter sequences.

    Returns:
        A single flat list of monotonically non-decreasing cumulative counts.
    """
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
        if start < prev_last:
            # The segment reset — but check if it's continuing the same block.
            # If it resets to 1 and the previous segment ended mid-block (prev_last > 1),
            # the first file ended mid-block and the second file resumes it,
            # so offset by prev_last - 1 to keep the same block number continuous.
            if start == 1:
                offset = prev_last - 1
            else:
                offset = prev_last
            seg_vals = [x + offset for x in seg_vals]
        out.extend(seg_vals)
        prev_last = out[-1]
    return out

def add_dicts(a, b):
    """Return a new dict that is the elementwise sum of *a* and *b*.

    Missing keys in either dict are treated as ``0``.  Non-integer-convertible
    values in *b* are silently ignored and the existing value from *a* is kept.

    Args:
        a: Base dict (or ``None`` / empty dict).
        b: Dict of increments to add (or ``None`` / empty dict).

    Returns:
        A new dict containing the union of keys from *a* and *b* with their
        values summed.
    """
    out = dict(a) if a else {}
    if b:
        for k, v in b.items():
            try:
                out[k] = int(out.get(k, 0)) + int(v)
            except Exception:
                out[k] = out.get(k, 0)
    return out

def offset_dict_list_cumulative(segments):
    """Merge cumulative dict-valued counter segments across recording files.

    Each element of each segment is a per-trial dict of cumulative counts
    (e.g. ``{"best": 3, "second": 1}``).  Later segments are offset by the
    last dict of the current merged output so the merged sequence is
    monotonically non-decreasing.

    Args:
        segments: List of lists-of-dicts, one inner list per recording file.

    Returns:
        A single flat list of cumulative-count dicts spanning all files.
    """
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
    """Merge per-file within-block trial index sequences across recording files.

    The within-block counter resets to ``1`` at the start of each new block.
    When a segment continues a block that was open at the end of the previous
    segment, the leading values (before the first ``1``) are offset by the
    previous segment's last value.  Values from the first reset to ``1``
    onward are kept as-is.

    Args:
        segments: List of lists of integers (or ``None`` values) representing
            per-file within-block trial index sequences.

    Returns:
        A single flat list of within-block trial indices that resets to ``1``
        only at genuine block transitions.
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
    """Return a hashable signature for the session's set of available choice towers.

    Reads the first non-``None`` entry of ``session_data["choice_towers"]``
    and converts it to a stable hashable form (tuple for lists/tuples/dicts,
    or a one-element tuple for strings).

    Args:
        session_data: Session dict containing a ``"choice_towers"`` key.

    Returns:
        A hashable tuple representing the tower set, or ``None`` if
        ``"choice_towers"`` is absent or all values are ``None``.
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

    if isinstance(first, (list, tuple)):
        towers_part = tuple(sorted(first, key=str))
    elif isinstance(first, dict):
        towers_part = tuple(sorted(first.keys()))
    elif isinstance(first, str):
        towers_part = (first,)
    else:
        towers_part = (repr(first),)

    default_initiation_tower = "B2"
    init_tower = session_data.get("initiation_tower", None)
    if init_tower is None:
        init_tower = default_initiation_tower

    return (towers_part, init_tower)

def normalize_tower_token(x):
    """Normalise a single tower token into a sortable, comparable tuple.

    Converts letter+digit tokens (e.g. ``"A1"``, ``" b02 "``) to
    ``(LETTER, int)`` tuples for robust comparison.  Falls back gracefully
    for raw strings, numbers, and nested tuples.

    Args:
        x: A tower token — typically a string like ``"A1"`` or ``"C3"``, a
            number, or a tuple of such values.

    Returns:
        A normalised ``(prefix, value)`` tuple:

        - String matching ``[A-Za-z]+[0-9]+`` → ``(UPPER_LETTER, int)``.
        - Other strings → ``("RAW", stripped_string)``.
        - Integers/floats → ``("NUM", x)``.
        - Tuples → recursively normalised tuple.

    Raises:
        TypeError: If *x* is an unsupported type.
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
    """Canonicalise a tower-set signature to be permutation-invariant.

    Normalises each token with :func:`normalize_tower_token` and sorts the
    result so that two tower sets with the same members but different orderings
    produce identical signatures.

    Rule of thumb:

    - Sequence-like (list/tuple/set/frozenset) → unordered multiset,
      canonicalised by normalising each token and sorting.
    - Dict → sort normalised ``(key, value)`` pairs.
    - Single string → normalise it.
    - ``None`` → kept as ``None``.

    Args:
        raw_sig: A raw signature as returned by :func:`choice_towers_signature`:
            a sequence, dict, string, or ``None``.

    Returns:
        A sorted, normalised tuple suitable for equality comparison, or
        ``None`` if *raw_sig* is ``None``.
    """
    if raw_sig is None:
        return None

    if (isinstance(raw_sig, tuple) and len(raw_sig) == 2
            and isinstance(raw_sig[0], tuple)
            and isinstance(raw_sig[1], str)):
        towers_norm = tuple(sorted(normalize_tower_token(x) for x in raw_sig[0]))
        init_norm = normalize_tower_token(raw_sig[1])
        return (towers_norm, init_norm)

    if isinstance(raw_sig, (list, tuple, set, frozenset)):
        normalized = [normalize_tower_token(x) for x in raw_sig]
        return tuple(sorted(normalized))

    if isinstance(raw_sig, dict):
        items = [(normalize_tower_token(k), normalize_tower_token(v)) for k, v in raw_sig.items()]
        return tuple(sorted(items))

    if isinstance(raw_sig, str):
        return normalize_tower_token(raw_sig)

    return raw_sig

def group_sessions_by_problem(data, copy_sessions=True):
    """Group sessions into numbered problems based on per-subject problem ordinal.

    Within each subject, sessions are sorted chronologically. Each time a new
    (previously unseen) tower signature appears for that subject, it is assigned
    the next ordinal (1, 2, 3, ...). If a signature recurs later for the same
    subject, it is folded back into its original ordinal (A -> B -> A yields
    ordinals 1, 2, 1 -- not 1, 2, 3).

    Problem numbers are then shared across subjects by ordinal position: every
    subject's 3rd problem is grouped under problem ID 3, regardless of which
    tower set it was.

    Args:
        data: Nested dict ``{subject: {session_key: session_dict}}`` after
            :func:`extract_trials` has been applied.  Sessions without trial
            information are skipped with a warning.
        copy_sessions: If ``True`` (default), each session dict is deep-copied
            into the output to avoid aliasing the input.

    Returns:
        Dict ``{problem_id: {subject: {session_key: session_dict}}}`` where
        *problem_id* is a 1-based integer representing the Nth distinct tower
        set each subject encountered, in chronological order.
    """

    def sort_ses_date(session_id: str):
        """Sort by session number (ses-XX) then date (date-YYYYMMDD)."""
        m_ses = re.search(r"ses-(\d+)", session_id)
        m_date = re.search(r"date-(\d{8})", session_id)
        ses_num = int(m_ses.group(1)) if m_ses else float("inf")
        dt = datetime.strptime(m_date.group(1), "%Y%m%d") if m_date else datetime.max
        return (ses_num, dt, session_id)

    problems = defaultdict(lambda: defaultdict(dict))

    for subject_id, subject_sessions in data.items():
        sessions_sorted = sorted(subject_sessions.keys(), key=sort_ses_date)

        # Per-subject: maps signature -> 1-based ordinal in the order first seen.
        # next_ordinal is a separate counter so that None-signature sessions don't
        # collide with real problem ordinals.
        sig_to_ordinal: dict = {}
        next_ordinal = 1

        for session_id in sessions_sorted:
            sess = subject_sessions[session_id]

            if not sess.get("trial_info"):
                print(f"[WARNING] Skipping session {session_id} for subject {subject_id} due to missing trial information.")
                continue

            raw_sig = choice_towers_signature(sess)
            sig = permutation_invariant_signature(raw_sig)

            # Include has_good in the signature to distinguish task variants
            # with the same towers but different performance-dependence
            has_good = sess.get("has_good", False)
            full_sig = (sig, has_good) if sig is not None else None

            if full_sig is not None and full_sig in sig_to_ordinal:
                num_problem = sig_to_ordinal[full_sig]
            elif full_sig is not None:
                num_problem = next_ordinal
                sig_to_ordinal[full_sig] = num_problem
                next_ordinal += 1
            else:
                num_problem = max(next_ordinal - 1, 1)

            problems[num_problem][subject_id][session_id] = deepcopy(sess) if copy_sessions else sess

    return {p: {s: dict(ss) for s, ss in subj.items()} for p, subj in problems.items()}

# --- Adding Variables to Analyze Value --- 
def add_reward_magnitude_features(session_data):
    """Compute and attach reward-magnitude derived features to a session dict.

    Adds three new keys:

    - ``"chosen_magnitude"`` — per-trial magnitude of the chosen arm.
    - ``"reward_magnitude_by_arm"`` — ``{tower: [magnitude_per_trial]}`` for
      every arm seen across the session.
    - ``"reward_magnitude_by_rank"`` — ``{"best": […], "second": […], "third": […]}``
      where ranks are assigned by descending magnitude on each trial.

    Also aliases ``"reward_magnitude_by_arm"`` under the legacy key
    ``"reward_magnitudes_by_tower"``.

    Args:
        session_data: Session dict containing:

        - ``"reward_magnitudes"`` — list of per-trial ``{tower: magnitude}``
          dicts.
        - ``"choice"`` — list of chosen tower tokens (same length).
        - ``"choice_towers"`` (optional) — list of available tower sets per
          trial; falls back to ``reward_magnitudes[t].keys()`` if absent.

    Returns:
        The same *session_data* dict with the three derived keys added.
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
        """Return the list of available tower identifiers for trial index *t*."""
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