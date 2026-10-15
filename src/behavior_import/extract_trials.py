"""Extract and process trial-by-trial data from behavioral experiment recordings."""

import re
from copy import deepcopy
from datetime import datetime

import pandas as pd

from src.behavior_import.import_data import parse_content_json
from src.behavior_import.problem_numbering import (  # noqa: F401  (re-exported)
    SessionFacts,
    normalize_tower_token,
    number_subject_sessions,
    permutation_invariant_signature,
)
from src.behavior_import.reconcile_reversals import reconcile_reversals
from src.behavior_import.split_recording_configurations import split_recording_configurations

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
    "long_pokes": ("long_pokes_before_choice", "long_pokes"),
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

# A recording that logs none of these has no bandit trials (e.g. port finding or pretraining).
BANDIT_KEYS = ALIASES["reward_magnitudes"] + ALIASES["choice_towers"] + ALIASES["rank_counts"]

# Sometimes, due to errors, we need to restart the pyControl recording midway through. These are
# variables that should be treated as continuous across such restarts.
SERIAL_VARS = {"trial"}
CUMULATIVE_NUMERIC_VARS = {"good_reversals", "bad_reversals", "blocks"}
CUMULATIVE_DICTLIST_VARS = {"rank_counts", "choice_counts"}
SPECIAL_VARS = {"trials_in_block"}

# Standard rank labels, best first.
RANK_KEYS = ("best", "second", "third")

# Rank-label spellings (including space-separated) mapped to the standard labels.
# unpack_chosen_rank uses a narrower map: no space-separated spellings, no strip/lower.
RANK_LABEL_MAP = {
    "second_best": "second",
    "third_best": "third",
    "second best": "second",
    "third best": "third",
    "best": "best",
    "second": "second",
    "third": "third",
}


def extract_trials_grouped_by_problem(data):
    """Extract trial variables and group sessions into problems by tower layout.

    A convenience wrapper that calls :func:`extract_trials` to parse all
    trial-level variables, then :func:`group_sessions_by_problem` to split
    sessions into problems (see :mod:`src.behavior_import.problem_numbering`).

    Args:
        data: Nested dict ``{subject: {session_key: session_dict}}`` as returned
            by :func:`src.behavior_import.import_data.import_data`.

    Returns:
        Dict ``{problem_id: {subject: {session_key: session_dict}}}``; sessions
        flagged as suspected settings errors are left out.
    """
    data = extract_trials(data)
    problems = group_sessions_by_problem(data)
    return problems


def recording_trial_info(df):
    """Return the per-trial dicts printed by one recording, in time order.

    Port-finding trials (those that contain ``"num_t_found"``) are dropped.
    A recording lacking a ``subtype`` column fails here with an ``AttributeError``
    (the ``""`` default has no ``.isin``). A recording lacking a ``type`` column
    yields no trials here, without an error; the pipeline then fails later with
    ``KeyError: 'type'`` in ``reconcile_reversals``.
    """
    # Known issue: with no ``type`` column, ``df.get("type", "") == "variable"`` is False, so this
    # returns [] instead of raising.
    is_print = (df.get("type", "") == "variable") & (df.get("subtype", "").isin(["print"]))
    printed = df[is_print].sort_values("time")
    raw_trial_info = printed["content"].map(parse_content_json).tolist()
    return [d for d in raw_trial_info if not (isinstance(d, dict) and "num_t_found" in d)]


def bandit_trial_info(df, subject, session_key):
    """Return a recording's trial dicts, or ``[]`` if it is not a bandit recording.

    The cohort directory can also contain port-finding and initiation
    pretraining. Those print trials but no bandit fields; keeping them would
    create all-``None`` bandit variables.
    """
    trial_info = recording_trial_info(df)
    if trial_info and not session_has_any_key(trial_info, BANDIT_KEYS):
        print(
            "[WARNING] Skipping recording with no bandit trial fields "
            f"for subject {subject}, session {session_key}"
        )
        return []
    return trial_info


def merge_segments(var_name, segments):
    """Merge one variable's per-recording lists using the rule for that variable."""
    if var_name in SERIAL_VARS:
        return concat_serial_numeric(segments)
    if var_name in CUMULATIVE_NUMERIC_VARS:
        return concat_cumulative_numeric(segments)
    if var_name in CUMULATIVE_DICTLIST_VARS:
        return offset_dict_list_cumulative(segments)
    if var_name in SPECIAL_VARS:
        return concat_trials_in_block(segments)
    return [value for segment in segments if segment for value in segment]


def merge_trial_variables(trial_vars_list):
    """Combine the per-recording ``{variable: list}`` dicts of one session into one."""
    # Variables come out of a set, so the key order of a merged session depends on PYTHONHASHSEED
    # and is not reproducible across runs.
    all_var_names = set().union(*[tv.keys() for tv in trial_vars_list])
    return {
        var_name: merge_segments(var_name, [tv.get(var_name, []) for tv in trial_vars_list])
        for var_name in all_var_names
    }


def session_trial_variables(trial_vars_list, subject, session_key):
    """Return one ``{variable: list}`` dict for a session, merging its recordings if several."""
    if len(trial_vars_list) == 1:
        return trial_vars_list[0]
    print(f"[INFO] Merging multiple files for subject {subject}, session {session_key}")
    return merge_trial_variables(trial_vars_list)


def add_derived_fields(session_data):
    """Add the per-tower and per-rank fields derived from the standardised trial variables."""
    # Known issue: unpack_reward_magnitudes writes ``reward_magnitudes_by_tower`` and
    # add_reward_magnitude_features then overwrites it (it also includes towers from
    # ``choice_towers``), so the order of these two steps changes the result.
    # fill_missing_chosen_rank_from_rank_counts must precede unpack_chosen_rank, which reads
    # ``chosen_rank``.
    for step in (
        unpack_reward_magnitudes,
        add_reward_magnitude_features,
        unpack_choices,
        fill_missing_chosen_rank_from_rank_counts,
        unpack_chosen_rank,
    ):
        session_data = step(session_data)
    return session_data


def drop_unlogged_reversal_vars(session_data):
    """Remove good/bad reversal counters from sessions that never logged them.

    Downstream code would otherwise try to merge ``int(None)``.
    """
    if not session_data.get("has_good", False):
        session_data.pop("good_reversals", None)
    if not session_data.get("has_bad", False):
        session_data.pop("bad_reversals", None)
    return session_data


def extract_session_trials(session_data, subject, session_key):
    """Attach trial variables and derived fields to one session, in place.

    A session without any bandit trial information gets empty
    ``"trial_variables"`` and ``"trial_info"`` and a warning.
    """
    # A session is one dataframe, or a list of them when the recording was restarted
    frames = session_data["data"]
    if isinstance(frames, pd.DataFrame):
        frames = [frames]

    trial_info_list = [bandit_trial_info(df, subject, session_key) for df in frames]
    trial_vars_list = [transpose_trials(trial_info, ALIASES) for trial_info in trial_info_list]
    if not any(trial_info_list):
        print(f"[WARNING] No trial information found for subject {subject}, session {session_key}")
        session_data["trial_variables"] = {}
        session_data["trial_info"] = []
        return session_data

    # Performance-dependent sessions log good/bad reversal counters
    flat_trial_info = [d for trial_info in trial_info_list for d in trial_info]
    session_data["has_good"] = session_has_any_key(flat_trial_info, ALIASES["good_reversals"])
    session_data["has_bad"] = session_has_any_key(flat_trial_info, ALIASES["bad_reversals"])
    session_data["trial_variables"] = trial_vars_list
    session_data["trial_info"] = trial_info_list
    session_data.update(session_trial_variables(trial_vars_list, subject, session_key))
    return drop_unlogged_reversal_vars(add_derived_fields(session_data))


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
    data = split_recording_configurations(data)
    for subject, subject_sessions in data.items():
        for session_key, session_data in subject_sessions.items():
            subject_sessions[session_key] = extract_session_trials(
                session_data, subject, session_key
            )
    return reconcile_reversals(data)


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
            value is a list of dicts mapping tower names to magnitudes.
            Missing trials or tower values remain ``None`` without shifting
            trial indices. Towers are collected across all trials.

    Returns:
        The same *session_data* dict with ``"reward_magnitudes_by_tower"`` added.
    """
    magnitudes = session_data.get("reward_magnitudes", [])
    choice_towers = dict.fromkeys(
        tower for values in magnitudes if isinstance(values, dict) for tower in values
    )
    magnitude_by_tower = {
        tower: [values.get(tower) if isinstance(values, dict) else None for values in magnitudes]
        for tower in choice_towers
    }
    session_data["reward_magnitudes_by_tower"] = magnitude_by_tower
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
    choice_towers = session_data["choice_towers"][0]
    choices_by_tower = {k: [] for k in choice_towers}
    for i in range(len(session_data["choice"])):
        current_choice = session_data["choice"][i]
        for tower in choice_towers:
            if current_choice == tower:
                choices_by_tower[tower].append(1)
            else:
                choices_by_tower[tower].append(0)
    session_data["choices_by_tower"] = choices_by_tower
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


def standardize_rank_key(k):
    """Normalise rank key *k* to ``"best"``, ``"second"``, or ``"third"``.

    Unrecognised strings come back stripped and lower-cased; ``None`` and
    non-strings are returned unchanged.
    """
    if k is None:
        return k
    if isinstance(k, str):
        norm = k.strip().lower()
        return RANK_LABEL_MAP.get(norm, norm)
    return k


def standardize_rank_counts(counts):
    """Return *counts* keyed only by ``"best"``/``"second"``/``"third"``.

    Other keys are dropped. A trial whose counts are not a dict yields ``{}``.
    """
    if not isinstance(counts, dict):
        return {}
    pairs = [(standardize_rank_key(k), v) for k, v in counts.items()]
    return {k: v for k, v in pairs if k in RANK_KEYS}


def infer_rank_from_counts(current, previous, is_first_trial):
    """Infer the rank chosen on a trial from the change in cumulative rank counts.

    Args:
        current: Standardised cumulative counts after this trial.
        previous: Standardised cumulative counts after the previous trial
            (``{}`` for the first trial).
        is_first_trial: Whether this is the session's first trial.

    Returns:
        The one rank whose count went up. On the first trial, if no single
        rank went up, the rank with the largest count (earliest on ties) as
        long as that count is positive. ``None`` if the rank is ambiguous.
    """
    deltas = {rk: (current.get(rk, 0) or 0) - (previous.get(rk, 0) or 0) for rk in RANK_KEYS}
    increased = [rk for rk, delta in deltas.items() if delta > 0]
    if len(increased) == 1:
        return increased[0]
    if is_first_trial:
        top = max(RANK_KEYS, key=lambda rk: current.get(rk, 0) or 0)
        if (current.get(top, 0) or 0) > 0:
            return top
    return None


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
        possible. Left untouched if either key is missing or empty, or if
        their lengths differ.
    """
    if "chosen_rank" not in session_data or "rank_counts" not in session_data:
        return session_data

    chosen = session_data.get("chosen_rank", [])
    rank_counts = session_data.get("rank_counts", [])
    # Known issue: a length mismatch between chosen_rank and rank_counts silently fills nothing.
    if not chosen or not rank_counts or len(chosen) != len(rank_counts):
        return session_data

    counts = [standardize_rank_counts(d) for d in rank_counts]
    previous_counts = [{}, *counts[:-1]]
    session_data["chosen_rank"] = [
        rank if rank is not None else infer_rank_from_counts(current, previous, i == 0)
        for i, (rank, current, previous) in enumerate(
            zip(chosen, counts, previous_counts, strict=True)
        )
    ]
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

    Missing keys in either dict are treated as ``0``.  Values must be
    integer-convertible; anything else raises, since a dropped increment would
    silently undercount the merged counters.

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
            out[k] = int(out.get(k, 0)) + int(v)
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
        ``"choice_towers"`` is absent, all values are ``None`` or the first is empty.
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

    if not towers_part:
        return None

    default_initiation_tower = "B2"
    init_tower = session_data.get("initiation_tower", None)
    if init_tower is None:
        init_tower = default_initiation_tower

    return (towers_part, init_tower)


def session_sort_key(session_id):
    """Sort key putting sessions in chronological order: ``ses-N`` number, then ``date-YYYYMMDD``.

    Keys without a session number or date sort last; the key string breaks ties.
    """
    m_ses = re.search(r"ses-(\d+)", session_id)
    m_date = re.search(r"date-(\d{8})", session_id)
    ses_num = int(m_ses.group(1)) if m_ses else float("inf")
    dt = datetime.strptime(m_date.group(1), "%Y%m%d") if m_date else datetime.max
    return (ses_num, dt, session_id)


def problem_signature(session_data):
    """Return the session's tower layout (choice towers, initiation tower), or ``None``.

    ``None`` means the session logged no choice towers.
    """
    return permutation_invariant_signature(choice_towers_signature(session_data))


def session_facts(subject_id, session_id, session_data):
    """The :class:`SessionFacts` for one extracted session.

    The layout comes from the trial-level ``choice_towers``; the settings evidence is the
    ``"settings_evidence"`` entry that ``import_data`` stores on every session.
    """
    evidence = session_data.get("settings_evidence")
    if evidence is None:
        raise ValueError(
            f"{subject_id} {session_id}: no 'settings_evidence'; "
            "sessions must come from import_data"
        )
    return SessionFacts(
        problem_signature(session_data),
        evidence["has_num_t_found"],
        evidence["starts_found_ports_false"],
        evidence["task_name"],
        evidence["has_port_finding_trials"],
    )


def signed_sessions(subject_id, subject_sessions):
    """Return ``[(session_id, SessionFacts)]`` in chronological order.

    Sessions without trial information are skipped with a warning.
    """
    signed = []
    for session_id in sorted(subject_sessions, key=session_sort_key):
        sess = subject_sessions[session_id]
        if not sess.get("trial_info"):
            print(
                f"[WARNING] Skipping session {session_id} for subject {subject_id} "
                "due to missing trial information."
            )
            continue
        signed.append((session_id, session_facts(subject_id, session_id, sess)))
    return signed


def group_sessions_by_problem(data, copy_sessions=True):
    """Group sessions into numbered problems per subject's tower layouts.

    Within each subject, sessions are sorted chronologically and numbered by
    :func:`src.behavior_import.problem_numbering.number_subject_sessions`, the same
    rule ``import_data`` uses for its ``"problem"`` field: a problem is the choice
    towers plus the initiation tower, a return to an earlier layout is a new problem,
    and a session that changes layout but looks like a settings error is left out
    (with a warning), neither a new problem nor pooled with another. Pretraining,
    port-finding and other tower-less sessions are left out too (see
    :mod:`src.behavior_import.problem_numbering`). Sessions without
    trial information are skipped before numbering, so the numbers can differ from
    ``import_data``'s when such sessions exist. Flagged sessions stay in *data* for
    single-session analyses.

    Args:
        data: Nested dict ``{subject: {session_key: session_dict}}`` after
            :func:`extract_trials` has been applied.
        copy_sessions: If ``True`` (default), each session dict is deep-copied
            into the output to avoid aliasing the input.

    Returns:
        Dict ``{problem_id: {subject: {session_key: session_dict}}}``.
    """
    problems = {}
    for subject_id, subject_sessions in data.items():
        signed = signed_sessions(subject_id, subject_sessions)
        assignments = number_subject_sessions(
            subject_id, [session_id for session_id, _ in signed], [facts for _, facts in signed]
        )
        for (session_id, _), assignment in zip(signed, assignments, strict=True):
            if assignment.problem is None:
                continue
            sess = subject_sessions[session_id]
            by_subject = problems.setdefault(assignment.problem, {})
            by_subject.setdefault(subject_id, {})[session_id] = (
                deepcopy(sess) if copy_sessions else sess
            )
    return problems


# --- Adding Variables to Analyze Value ---
def available_towers(towers_entry, magnitudes):
    """Return the tower identifiers on offer in one trial.

    *towers_entry* is that trial's ``choice_towers`` value (dict, list, tuple,
    set or a single string). Anything else, including ``None``, means the
    towers were not logged, so the keys of the trial's *magnitudes* dict are
    used instead.
    """
    if towers_entry is not None:
        if isinstance(towers_entry, dict):
            return list(towers_entry.keys())
        if isinstance(towers_entry, (list, tuple, set)):
            return list(towers_entry)
        if isinstance(towers_entry, str):
            return [towers_entry]
    return list(magnitudes.keys())


def top_magnitudes(towers, magnitudes, n):
    """Return the *n* largest magnitudes among *towers*, best first.

    Towers without a magnitude are ignored.
    """
    pairs = [(tw, magnitudes.get(tw)) for tw in towers]
    pairs = [(tw, m) for tw, m in pairs if m is not None]
    pairs.sort(key=lambda pair: (-pair[1], str(pair[0])))
    return [m for _, m in pairs[:n]]


def reward_magnitude_features(mags_seq, choice_seq, towers_seq):
    """Derive per-trial value features from the logged reward magnitudes.

    Args:
        mags_seq: Non-empty list with one ``{tower: magnitude}`` dict per trial
            (other entries count as "no magnitudes logged").
        choice_seq: Chosen tower per trial. Ignored (treated as all missing)
            unless it has one entry per trial.
        towers_seq: Per-trial available towers, or ``None``/shorter than
            *mags_seq* where the towers must be read from the magnitude dicts.

    Returns:
        ``(chosen_magnitude, magnitude_by_arm, magnitude_by_rank)``: a list, a
        ``{tower: list}`` dict and a ``{rank: list}`` dict. Entries are ``None``
        where nothing was logged.
    """
    n = len(mags_seq)
    mags = [m if isinstance(m, dict) else {} for m in mags_seq]
    # Known issue: a choice_seq whose length differs from mags_seq silently makes every
    # chosen magnitude None.
    if not choice_seq or len(choice_seq) != n:
        choice_seq = [None] * n
    # Known issue: a towers_seq shorter than mags_seq is silently padded with None, so the
    # trials past its end read their towers from the magnitude dicts.
    entries = towers_seq if towers_seq and isinstance(towers_seq, list) else []
    entries = (entries + [None] * n)[:n]
    available = [available_towers(entry, m) for entry, m in zip(entries, mags, strict=True)]

    # dict.fromkeys keeps first-seen order, so the output columns are stable
    all_towers = list(dict.fromkeys(tw for towers in available for tw in towers))
    by_arm = {tw: [m.get(tw) for m in mags] for tw in all_towers}
    chosen = [m[ch] if ch in m else None for ch, m in zip(choice_seq, mags, strict=True)]

    tops = [
        top_magnitudes(towers, m, len(RANK_KEYS)) for towers, m in zip(available, mags, strict=True)
    ]
    by_rank = {
        rk: [top[i] if i < len(top) else None for top in tops] for i, rk in enumerate(RANK_KEYS)
    }
    return chosen, by_arm, by_rank


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
        The same *session_data* dict with the three derived keys added, or
        unchanged if there are no reward magnitudes.
    """
    mags_seq = session_data.get("reward_magnitudes", [])
    if not mags_seq or not isinstance(mags_seq, list):
        return session_data

    chosen, by_arm, by_rank = reward_magnitude_features(
        mags_seq, session_data.get("choice", []), session_data.get("choice_towers", None)
    )
    session_data["chosen_magnitude"] = chosen
    session_data["reward_magnitude_by_arm"] = by_arm
    session_data["reward_magnitude_by_rank"] = by_rank
    # Downstream code reads the per-tower magnitudes under this older name.
    session_data["reward_magnitudes_by_tower"] = by_arm
    return session_data
