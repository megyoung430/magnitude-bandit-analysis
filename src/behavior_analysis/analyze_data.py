import json
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from src.behavior_preprocess import *

RANK_ORDER = ("best", "second_best", "third_best")

# ============ Helpers for pokes around state boundaries ============

_POKE_PATTERN = re.compile(r"\b([ABC][123])_in\b")

def _split_by_state(df: pd.DataFrame, state: str) -> List[pd.DataFrame]:
    """Split df into slices that start at a state row and end *before* the next state row."""
    state_idx = df.index[df.get("content") == state].to_list()
    if not state_idx:
        return []
    bounds = state_idx + [len(df)]
    slices = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        # start at state row, continue up to but not including the next state row
        slices.append(df.loc[start:end - 1])
    return slices

def _first_poke_in_slice(slice_df: pd.DataFrame, exclude: Optional[str] = None) -> Optional[str]:
    """Return first poke name in slice, excluding one tower if given."""
    hits = slice_df.get("content", pd.Series(dtype=object)).str.extract(_POKE_PATTERN, expand=False)
    for p in hits.dropna():
        if exclude is None or p != exclude:
            return p
    return None

def _all_pokes_in_slice(slice_df: pd.DataFrame, exclude: Optional[str] = None) -> List[str]:
    hits = slice_df.get("content", pd.Series(dtype=object)).str.extractall(_POKE_PATTERN)[0].dropna()
    if exclude is None:
        return hits.tolist()
    return [p for p in hits.tolist() if p != exclude]

def get_first_pokes_after_initiation(
    df: pd.DataFrame,
    state: str = "wait_for_choice_poke",
    initiation_tower: str = "B2",
) -> Tuple[Dict[str, int], Dict[int, Optional[str]]]:
    """First poke after initiation (wait_for_choice_poke), ignoring the initiation port."""
    counts = {k: 0 for k in ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]}
    trials: Dict[int, Optional[str]] = {}

    for tnum, sl in enumerate(_split_by_state(df, state), start=1):
        poke = _first_poke_in_slice(sl, exclude=initiation_tower)
        if poke:
            counts[poke] += 1
        trials[tnum] = poke

    return counts, trials

def get_first_pokes_after_choice(
    df: pd.DataFrame,
    state: str = "wait_for_initiation_poke",
    exclude_choice: bool = False,
) -> Tuple[Dict[str, int], Dict[int, Optional[str]]]:
    """First poke after choice (wait_for_initiation_poke). Optionally exclude the chosen tower."""
    counts = {k: 0 for k in ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]}
    trials: Dict[int, Optional[str]] = {}

    current_choice: Optional[str] = None
    for tnum, sl in enumerate(_split_by_state(df, state), start=1):
        exclude = current_choice if exclude_choice else None
        poke = _first_poke_in_slice(sl, exclude=exclude)
        if poke:
            counts[poke] += 1
        trials[tnum] = poke

        # Update choice from any print lines with chosen_tower in this slice
        choice_strs = sl.loc[sl.get("subtype") == "print", "content"].dropna().astype(str)
        for txt in choice_strs:
            try:
                j = json.loads(txt)
                if isinstance(j, dict) and "chosen_tower" in j:
                    current_choice = j["chosen_tower"]
            except (json.JSONDecodeError, TypeError):
                continue

    return counts, trials

def get_all_pokes_after_state(
    df: pd.DataFrame,
    state: str,
    exclude: Optional[str] = None,
) -> Dict[int, List[str]]:
    """List all pokes following each occurrence of a state until the next occurrence."""
    out: Dict[int, List[str]] = {}
    for tnum, sl in enumerate(_split_by_state(df, state), start=1):
        out[tnum] = _all_pokes_in_slice(sl, exclude=exclude)
    return out

def get_transition_matrix(df, col="chosen_tower", order=("A2","B1","C2")):
    """Compute the transition matrix for a given column in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing trial data.
        col (str, optional): The column to analyze for transitions. Defaults to "chosen_tower".
        order (tuple, optional): The order of arms to consider. Defaults to ("A2","B1","C2").

    Returns:
        pd.DataFrame: A DataFrame representing the transition probabilities between arms.
    """
    towers_df = (df[col]
         .astype(str)
         .str.upper()
         .str.strip())

    chosen_towers = towers_df[towers_df.isin(order)].reset_index(drop=True)
    if len(chosen_towers) < 2:
        return pd.DataFrame(0.0, index=order, columns=order)

    frm = pd.Categorical(chosen_towers.iloc[:-1], categories=order)
    to  = pd.Categorical(chosen_towers.iloc[1:],  categories=order)

    counts = pd.crosstab(frm, to).reindex(index=order, columns=order, fill_value=0)
    total_transitions = len(chosen_towers) - 1
    assert counts.values.sum() == total_transitions, (f"Counts ({counts.values.sum()}) != expected transitions ({total_transitions})")
    
    probs = counts.div(counts.sum(axis=1).replace(0, 1), axis=0)
    return probs

# ============ Block-change analysis ============

def block_switch_first_choice_counts(
    session_dfs: List[pd.DataFrame],
    block_col: str = "num_blocks",
    choice_col: str = "chosen_tower",
    best_cols: Tuple[str, str, str] = ("best", "second_best", "third_best"),
) -> Dict[str, int]:
    """
    Count first choice after leaving the previous block's best arm.

    For each block transition (block id change within a session), find the first trial 
    whose choice is NOT the previous block's best. Only count choices to:
      - "prev_second" if it equals previous block's second_best
      - "prev_third"  if it equals previous block's third_best
    
    If the animal stays with prev_best for the rest of the session, continues searching
    into the next session (since reward contingencies persist across sessions).
    
    Choices to arms other than prev_second or prev_third are ignored.

    Returns dict with counts for prev_second and prev_third only.
    
    Args:
        session_dfs: List of DataFrames, one per session (in chronological order)
        block_col: Column name for block ID
        choice_col: Column name for chosen tower
        best_cols: Tuple of column names (best, second_best, third_best)
    """
    counts = {"prev_second": 0, "prev_third": 0}
    
    # Process each session
    for sess_idx, df in enumerate(session_dfs):
        need = [block_col, choice_col, *best_cols]
        miss = [c for c in need if c not in df.columns]
        if miss:
            continue  # Skip sessions missing required columns

        block = pd.to_numeric(df[block_col], errors="coerce")
        choice = df[choice_col].astype(str).str.strip().str.upper()
        best = df[best_cols[0]].astype(str).str.strip().str.upper()
        second = df[best_cols[1]].astype(str).str.strip().str.upper()
        third = df[best_cols[2]].astype(str).str.strip().str.upper()

        # iterate over block changes within this session
        for i in range(1, len(df)):
            if pd.isna(block.iat[i]) or pd.isna(block.iat[i-1]):
                continue
            if block.iat[i] == block.iat[i-1]:
                continue

            # We just entered a new block at i. Identify previous block ranks.
            prev_best = best.iat[i-1]
            prev_second = second.iat[i-1]
            prev_third = third.iat[i-1]

            # Find first trial that is not the previous best
            # Search within current session first
            j = i
            while j < len(df) and block.iat[j] == block.iat[i] and choice.iat[j] == prev_best:
                j += 1
            
            # If we reached end of session still on prev_best, search next session
            if j >= len(df) and sess_idx + 1 < len(session_dfs):
                next_df = session_dfs[sess_idx + 1]
                next_need = [choice_col]
                if all(c in next_df.columns for c in next_need):
                    next_choice = next_df[choice_col].astype(str).str.strip().str.upper()
                    # Search through next session
                    for k in range(len(next_df)):
                        if next_choice.iat[k] != prev_best:
                            ch = next_choice.iat[k]
                            # Count if it goes to prev_second or prev_third
                            if ch == prev_second:
                                counts["prev_second"] += 1
                            elif ch == prev_third:
                                counts["prev_third"] += 1
                            break
                continue
            
            # Check if we found a choice within the same block in current session
            if j >= len(df) or block.iat[j] != block.iat[i]:
                continue

            ch = choice.iat[j]
            # Only count if choice goes to prev_second or prev_third
            if ch == prev_second:
                counts["prev_second"] += 1
            elif ch == prev_third:
                counts["prev_third"] += 1
            # Ignore other choices

    return counts

def block_switch_n_trial_choices_before(
    session_dfs: List[pd.DataFrame],
    n_trials_before: int = 5,
    block_col: str = "num_blocks",
    choice_col: str = "chosen_tower",
    best_cols: Tuple[str, str, str] = ("best", "second_best", "third_best"),
) -> pd.DataFrame:
    """
    Track choices for n trials BEFORE each block switch, handling cross-session boundaries.
    
    Returns a DataFrame with columns:
        - block_switch_idx: index of the block switch
        - trial_offset: -n_trials_before to -1 (negative trial numbers relative to switch)
        - prev_best: the best arm from the previous block
        - new_best: the best arm in the new block
        - third_arm: the remaining arm
        - choice: the actual choice made
        - choice_type: "prev_best", "new_best", or "third_arm"
    
    Args:
        session_dfs: List of DataFrames, one per session (in chronological order)
        n_trials_before: Number of trials to track before each block switch
        block_col: Column name for block ID
        choice_col: Column name for chosen tower
        best_cols: Tuple of column names (best, second_best, third_best)
    """
    # Concatenate all sessions with session tracking
    all_rows = []
    for sess_idx, df in enumerate(session_dfs):
        df_copy = df.copy()
        df_copy["_session_idx"] = sess_idx
        df_copy["_global_idx"] = range(len(all_rows), len(all_rows) + len(df_copy))
        all_rows.append(df_copy)
    
    combined = pd.concat(all_rows, ignore_index=True)
    
    # Validate columns
    need = [block_col, choice_col, *best_cols]
    miss = [c for c in need if c not in combined.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")
    
    # Normalize
    combined[block_col] = pd.to_numeric(combined[block_col], errors="coerce")
    combined[choice_col] = combined[choice_col].astype(str).str.strip().str.upper()
    for col in best_cols:
        combined[col] = combined[col].astype(str).str.strip().str.upper()
    
    # Find block switches (within sessions only - don't count session boundary as switch)
    results = []
    for i in range(1, len(combined)):
        # Skip if session changed
        if combined["_session_idx"].iat[i] != combined["_session_idx"].iat[i-1]:
            continue
        
        # Skip if block didn't change
        block_prev = combined[block_col].iat[i-1]
        block_curr = combined[block_col].iat[i]
        if pd.isna(block_prev) or pd.isna(block_curr) or block_prev == block_curr:
            continue
        
        # Block switch detected at position i
        prev_best = combined[best_cols[0]].iat[i-1]
        new_best = combined[best_cols[0]].iat[i]
        
        # Determine the third arm (the one that's neither prev nor new best)
        prev_second = combined[best_cols[1]].iat[i-1]
        prev_third = combined[best_cols[2]].iat[i-1]
        new_second = combined[best_cols[1]].iat[i]
        new_third = combined[best_cols[2]].iat[i]
        
        # Third arm is the one not in {prev_best, new_best}
        all_arms = {prev_best, prev_second, prev_third, new_best, new_second, new_third}
        third_arm = [a for a in all_arms if a not in {prev_best, new_best}]
        third_arm = third_arm[0] if len(third_arm) >= 1 else None
        
        # Collect n_trials_before (going backwards)
        for offset in range(1, n_trials_before + 1):
            trial_idx = i - offset
            if trial_idx < 0:
                break  # Ran out of data
            
            # Skip if session changed
            if combined["_session_idx"].iat[trial_idx] != combined["_session_idx"].iat[i]:
                break
            
            choice = combined[choice_col].iat[trial_idx]
            
            # Categorize choice (using previous block's arms)
            if choice == prev_best:
                choice_type = "prev_best"
            elif choice == prev_second:
                choice_type = "new_best"  # In the context of "before", this is second best
            elif choice == prev_third:
                choice_type = "third_arm"
            else:
                choice_type = "other"
            
            results.append({
                "block_switch_idx": i,
                "trial_offset": -offset,  # Negative offset for before
                "prev_best": prev_best,
                "new_best": new_best,
                "third_arm": third_arm,
                "choice": choice,
                "choice_type": choice_type,
            })
    
    return pd.DataFrame(results)

def block_switch_n_trial_choices(
    session_dfs: List[pd.DataFrame],
    n_trials: int = 10,
    block_col: str = "num_blocks",
    choice_col: str = "chosen_tower",
    best_cols: Tuple[str, str, str] = ("best", "second_best", "third_best"),
) -> pd.DataFrame:
    """
    Track choices for n trials after each block switch, handling cross-session boundaries.
    
    Returns a DataFrame with columns:
        - block_switch_idx: index of the block switch
        - trial_offset: 0 to n_trials-1 (trial number relative to switch)
        - prev_best: the best arm from the previous block
        - new_best: the best arm in the new block
        - third_arm: the remaining arm
        - choice: the actual choice made
        - choice_type: "prev_best", "new_best", or "third_arm"
    
    Args:
        session_dfs: List of DataFrames, one per session (in chronological order)
        n_trials: Number of trials to track after each block switch
        block_col: Column name for block ID
        choice_col: Column name for chosen tower
        best_cols: Tuple of column names (best, second_best, third_best)
    """
    # Concatenate all sessions with session tracking
    all_rows = []
    for sess_idx, df in enumerate(session_dfs):
        df_copy = df.copy()
        df_copy["_session_idx"] = sess_idx
        df_copy["_global_idx"] = range(len(all_rows), len(all_rows) + len(df_copy))
        all_rows.append(df_copy)
    
    combined = pd.concat(all_rows, ignore_index=True)
    
    # Validate columns
    need = [block_col, choice_col, *best_cols]
    miss = [c for c in need if c not in combined.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")
    
    # Normalize
    combined[block_col] = pd.to_numeric(combined[block_col], errors="coerce")
    combined[choice_col] = combined[choice_col].astype(str).str.strip().str.upper()
    for col in best_cols:
        combined[col] = combined[col].astype(str).str.strip().str.upper()
    
    # Find block switches (within sessions only - don't count session boundary as switch)
    results = []
    for i in range(1, len(combined)):
        # Skip if session changed
        if combined["_session_idx"].iat[i] != combined["_session_idx"].iat[i-1]:
            continue
        
        # Skip if block didn't change
        block_prev = combined[block_col].iat[i-1]
        block_curr = combined[block_col].iat[i]
        if pd.isna(block_prev) or pd.isna(block_curr) or block_prev == block_curr:
            continue
        
        # Block switch detected at position i
        prev_best = combined[best_cols[0]].iat[i-1]
        new_best = combined[best_cols[0]].iat[i]
        
        # Determine the third arm (the one that's neither prev nor new best)
        prev_second = combined[best_cols[1]].iat[i-1]
        prev_third = combined[best_cols[2]].iat[i-1]
        new_second = combined[best_cols[1]].iat[i]
        new_third = combined[best_cols[2]].iat[i]
        
        # Third arm is the one not in {prev_best, new_best}
        all_arms = {prev_best, prev_second, prev_third, new_best, new_second, new_third}
        third_arm = [a for a in all_arms if a not in {prev_best, new_best}]
        third_arm = third_arm[0] if len(third_arm) >= 1 else None
        
        # Collect next n_trials (may span sessions)
        for offset in range(n_trials):
            trial_idx = i + offset
            if trial_idx >= len(combined):
                break  # Ran out of data
            
            choice = combined[choice_col].iat[trial_idx]
            
            # Categorize choice
            if choice == prev_best:
                choice_type = "prev_best"
            elif choice == new_best:
                choice_type = "new_best"
            elif choice == third_arm:
                choice_type = "third_arm"
            else:
                choice_type = "other"
            
            results.append({
                "block_switch_idx": i,
                "trial_offset": offset,
                "prev_best": prev_best,
                "new_best": new_best,
                "third_arm": third_arm,
                "choice": choice,
                "choice_type": choice_type,
            })
    
    return pd.DataFrame(results)

def block_switch_choice_probabilities(
    session_dfs: List[pd.DataFrame],
    n_trials: int = 10,
    n_trials_before: int = 0,
    block_col: str = "num_blocks",
    choice_col: str = "chosen_tower",
    best_cols: Tuple[str, str, str] = ("best", "second_best", "third_best"),
) -> pd.DataFrame:
    """
    Compute choice probabilities around block switches (before and/or after).
    
    Returns DataFrame with columns:
        - trial_offset: negative values for before (-n_trials_before to -1), 0+ for after
        - prev_best: proportion choosing previous block's best
        - new_best: proportion choosing new block's best
        - third_arm: proportion choosing the third arm
        - n_switches: number of block switches contributing to this offset
    
    Args:
        session_dfs: List of DataFrames (sessions in chronological order)
        n_trials: Number of trials after switch to analyze
        n_trials_before: Number of trials before switch to analyze (default 0)
        block_col, choice_col, best_cols: Column names
    """
    # Get data after block switches
    df_after = block_switch_n_trial_choices(
        session_dfs, n_trials, block_col, choice_col, best_cols
    )
    
    # Get data before block switches
    df_before = pd.DataFrame()
    if n_trials_before > 0:
        df_before = block_switch_n_trial_choices_before(
            session_dfs, n_trials_before, block_col, choice_col, best_cols
        )
    
    # Combine
    if n_trials_before == 0:
        df = df_after
    elif df_after.empty and df_before.empty:
        return pd.DataFrame({
            "trial_offset": list(range(-n_trials_before, 0)) + list(range(n_trials)),
            "prev_best": [np.nan] * (n_trials_before + n_trials),
            "new_best": [np.nan] * (n_trials_before + n_trials),
            "third_arm": [np.nan] * (n_trials_before + n_trials),
            "n_switches": [0] * (n_trials_before + n_trials),
        })
    else:
        df = pd.concat([df_before, df_after], ignore_index=True)
    
    if df.empty:
        return pd.DataFrame({
            "trial_offset": list(range(-n_trials_before, 0)) + list(range(n_trials)),
            "prev_best": [np.nan] * (n_trials_before + n_trials),
            "new_best": [np.nan] * (n_trials_before + n_trials),
            "third_arm": [np.nan] * (n_trials_before + n_trials),
            "n_switches": [0] * (n_trials_before + n_trials),
        })
    
    # Count choices by offset
    summary = []
    offsets = list(range(-n_trials_before, 0)) + list(range(n_trials))
    for offset in offsets:
        subset = df[df["trial_offset"] == offset]
        if len(subset) == 0:
            summary.append({
                "trial_offset": offset,
                "prev_best": np.nan,
                "new_best": np.nan,
                "third_arm": np.nan,
                "n_switches": 0,
            })
            continue
        
        counts = subset["choice_type"].value_counts()
        total = len(subset)
        summary.append({
            "trial_offset": offset,
            "prev_best": counts.get("prev_best", 0) / total,
            "new_best": counts.get("new_best", 0) / total,
            "third_arm": counts.get("third_arm", 0) / total,
            "n_switches": len(subset["block_switch_idx"].unique()),
        })
    
    return pd.DataFrame(summary)

def count_reversals_per_session(
    session_dfs: List[pd.DataFrame],
    block_col: str = "num_blocks",
) -> List[int]:
    """
    Count the number of reversals (block switches) in each session.
    
    A reversal is when block ID changes within a session.
    Returns number of reversals = (number of unique blocks - 1) per session.
    
    Args:
        session_dfs: List of session DataFrames
        block_col: Column containing block IDs
    
    Returns:
        List of integers, one per session
    """
    reversals = []
    for df in session_dfs:
        if block_col not in df.columns:
            reversals.append(0)
            continue
        
        blocks = pd.to_numeric(df[block_col], errors="coerce").dropna()
        if len(blocks) == 0:
            reversals.append(0)
            continue
        
        # Count unique blocks
        n_blocks = blocks.nunique()
        # Number of reversals = n_blocks - 1
        reversals.append(max(0, n_blocks - 1))
    
    return reversals

def count_total_reversals(
    session_dfs: List[pd.DataFrame],
    block_col: str = "num_blocks",
) -> int:
    """
    Count total number of reversals across all sessions.
    
    Args:
        session_dfs: List of session DataFrames
        block_col: Column containing block IDs
    
    Returns:
        Total number of reversals
    """
    return sum(count_reversals_per_session(session_dfs, block_col))

def block_duration_distribution(
    session_dfs: List[pd.DataFrame],
    block_col: str = "num_blocks",
) -> List[int]:
    """
    Calculate the distribution of block durations (number of trials per block).
    
    Handles blocks that span multiple sessions. If a session ends during a block
    and the next session continues the same block, those trials are combined.
    
    Args:
        session_dfs: List of session DataFrames (in chronological order)
        block_col: Column containing block IDs
    
    Returns:
        List of block durations (number of trials in each block)
    
    Example:
        If session 1 has 50 trials ending in block 5, and session 2 starts
        with block 5 for 10 trials then switches to block 6, the duration
        for block 5 includes trials from both sessions.
    """
    if not session_dfs:
        return []
    
    # Concatenate all sessions with session tracking
    all_rows = []
    for sess_idx, df in enumerate(session_dfs):
        if block_col not in df.columns:
            continue
        df_copy = df[[block_col]].copy()
        df_copy["_session_idx"] = sess_idx
        df_copy[block_col] = pd.to_numeric(df_copy[block_col], errors="coerce")
        all_rows.append(df_copy)
    
    if not all_rows:
        return []
    
    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined.dropna(subset=[block_col])
    
    if len(combined) == 0:
        return []
    
    # Track block durations
    durations = []
    current_block = None
    current_duration = 0
    
    for i in range(len(combined)):
        block_id = combined[block_col].iat[i]
        
        if current_block is None:
            # Start first block
            current_block = block_id
            current_duration = 1
        elif block_id == current_block:
            # Continue same block
            current_duration += 1
        else:
            # Block changed - record duration and start new block
            durations.append(current_duration)
            current_block = block_id
            current_duration = 1
    
    # Record the final block
    if current_duration > 0:
        durations.append(current_duration)
    
    return durations

# ============ Choice counts by rank/arm ============

def count_choice_by_rank(
    df: pd.DataFrame,
    first_poke_across_trials: Optional[Dict[int, str]] = None,
    control: Optional[str] = "B3",
    *,
    rank_source: str = "actual",   # "actual" uses best/second/third; "belief" uses best_per/second_per/third_per
    per_suffix: str = "_per",
    weighted: bool = False,          # if True, use rank shares to apportion ties
) -> Dict[str, float]:
    """Count choices by rank with optional control precedence and belief/actual toggle.

    If weighted=True, ties are apportioned using rank-share columns (best_share_*/best_per_share_*).
    """

    if rank_source.lower() in ("belief", "per"):
        rank_cols = ["best_per", "second_best_per", "third_best_per"]
    elif rank_source.lower() == "actual":
        rank_cols = ["best", "second_best", "third_best"]
    else:
        raise ValueError("rank_source must be 'actual' or 'belief'/'per'.")

    # Ensure required rank/share columns exist when needed (weighted or missing ranks).
    need = ["trial", "chosen_tower", *rank_cols]
    if weighted or any(c not in df.columns for c in need):
        df = _ensure_rank_shares(df.copy(), rank_source=rank_source, per_suffix=per_suffix)
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"df missing required columns: {miss}")

    d = df.copy()
    d["trial"] = pd.to_numeric(d["trial"], errors="coerce").astype("Int64")

    def norm(s: pd.Series) -> pd.Series:
        return s.astype("string").str.strip().str.upper()

    d["chosen_tower"] = norm(d["chosen_tower"])
    for c in rank_cols:
        d[c] = norm(d[c])

    use_control = bool(control) and first_poke_across_trials
    if use_control:
        fp = pd.Series(first_poke_across_trials, name="first_poke")
        fp.index = pd.to_numeric(pd.Index(fp.index), errors="coerce").astype("Int64")
        fp = norm(fp)
        d["first_poke"] = d["trial"].map(fp)
        control = str(control).strip().upper()
    else:
        d["first_poke"] = pd.Series(pd.NA, index=d.index, dtype="string")
        control = None

    order = ["best", "second_best", "third_best"]
    if use_control:
        order.append(control)
    counts: Dict[str, float] = {k: 0.0 for k in order}

    # Unweighted path: simple bucket matching with optional control precedence.
    if not weighted:
        d["bucket"] = pd.Series(pd.NA, index=d.index, dtype="string")
        if use_control:
            d.loc[d["first_poke"].eq(control), "bucket"] = control

        d.loc[d["bucket"].isna() & d["chosen_tower"].eq(d[rank_cols[0]]), "bucket"] = "best"
        d.loc[d["bucket"].isna() & d["chosen_tower"].eq(d[rank_cols[1]]), "bucket"] = "second_best"
        d.loc[d["bucket"].isna() & d["chosen_tower"].eq(d[rank_cols[2]]), "bucket"] = "third_best"

        counts.update(
            d["bucket"].value_counts(dropna=True).reindex(order, fill_value=0).to_dict()
        )
        return counts

    # Weighted path: use tie-aware rank weights from choice_rank_weights; control still takes precedence.
    weights = choice_rank_weights(
        d,
        rank_source=rank_source,
        per_suffix=per_suffix,
        choice_col="chosen_tower",
        require_shares=True,
    )

    for w, row in zip(weights, d.itertuples(index=False)):
        if use_control and row.first_poke == control:
            counts[control] += 1.0
            continue
        counts["best"] += float(w[0])
        counts["second_best"] += float(w[1])
        counts["third_best"] += float(w[2])

    return counts

def count_choice_by_arm(df: pd.DataFrame, first_poke_across_trials: dict,
    arms=None,                 # e.g. ["A2","B1","C2"]; if None, auto-detect from columns/choices
    control: str = "B3") -> dict:
    """
    Count choices by actual tower (A2/B1/C2/...), with 'control' taking precedence
    when the trial's first_poke equals the control (e.g., 'B3').

    Expects df to have at least ['trial','chosen_tower'] and usually arm columns (A2/B1/C2...).

    Returns: dict like {'A2': n, 'B1': n, 'C2': n, 'B3': k} (plus any other arms).
    """
    def _dedupe_preserve_order(seq):
        seen = set(); out = []
        for a in seq:
            A = str(a).strip().upper()
            if A not in seen:
                seen.add(A); out.append(A)
        return out
    
    need = ["trial", "chosen_tower"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"df missing required columns: {miss}")

    # --- 1) Determine which arms to report
    if arms is None:
        # prefer arm-like columns (A1..C3) if present, else from chosen_tower values
        col_arms = [c for c in df.columns if re.fullmatch(r"[A-Za-z]\d+", str(c) or "")]
        if col_arms:
            arms = col_arms
        else:
            vals = df["chosen_tower"].dropna().astype(str).str.strip().str.upper().unique().tolist()
            arms = sorted([v for v in vals if re.fullmatch(r"[A-Z]\d+", v)],
                          key=lambda x: (x[0], int(x[1:])))
    arms = _dedupe_preserve_order(arms)
    control = str(control).strip().upper()
    arms_plus = arms + ([control] if control and control not in arms else [])

    # --- 2) Align trials and build bucket per row
    df = df.copy()
    df["trial"] = pd.to_numeric(df["trial"], errors="coerce").astype("Int64")

    fp = pd.Series(first_poke_across_trials, name="first_poke")
    fp.index = pd.to_numeric(pd.Index(fp.index), errors="coerce").astype("Int64")

    merged = df[["trial", "chosen_tower"]].merge(
        fp.rename_axis("trial").reset_index(), on="trial", how="left"
    )

    def norm(s): return s.astype("string").str.strip().str.upper()
    merged["chosen_tower"] = norm(merged["chosen_tower"])
    merged["first_poke"]   = norm(merged["first_poke"])

    # control precedence
    merged["bucket"] = merged["chosen_tower"]
    merged.loc[merged["first_poke"].eq(control), "bucket"] = control

    # keep only known arms/control
    merged = merged[merged["bucket"].isin(set(arms_plus))]

    # --- 3) Count (or normalize)
    counts = (merged["bucket"]
              .value_counts(dropna=False)
              .reindex(arms_plus, fill_value=0))
    return counts.to_dict()

def sum_counts(dicts, keys):
    out = {k: 0 for k in keys}
    for d in dicts:
        for k in keys:
            out[k] += int(d.get(k, 0))
    return out

def count_choice_by_rank_across_sessions(
    session_trial_dfs: List[pd.DataFrame],
    session_first_pokes: Optional[List[Optional[Dict[int, str]]]] = None,
    control: Optional[str] = "B3",
    *,
    rank_source: str = "actual",     # "actual" -> best/second/third; "belief"/"per" -> best_per/second_per/third_per
    belief_suffix: str = "_per",
    weighted: bool = False,
) -> Dict[str, int]:
    """Aggregate choice-by-rank counts across sessions with optional control and belief ranks."""

    if session_first_pokes is None:
        session_first_pokes = [None] * len(session_trial_dfs)
    if len(session_trial_dfs) != len(session_first_pokes):
        raise ValueError("session_trial_dfs and session_first_pokes must have the same length.")

    keys = ["best", "second_best", "third_best"]
    if control:
        keys.append(str(control).strip().upper())

    all_counts = []
    for df_sess, fp_sess in zip(session_trial_dfs, session_first_pokes):
        if rank_source.lower() in ("belief", "per"):
            need = {"best_per", "second_best_per", "third_best_per"}
            have_shares = any(c.startswith(f"best{belief_suffix}_share_") for c in df_sess.columns)
            if not need.issubset(df_sess.columns) or (weighted and not have_shares):
                df_sess = add_rank_and_equality_cols(df_sess, add_per=True, per_suffix=belief_suffix)
        elif rank_source.lower() == "actual":
            need = {"best", "second_best", "third_best"}
            have_shares = any(c.startswith("best_share_") for c in df_sess.columns)
            if not need.issubset(df_sess.columns) or (weighted and not have_shares):
                df_sess = add_rank_and_equality_cols(df_sess)
        else:
            raise ValueError("rank_source must be 'actual' or 'belief'/'per'.")

        c = count_choice_by_rank(
            df_sess,
            first_poke_across_trials=fp_sess,
            control=control,
            rank_source=rank_source,
            per_suffix=belief_suffix,
            weighted=weighted,
        )
        all_counts.append(c)

    return sum_counts(all_counts, keys)

def count_choice_by_arm_across_sessions(session_trial_dfs: List[pd.DataFrame], session_first_pokes: List[Dict[int, str]],
    arms: Optional[List[str]] = None,   # e.g., ["A2","B1","C2"]; if None, auto-detect
    control: str = "B3") -> Dict[str, int]:
    """
    Sum choice-by-arm counts across sessions with control precedence.
    Uses the shared `sum_counts` to combine per-session dicts.
    """
    if len(session_trial_dfs) != len(session_first_pokes):
        raise ValueError("session_trial_dfs and session_first_pokes must have the same length.")

    control = str(control).strip().upper()

    # --- Auto-detect arms (union across sessions), excluding control
    if arms is None:
        arms_set = set()
        arm_pat = re.compile(r"^[A-Za-z]\d+$")
        for df in session_trial_dfs:
            col_arms = [c for c in df.columns if arm_pat.fullmatch(str(c) or "")]
            if col_arms:
                arms_set.update(col_arms)
            elif "chosen_tower" in df.columns:
                vals = (df["chosen_tower"].dropna().astype(str).str.strip().str.upper())
                arms_set.update(v for v in vals if arm_pat.fullmatch(v))
        arms = sorted([a for a in arms_set if a != control], key=lambda x: (x[0], int(x[1:])))

    # Keys to sum in a fixed order (arms + control at the end)
    keys = list(arms)
    if control not in keys:
        keys.append(control)

    # --- Per-session counts
    all_counts = []
    for df_sess, fp_sess in zip(session_trial_dfs, session_first_pokes):
        c = count_choice_by_arm(df_sess, fp_sess, arms=arms, control=control)
        all_counts.append(c)

    # --- Sum using the shared helper
    return sum_counts(all_counts, keys)

# ============ Choice counts by value rather than rank ============

def value_offer_choice_counts(
    df: pd.DataFrame,
    arms: Optional[List[str]] = None,
    chosen_col: str = "chosen_tower",
    value_source: str = "belief",       # "actual" uses A2/B1/C2 ... ; "belief" uses A2_per/B1_per/C2_per
    belief_suffix: str = "_per",
) -> pd.DataFrame:
    """
    Per-value counts:
        value | n_offered | n_chosen | p_chosen

    - Toggle which choice column to use via `chosen_col`.
    - Toggle whether to read arm values from TRUE columns or BELIEF columns via `value_source`.
    """
    arm_pat = re.compile(r"^[A-Za-z]\d+$")

    # Determine arms
    if arms is None:
        arms = [c for c in df.columns if arm_pat.fullmatch(str(c) or "")]
        if not arms:
            raise ValueError("Could not auto-detect arm columns. Pass `arms=[...]`.")
    arms = list(arms)

    # Select value matrix: actual vs belief
    if value_source.lower() in ("belief", "per"):
        value_cols = [f"{a}{belief_suffix}" for a in arms]
        missing = [c for c in value_cols if c not in df.columns]
        if missing:
            raise ValueError(f"value_source='belief' but missing columns: {missing}")
        vals = df[value_cols].apply(pd.to_numeric, errors="coerce")
    elif value_source.lower() == "actual":
        vals = df[arms].apply(pd.to_numeric, errors="coerce")
    else:
        raise ValueError("value_source must be 'actual' or 'belief'/'per'")

    # Canonicalize numeric values (e.g., 2.0 -> 2)
    def canon(x):
        if pd.isna(x): return np.nan
        xf = float(x)
        r = round(xf)
        return int(r) if abs(xf - r) < 1e-9 else xf

    vals = vals.applymap(canon)

    # n_offered: values per trial across arms
    offered_counts = vals.stack(dropna=True).value_counts().sort_index()

    # n_chosen: value on the chosen arm each trial
    chosen_arm = df[chosen_col].astype(str).str.strip().str.upper()
    arm_to_idx = {a.upper(): j for j, a in enumerate(arms)}  # index in *arms* order
    idx = chosen_arm.map(arm_to_idx).fillna(-1).astype(int).to_numpy()

    arr = vals.to_numpy()  # shape (N, K)
    N, _ = arr.shape
    chosen_vals = np.full(N, np.nan, dtype=float)
    mask = idx >= 0
    if mask.any():
        chosen_vals[mask] = arr[np.arange(N)[mask], idx[mask]]
    chosen_vals = pd.Series(chosen_vals).dropna().map(canon)

    chosen_counts = chosen_vals.value_counts().sort_index()

    # Assemble
    all_values = sorted(set(offered_counts.index).union(set(chosen_counts.index)))
    out = pd.DataFrame({
        "value": all_values,
        "n_offered": [int(offered_counts.get(v, 0)) for v in all_values],
        "n_chosen":  [int(chosen_counts.get(v, 0)) for v in all_values],
    })
    out["p_chosen"] = out.apply(
        lambda r: (r["n_chosen"] / r["n_offered"]) if r["n_offered"] > 0 else 0.0, axis=1
    )
    return out

def value_offer_choice_counts_across_sessions(
    session_trial_dfs: List[pd.DataFrame],
    arms: Optional[List[str]] = None,
    chosen_col: str = "chosen_tower",
    value_source: str = "belief",   # "actual" or "belief"/"per"
    belief_suffix: str = "_per",
) -> pd.DataFrame:
    """
    Sum n_offered and n_chosen across sessions, then recompute p_chosen.
    Toggle choice column via `chosen_col`, and value source via `value_source`.
    """
    pieces = [
        value_offer_choice_counts(
            df,
            arms=arms,
            chosen_col=chosen_col,
            value_source=value_source,
            belief_suffix=belief_suffix,
        )
        for df in session_trial_dfs
    ]

    out = (
        pd.concat(pieces, ignore_index=True)
          .groupby("value", as_index=False)[["n_offered", "n_chosen"]]
          .sum()
    )
    out["p_chosen"] = out["n_chosen"] / out["n_offered"].replace(0, np.nan)
    out["p_chosen"] = out["p_chosen"].fillna(0.0)
    return out.sort_values("value").reset_index(drop=True)

# ============ Rank-based transition matrices ============

def _ensure_rank_shares(df: pd.DataFrame, *, rank_source: str, per_suffix: str) -> pd.DataFrame:
    """Ensure rank/share columns exist by invoking add_rank_and_equality_cols when missing."""
    if rank_source.lower() in {"belief", "per", "perceived"}:
        have = any(c.startswith(f"best{per_suffix}_share_") for c in df.columns)
        return add_rank_and_equality_cols(df, add_per=True, per_suffix=per_suffix) if not have else df
    have = any(c.startswith("best_share_") for c in df.columns)
    return add_rank_and_equality_cols(df) if not have else df

def choice_rank_weights(
    df: pd.DataFrame,
    *,
    rank_source: str = "actual",
    per_suffix: str = "_per",
    choice_col: str = "chosen_tower",
    require_shares: bool = True,
) -> np.ndarray:
    """Return (N,3) weights of the chosen arm across ranks [best, second, third], tie-aware."""

    tag = per_suffix if rank_source.lower() in {"belief", "per", "perceived"} else ""

    if choice_col not in df.columns:
        raise ValueError(f"df missing required column '{choice_col}'")

    d = _ensure_rank_shares(df.copy(), rank_source=rank_source, per_suffix=per_suffix)
    d[choice_col] = d[choice_col].astype("string").str.strip().str.upper()

    share_prefix = f"best{tag}_share_"
    has_shares = any(c.startswith(share_prefix) for c in d.columns)
    if not has_shares and require_shares:
        raise ValueError(
            f"Share columns not found (e.g., {share_prefix}<arm>). "
            "Run add_rank_and_equality_cols(...) first or set require_shares=False."
        )

    W = np.zeros((len(d), 3), dtype=float)
    for i, r in d.iterrows():
        arm = r.get(choice_col)
        if pd.isna(arm):
            continue
        arm = str(arm).strip().upper()
        W[i, 0] = float(r.get(f"best{tag}_share_{arm}", 0.0))
        W[i, 1] = float(r.get(f"second_best{tag}_share_{arm}", 0.0))
        W[i, 2] = float(r.get(f"third_best{tag}_share_{arm}", 0.0))
    return W

def rank_transition_matrix(
    df: pd.DataFrame,
    order: Tuple[str, str, str] = RANK_ORDER,
    return_counts: bool = False,
    rank_kind: str = "actual",
    per_suffix: str = "_per",
):
    """Rank transition matrix using tie-aware rank-share weights (no cross-session transitions)."""

    d = df.copy()
    if "trial" in d.columns:
        d["trial"] = pd.to_numeric(d["trial"], errors="coerce")
        d = d.sort_values("trial")

    W = choice_rank_weights(
        d,
        rank_source=rank_kind,
        per_suffix=per_suffix,
        require_shares=True,
    )

    counts = pd.DataFrame(0.0, index=order, columns=order, dtype=float)
    if len(W) >= 2:
        for w0, w1 in zip(W[:-1], W[1:]):
            counts += np.outer(w0, w1)

    probs = counts.div(counts.sum(axis=1).replace(0, 1.0), axis=0)
    return (probs, counts) if return_counts else probs

def rank_transition_matrix_across_sessions(
    session_trial_dfs: List[pd.DataFrame],
    order: Tuple[str, str, str] = RANK_ORDER,
    return_counts: bool = False,
    rank_kind: str = "actual",
    per_suffix: str = "_per",
):
    """Sum rank transitions within each session (no cross-session transitions), then normalize."""
    agg_counts = pd.DataFrame(0.0, index=order, columns=order, dtype=float)
    for df in session_trial_dfs:
        _, c = rank_transition_matrix(
            df,
            order=order,
            return_counts=True,
            rank_kind=rank_kind,
            per_suffix=per_suffix,
        )
        agg_counts = agg_counts.add(c, fill_value=0.0)
    probs = agg_counts.div(agg_counts.sum(axis=1).replace(0, 1.0), axis=0)
    return (probs, agg_counts) if return_counts else probs

# ============ Stay/Best-Alternative/Worst-Alternative analysis ============

# Mappings for best/worst alternatives from each starting rank
# Keys: 0=best, 1=second_best, 2=third_best (row indices)
# Values: target rank index
BEST_ALT = {0: 1, 1: 0, 2: 0}   # from best→second, from second→best, from third→best
WORST_ALT = {0: 2, 1: 2, 2: 1}  # from best→third, from second→third, from third→second

def _session_transition_counts_fractional(
    df: pd.DataFrame,
    *,
    rank_source: str = "actual",
    per_suffix: str = "_per",
    choice_col: str = "chosen_tower",
) -> np.ndarray:
    """
    Compute 3x3 fractional transition counts within a single session.
    Row i = rank at time t, column j = rank at time t+1.
    Uses tie-aware rank shares from choice_rank_weights.
    No cross-session transitions.
    """
    d = df.copy()
    if "trial" in d.columns:
        d["trial"] = pd.to_numeric(d["trial"], errors="coerce")
        d = d.sort_values("trial")

    W = choice_rank_weights(
        d,
        rank_source=rank_source,
        per_suffix=per_suffix,
        choice_col=choice_col,
        require_shares=True,
    )

    counts = np.zeros((3, 3), dtype=float)
    if len(W) >= 2:
        for w0, w1 in zip(W[:-1], W[1:]):
            counts += np.outer(w0, w1)
    return counts

def _subject_transition_counts_across_sessions(
    session_trial_dfs: List[pd.DataFrame],
    *,
    rank_source: str = "actual",
    per_suffix: str = "_per",
    choice_col: str = "chosen_tower",
) -> np.ndarray:
    """Sum 3x3 transition counts across sessions (no cross-session transitions)."""
    agg = np.zeros((3, 3), dtype=float)
    for df in session_trial_dfs:
        agg += _session_transition_counts_fractional(
            df, rank_source=rank_source, per_suffix=per_suffix, choice_col=choice_col
        )
    return agg

def compress_transitions_to_three_metrics(
    counts: np.ndarray,
    combine: str = "pooled",
) -> Dict[str, float]:
    """
    Compress 3x3 transition counts into three metrics:
      - Stay: probability of repeating the same rank
      - Best Alternative: probability of switching to the best alternative rank
      - Worst Alternative: probability of switching to the worst alternative rank

    Args:
        counts: 3x3 array of transition counts (row=rank_t, col=rank_t+1)
        combine: "pooled" (sum all numerators/denominators) or
                 "mean_by_row" (unweighted mean of per-row probabilities)

    Returns:
        Dict with keys: "Stay", "Best Alternative", "Worst Alternative"
    """
    assert counts.shape == (3, 3), "counts must be 3x3"

    if combine == "pooled":
        total = counts.sum()
        if total <= 0:
            return {
                "Stay": np.nan,
                "Best Alternative": np.nan,
                "Worst Alternative": np.nan,
            }
        stay = np.trace(counts)
        go_best = sum(counts[i, BEST_ALT[i]] for i in range(3))
        go_worst = sum(counts[i, WORST_ALT[i]] for i in range(3))
        return {
            "Stay": stay / total,
            "Best Alternative": go_best / total,
            "Worst Alternative": go_worst / total,
        }
    elif combine == "mean_by_row":
        vals = []
        for i in range(3):
            row_sum = counts[i, :].sum()
            if row_sum <= 0:
                continue
            vals.append([
                counts[i, i] / row_sum,
                counts[i, BEST_ALT[i]] / row_sum,
                counts[i, WORST_ALT[i]] / row_sum,
            ])
        if not vals:
            return {
                "Stay": np.nan,
                "Best Alternative": np.nan,
                "Worst Alternative": np.nan,
            }
        m = np.nanmean(np.array(vals), axis=0)
        return {
            "Stay": m[0],
            "Best Alternative": m[1],
            "Worst Alternative": m[2],
        }
    else:
        raise ValueError("combine must be 'pooled' or 'mean_by_row'")

def compute_threebar_metrics_by_subject(
    subject_to_session_dfs: Dict[str, List[pd.DataFrame]],
    *,
    rank_source: str = "actual",
    per_suffix: str = "_per",
    choice_col: str = "chosen_tower",
    combine: str = "pooled",
) -> Dict[str, Dict[str, float]]:
    """
    Compute Stay/Best-Alternative/Worst-Alternative metrics for each subject.

    Args:
        subject_to_session_dfs: Dict mapping subject ID to list of session DataFrames
        rank_source: "actual" or "belief"/"per"
        per_suffix: suffix for belief columns
        choice_col: column containing chosen arm
        combine: "pooled" or "mean_by_row"

    Returns:
        Dict mapping subject ID to dict of metrics
    """
    out = {}
    for subj, dfs in subject_to_session_dfs.items():
        C = _subject_transition_counts_across_sessions(
            dfs,
            rank_source=rank_source,
            per_suffix=per_suffix,
            choice_col=choice_col,
        )
        out[subj] = compress_transitions_to_three_metrics(C, combine=combine)
    return out