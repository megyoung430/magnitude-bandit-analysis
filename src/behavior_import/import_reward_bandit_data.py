import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# ---- regex to robustly parse subject/session from any ancestor directory ----
SESSION_RE = re.compile(r"ses-(\d+)_date-(\d{8})")
SUBJECT_RE = re.compile(r"sub-\d+_id-([^/]+)")

# ============ JSON/Content Processing ============

def _read_single_tsv(tsv_path: Path, sep: str = "\t") -> pd.DataFrame:
    """
    Read a single TSV file and add metadata columns.
    
    Args:
        tsv_path: Path to TSV file
        sep: Column separator (default: tab)
    
    Returns:
        DataFrame with original columns plus metadata
    """
    df = pd.read_csv(tsv_path, sep=sep, engine="python")
    
    # Ensure expected columns exist
    for col in ("time", "type", "subtype", "content"):
        if col not in df.columns:
            df[col] = pd.NA
    
    # Flatten JSON content into columns
    df = _expand_content_columns(df, content_col="content")
    
    # Extract and add metadata
    subj, ses_num, ses_date = _extract_subject_session(tsv_path)
    df["subject_id"] = subj
    df["session_number"] = ses_num
    df["session_date"] = ses_date
    df["source_file"] = str(tsv_path)
    
    return df

def _safe_json_load(x):
    """Parse JSON safely, returning empty dict on failure."""
    if pd.isna(x):
        return {}
    if isinstance(x, (dict, list)):
        return x
    try:
        return json.loads(x)
    except Exception:
        return {}

def _expand_content_columns(df: pd.DataFrame, content_col: str = "content") -> pd.DataFrame:
    """Flatten JSON content into DataFrame columns."""
    parsed = df[content_col].map(_safe_json_load)
    
    # Check if any content exists
    if parsed.empty or not parsed.map(bool).any():
        return df
    
    # Normalize and add columns
    norm = pd.json_normalize(parsed, sep=".")
    for col in norm.columns:
        df[col] = norm[col]
    
    return df

def _normalize_tower_token(token) -> Optional[str]:
    """Normalize a tower token to a clean string identifier."""
    if token is None or (isinstance(token, float) and pd.isna(token)):
        return None
    if isinstance(token, str):
        stripped = token.strip().strip("[]'\"")
        return stripped or None
    return str(token)

def _normalize_tower_list(value) -> set:
    """Best-effort parsing of tower lists that may be JSON, Python repr, or CSV."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    if isinstance(value, list):
        return {t for t in (_normalize_tower_token(v) for v in value) if t}
    if isinstance(value, str):
        loaded = _safe_json_load(value)
        if isinstance(loaded, list):
            return {t for t in (_normalize_tower_token(v) for v in loaded) if t}
        # Fallback: comma-separated or space-separated
        parts = re.split(r"[,\s]+", value.strip("[] "))
        return {t for t in (_normalize_tower_token(v) for v in parts) if t}
    return {_normalize_tower_token(value)}

# ============ Metadata Extraction ============

def _extract_subject_session(tsv_path: Path) -> tuple[str, str, str]:
    """
    Extract metadata from file path.
    
    Returns:
        (subject_id, session_number, session_date)
    """
    subject_id = None
    session_number = None
    session_date = None
    
    # Search path hierarchy
    for p in [tsv_path] + list(tsv_path.parents):
        if not session_number:
            session_number, session_date = _extract_session_from_path(p)
        if not subject_id:
            subject_id = _extract_subject_from_path(p)
    
    # Fallback to filename
    if not subject_id:
        subject_id = _extract_subject_from_filename(tsv_path)
    
    # Standardize session number
    if session_number:
        session_number = f"{int(session_number):02d}"
    
    return subject_id, session_number, session_date

def _extract_session_from_path(path: Path) -> tuple[str, str]:
    """Extract session number and date from path component."""
    m = SESSION_RE.search(path.name)
    if m:
        ses_num = m.group(1)
        ymd = m.group(2)
        ses_date = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}"
        return ses_num, ses_date
    return None, None


def _extract_subject_from_path(path: Path) -> str:
    """Extract subject ID from path component."""
    m = SUBJECT_RE.search(path.name)
    return m.group(1) if m else None


def _extract_subject_from_filename(tsv_path: Path) -> str:
    """Extract subject ID from filename as fallback."""
    stem = tsv_path.stem
    if "-" in stem:
        return stem.split("-", 1)[0]
    return None

# ============ DataFrame Utilities ============

def _first_non_na(series: pd.Series):
    """Return first non-NA value from Series."""
    if series is None:
        return None
    s = series.dropna()
    return s.iloc[0] if len(s) > 0 else None


def _dedupe_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows based on identifying columns."""
    key_cols = ["source_file", "file_part", "time", "type", "subtype", "content"]
    cols = [c for c in key_cols if c in df.columns]
    
    if not cols:
        return df
    
    # Create unique key
    key = df[cols].astype(str).agg("||".join, axis=1)
    return df.loc[~key.duplicated()].copy()


def _extract_session_towers(df: pd.DataFrame) -> Dict[str, Optional[object]]:
    """Extract choice/initiation towers from run_start events, if present."""
    towers = {"choice_towers": set(), "initiation_tower": None}

    if df.empty:
        return towers

    mask = (df.get("type") == "variable") & (df.get("subtype") == "run_start")
    run_start = df.loc[mask]

    choice_raw = _first_non_na(run_start.get("current_choice_towers"))
    init_raw = _first_non_na(run_start.get("current_initiation_tower"))

    towers["choice_towers"] = _normalize_tower_list(choice_raw)
    towers["initiation_tower"] = _normalize_tower_token(init_raw)

    return towers

# ============ Import Functions ============
   
def import_reward_bandit_data(
    root: str = "../../3x3_maze_reward_bandit/rawdata",
    make_global_time: bool = True,
    drop_exact_duplicates: bool = True,
    verbose: bool = True
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Import behavioral experiment data from TSV files.
    
    Args:
        root: Root directory containing TSV files
        make_global_time: Create continuous timeline across file parts
        drop_exact_duplicates: Remove duplicate rows
        verbose: Print progress messages
    
    Returns:
        Nested dict: {subject_id: {session_key: {"data": DataFrame, "choice_towers": set, "initiation_tower": str}}}
        
    Example:
        >>> data = import_reward_bandit_data("/path/to/data")
        >>> data["MY_24N"]["ses-01_2025-08-31"].head()
    """
    root = Path(root).resolve()
    
    # Step 1: Collect all TSV files with metadata
    tsv_files = _collect_tsv_files(root, verbose)
    
    # Step 2: Group by subject and session
    staging = _group_by_subject_session(tsv_files, verbose)
    
    # Step 3: Merge and process each session
    subjects = _merge_sessions(staging, make_global_time, drop_exact_duplicates)
    
    if verbose:
        total = sum(len(s) for s in subjects.values())
        print(f"[INFO] Built {len(subjects)} subject(s), {total} session(s).")
    
    return subjects

def _collect_tsv_files(root: Path, verbose: bool) -> list:
    """Scan directory and read all TSV files."""
    tsv_files = []
    for tsv_path in root.rglob("*.tsv"):
        try:
            df = _read_single_tsv(tsv_path)
            tsv_files.append((tsv_path, df))
        except Exception as e:
            if verbose:
                print(f"[WARN] Failed to read {tsv_path}: {e}")
    return tsv_files

def _group_by_subject_session(tsv_files: list, verbose: bool) -> dict:
    """Group DataFrames by (subject_id, session_key)."""
    staging = {}
    
    for tsv_path, df in tsv_files:
        subj = _first_non_na(df.get("subject_id"))
        if not subj:
            if verbose:
                print(f"[WARN] Skipping {tsv_path}: no subject_id parsed.")
            continue
        
        session_key = _create_session_key(df, tsv_path)
        staging.setdefault((subj, session_key), []).append(df)
    
    return staging

def _create_session_key(df: pd.DataFrame, tsv_path: Path) -> str:
    """Create standardized session key from metadata."""
    ses_num = _first_non_na(df.get("session_number"))
    ses_date = _first_non_na(df.get("session_date"))
    
    if ses_num and ses_date:
        return f"ses-{ses_num}_{ses_date}"
    elif ses_num:
        return f"ses-{ses_num}"
    elif ses_date:
        return f"ses-unknown_{ses_date}"
    else:
        return tsv_path.stem

def _merge_sessions(
    staging: dict,
    make_global_time: bool,
    drop_exact_duplicates: bool
) -> dict:
    """Merge multiple files within each session."""
    subjects = {}
    
    for (subj, session_key), dfs in staging.items():
        # Sort files by timestamp
        sorted_parts = _sort_session_parts(dfs)
        
        # Add file_part and global_time
        processed_parts = _process_session_parts(sorted_parts, make_global_time)
        
        # Concatenate and clean
        df_session = pd.concat(processed_parts, ignore_index=True)
        
        if drop_exact_duplicates:
            df_session = _dedupe_safe(df_session)
        
        df_session = _sort_session_data(df_session, make_global_time)
        
        towers = _extract_session_towers(df_session)

        subjects.setdefault(subj, {})
        subjects[subj][session_key] = {
            "data": df_session,
            "df": df_session,  # backward-compatible alias
            "choice_towers": towers.get("choice_towers", set()),
            "initiation_tower": towers.get("initiation_tower"),
        }
    
    return subjects

def _sort_session_parts(dfs: list) -> list:
    """Sort DataFrames by earliest timestamp."""
    parts = []
    for d in dfs:
        first_time = (
            pd.to_numeric(d.get("time"), errors="coerce").min()
            if "time" in d.columns else np.nan
        )
        src = d["source_file"].iloc[0] if "source_file" in d.columns else ""
        parts.append((first_time, src, d))
    
    parts.sort(key=lambda x: (np.isnan(x[0]), x[0], x[1]))
    return parts

def _process_session_parts(parts: list, make_global_time: bool) -> list:
    """Add file_part and global_time to each DataFrame."""
    processed = []
    
    for idx, (_, _, df) in enumerate(parts, start=1):
        df = df.copy()
        df["file_part"] = idx
        df["time"] = pd.to_numeric(df.get("time", np.nan), errors="coerce")
        
        if make_global_time:
            offset = _calculate_time_offset(processed, df)
            df["global_time"] = df["time"] + offset
        
        processed.append(df)
    
    return processed

def _calculate_time_offset(previous_dfs: list, current_df: pd.DataFrame) -> float:
    """Calculate time offset to create continuous timeline."""
    if not previous_dfs:
        return 0.0
    
    prev = previous_dfs[-1]
    prev_max = pd.to_numeric(prev.get("global_time"), errors="coerce").max()
    
    if pd.isna(prev_max):
        prev_max = pd.to_numeric(prev.get("time"), errors="coerce").max()
        if pd.isna(prev_max):
            prev_max = 0.0
    
    return float(prev_max) + 1e-6

def _sort_session_data(df: pd.DataFrame, has_global_time: bool) -> pd.DataFrame:
    """Sort final session DataFrame."""
    if has_global_time and "global_time" in df.columns:
        sort_cols = ["global_time", "file_part", "time"]
    else:
        sort_cols = ["file_part", "time"]
    
    return df.sort_values(sort_cols, kind="stable", ignore_index=True)