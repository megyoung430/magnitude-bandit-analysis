"""Import behavioral experiment data from directory trees of TSV files.

The primary entry point is :func:`import_data`, which recursively scans a root
directory for ``.tsv`` files, groups them by subject and session, and returns a
nested dict structure used by all downstream analysis functions.

Key constants defined here:

- ``CHOICE_TOWERS_KEYS`` / ``INITIATION_TOWER_KEYS`` – ordered lists of
  content-dict field names that store choice-tower and initiation-tower
  identifiers across different pyControl protocol versions.
- ``FILE_RE``, ``SESSION_RE``, ``SUBJECT_RE``, ``COHORT_RE`` – compiled
  regexes for parsing paths and filenames.
"""
from __future__ import annotations

import re
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
# ========== Content Key Aliases ==========
CHOICE_TOWERS_KEYS = (
    "current_choice_towers",
    "curr_choice_tows",
    "choice_towers",
)
INITIATION_TOWER_KEYS = (
    "current_initiation_tower",
    "curr_initiation_tow",
    "initiation_tower",
)

# ========== File, Session, Subject, and Cohort Formats ==========
FILE_RE = re.compile(r".*-(\d{4})-(\d{2})-(\d{2})-(\d{6})\.tsv$")
SESSION_RE = re.compile(r"ses-(\d+)_date-(\d{8})")
SUBJECT_RE = re.compile(r"sub-\d+_id-([^/]+)")
COHORT_RE = re.compile(r"cohort-(\d+)")

# ========== Main Data Import Functions ==========
def import_data(root: str | Path) -> dict:
    """Import behavioral experiment data from a directory tree of TSV files.

    Recursively scans *root* for ``.tsv`` files, groups them by subject and
    session (using directory-path components that match ``sub-<N>_id-<id>`` and
    ``ses-<N>_date-<YYYYMMDD>``), and merges multiple files that belong to the
    same session.  Also tracks problem number by detecting changes in the set
    of choice towers across sessions.

    Args:
        root: Root directory containing subject/session sub-directories with
            ``.tsv`` files.

    Returns:
        Nested dict of the form::

            {
              subject_id: {
                session_key: {
                  "data": DataFrame | list[DataFrame],
                  "df": DataFrame | list[DataFrame],
                  "problem": int,
                  "choice_towers": set,
                  "initiation_tower": str | None,
                }
              }
            }

        where *session_key* has the form ``"ses-<N>_date-<YYYYMMDD>"``.
    """
    root = Path(root).resolve()

    # Collect all TSV files with metadata
    tsv_files = collect_tsv_files(root)

    # Group by subject and session
    grouped_tsv_files = group_by_subj_and_ses(tsv_files)

    # Merge and process each session
    subjects = {}
    unique_subjects = {subj for subj, _ in grouped_tsv_files.keys()}
    for subject_id in unique_subjects:
        subjects.setdefault(subject_id, {})

        relevant_sessions_keys = [k for k in grouped_tsv_files.keys() if k[0] == subject_id]
        sorted_sessions = sorted(relevant_sessions_keys, key=lambda k: sort_session_key(k[1]))
        num_problem = 1
        prev_towers = extract_session_towers(grouped_tsv_files[sorted_sessions[0]][0])

        for key in sorted_sessions:
            df_list = grouped_tsv_files[key]

            # Check if the problem has switched
            towers = extract_session_towers(df_list[0])
            if set(towers) != set(prev_towers):
                num_problem += 1
                prev_towers = towers
            
            original_dfs = df_list[0] if len(df_list) == 1 else df_list
            expanded_dfs = [expand_content_columns(df) for df in df_list]
            subjects[subject_id][key[1]] = {
                "data": expanded_dfs[0] if len(expanded_dfs) == 1 else expanded_dfs,
                "df": original_dfs[0] if len(original_dfs) == 1 else original_dfs,
                "problem": num_problem,
                "choice_towers": towers.get("choice_towers", set()),
                "initiation_tower": towers.get("initiation_tower")
            }

    total = sum(len(s) for s in subjects.values())
    print(f"[INFO] Processed {len(subjects)} subjects(s), {total} session(s).")
    return subjects

# ========== File Parsing Helper Functions ==========
def collect_tsv_files(root: Path) -> list:
    """Recursively scan *root* and read every TSV file found.

    Args:
        root: Directory to search recursively.

    Returns:
        List of ``(tsv_path, dataframe)`` tuples for every ``.tsv`` file that
        was read successfully.  Files that raise an exception during reading are
        skipped with a ``[WARN]`` message printed to stdout.
    """
    tsv_files = []
    for tsv_path in root.rglob("*.tsv"):
        try:
            df = read_tsv_file(tsv_path)
            tsv_files.append((tsv_path, df))
        except Exception as e:
            print(f"[WARN] Failed to read {tsv_path}: {e}.")
    return tsv_files

def read_tsv_file(tsv_path, sep="\t"):
    """Read a single TSV file into a pandas DataFrame.

    Args:
        tsv_path: Path to the ``.tsv`` file.
        sep: Column separator (default: ``"\\t"``).

    Returns:
        A ``pandas.DataFrame`` with one row per line of the file.
    """
    df = pd.read_csv(tsv_path, sep=sep)
    return df

def extract_session_towers(df):
    """Extract choice-tower and initiation-tower identifiers from a session DataFrame.

    Parses the ``run_start`` variable row in *df* and returns the set of choice
    tower keys and the initiation tower key.  Field names are looked up via the
    ``CHOICE_TOWERS_KEYS`` and ``INITIATION_TOWER_KEYS`` alias lists to handle
    protocol-version differences.

    Args:
        df: A session DataFrame as returned by :func:`read_tsv_file`, expected
            to contain at least one row with ``type=="variable"`` and
            ``subtype=="run_start"``.

    Returns:
        Dict ``{"choice_towers": set, "initiation_tower": str | None}``.
    """
    towers = {"choice_towers": set(), "initiation_tower": None}
    
    mask = (df.get("type") == "variable") & (df.get("subtype") == "run_start")
    run_start = df.loc[mask]
    run_start = json.loads(run_start['content'].iloc[0])

    for key in CHOICE_TOWERS_KEYS:
        if key in run_start:
            towers["choice_towers"] = set(run_start[key])
            break

    for key in INITIATION_TOWER_KEYS:
        if key in run_start and run_start[key] is not None:
            towers["initiation_tower"] = run_start[key]
            break

    if towers["initiation_tower"] is None and "problems" in run_start and "num_problem" in run_start:
        num_problem = str(run_start["num_problem"])
        problems_dict = run_start["problems"]
        if num_problem in problems_dict:
            tower_list = problems_dict[num_problem]
            if isinstance(tower_list, list):
                if len(tower_list) == 4:
                    # First element is non-default initiation tower
                    towers["initiation_tower"] = tower_list[0]
                    towers["choice_towers"] = set(tower_list[1:])
                elif len(tower_list) == 3:
                    # No initiation tower listed, default is B2
                    towers["initiation_tower"] = "B2"
                    towers["choice_towers"] = set(tower_list)

    return towers

def expand_content_columns(df, content_col="content"):
    """Flatten a JSON-string column into individual DataFrame columns.

    Parses each value in *content_col* as JSON and normalises the resulting
    dicts into new columns (dot-separated for nested keys), adding them
    in-place to *df*.

    Args:
        df: A ``pandas.DataFrame`` containing a column of JSON strings.
        content_col: Name of the column to expand (default: ``"content"``).

    Returns:
        The same ``df`` with additional columns added for each JSON key found.
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
    parsed = df[content_col].map(safe_json_load)
    norm = pd.json_normalize(parsed, sep=".")
    for col in norm.columns:
        df[col] = norm[col]
    return df

# ========== Metadata Helper Functions ==========
def parse_file_timestamp(tsv_path, df=None):
    """Return a datetime used to order multiple files within the same session.

    Attempts the following strategies in priority order:

    1. Timestamp embedded in the filename: ``...-YYYY-MM-DD-HHMMSS.tsv``.
    2. ``start_time`` value from the DataFrame's ``info`` / ``start_time``
       row (only checked if *df* is provided).
    3. File modification time (``mtime``) as a last resort.

    Args:
        tsv_path: Path to the ``.tsv`` file.
        df: Optional ``pandas.DataFrame`` for the same file; used to try
            reading the ``start_time`` info row (default: ``None``).

    Returns:
        A ``datetime.datetime`` object (timezone-naive).
    """
    p = Path(tsv_path)
    m = FILE_RE.match(p.name)
    # Attempt 1: Timestamp in filename
    if m:
        y, mo, d, hhmmss = m.groups()
        return datetime.strptime(f"{y}{mo}{d}{hhmmss}", "%Y%m%d%H%M%S")
    # Attempt 2: start_time from info row in dataframe
    if df is not None and "type" in df.columns and "subtype" in df.columns and "content" in df.columns:
        s = df.loc[(df["type"].eq("info")) & (df["subtype"].eq("start_time")), "content"]
        if not s.empty:
            dt = pd.to_datetime(s.iloc[0], errors="coerce")
            if pd.notna(dt):
                return dt.to_pydatetime().replace(tzinfo=None)
    # Attempt 3: File modification time
    try:
        return datetime.fromtimestamp(p.stat().st_mtime)
    except Exception:
        return datetime.min

def group_by_subj_and_ses(tsv_files: list) -> dict:
    """Group DataFrames by (subject_id, session_key), sorted within-session by time.

    Args:
        tsv_files: List of ``(tsv_path, dataframe)`` tuples as returned by
            :func:`collect_tsv_files`.

    Returns:
        Dict keyed by ``(subject_id, session_key)`` tuples, where each value
        is a list of ``pandas.DataFrame`` objects sorted chronologically by
        file timestamp (earliest first).  Files whose subject or session key
        cannot be parsed are skipped with a ``[WARN]`` message.
    """
    grouped = {}
    for tsv_path, df in tsv_files:
        subject_id, subject_number, subject_key = create_subject_id(tsv_path)
        if not subject_id:
            print(f"[WARN] Skipping {tsv_path}: no subject_id parsed.")
            continue
        session_num, session_date, session_key = create_session_key(tsv_path)
        file_ts = parse_file_timestamp(tsv_path, df)
        grouped.setdefault((subject_id, session_key), []).append((file_ts, tsv_path, df))
    # If there are multiple files for a session, sort each session's files chronologically
    for k, items in grouped.items():
        items.sort(key=lambda x: x[0])
        grouped[k] = [df for _, _, df in items]
    return grouped

def create_subject_id(tsv_path) -> tuple:
    """Parse a standardised subject identifier from a file path.

    Scans *tsv_path* component-by-component for a part matching
    ``sub-<N>_id-<id>``.

    Args:
        tsv_path: Path to a ``.tsv`` file whose directory tree contains a
            ``sub-<N>_id-<id>`` component.

    Returns:
        A three-tuple ``(subject_id, subject_number, subject_key)`` where
        *subject_id* is the string after ``id-``, *subject_number* is the
        integer after ``sub-``, and *subject_key* is the full
        ``"sub-<N>_id-<id>"`` string.  All three are ``None`` / ``"sub-None_id-None"``
        if the pattern is not found.
    """
    tsv_path = Path(tsv_path)
    subject_id = None
    subject_number = None

    for part in tsv_path.parts:
        if part.startswith("sub-") and "id-" in part:
            m = re.match(r"sub-(\d+)_id-(.+)", part)
            if m:
                subject_number = int(m.group(1))
                subject_id = m.group(2)
    
    subject_key = f"sub-{subject_number}_id-{subject_id}"
    return subject_id, subject_number, subject_key

def create_session_key(tsv_path) -> tuple:
    """Parse a standardised session key from a file path.

    Scans *tsv_path* component-by-component for a part matching
    ``ses-<N>_date-<YYYYMMDD>``.

    Args:
        tsv_path: Path to a ``.tsv`` file whose directory tree contains a
            ``ses-<N>_date-<YYYYMMDD>`` component.

    Returns:
        A three-tuple ``(session_num, session_date, session_key)`` where
        *session_num* is the integer after ``ses-``, *session_date* is a
        ``datetime.datetime`` object, and *session_key* is the normalised
        string ``"ses-<N>_date-<YYYYMMDD>"``.
    """
    tsv_path = Path(tsv_path)
    session_num = None
    session_date = None

    for part in tsv_path.parts:
        if part.startswith("ses-") and "date-" in part:
            m = re.match(r"ses-(\d+)_date-(\d{8})", part)
            if m:
                session_num = int(m.group(1))
                session_date = datetime.strptime(m.group(2), "%Y%m%d")
    
    session_key = f"ses-{session_num}_date-{session_date.strftime('%Y%m%d')}"
    return session_num, session_date, session_key

def sort_session_key(session_key: str) -> tuple:
    """Return a sort key for a session string of the form ``ses-N_date-YYYYMMDD``.

    Args:
        session_key: Session identifier string, e.g. ``"ses-3_date-20240115"``.

    Returns:
        A ``(date_int, session_number)`` tuple suitable for use as a sort key,
        where *date_int* is the date as a plain integer (``YYYYMMDD``) and
        *session_number* is the integer session index.
    """
    match = re.search(r"ses-(\d+)_date-(\d{8})", session_key)
    ses_num = int(match.group(1))
    date = int(match.group(2))
    return (date, ses_num)