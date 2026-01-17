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
def import_data(root):
    """Import behavioral experiment data from TSV files.

    Args:
        root (string): Root directory containing TSV files.

    Returns:
        subjects (nested dictionary): {subject_id: {session_key: {"data": DataFrame, "choice_towers": set, "initiation_tower": str}}}
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
def collect_tsv_files(root):
    """Scan directory and reads all TSV files and returns list of (path, dataframe) tuples."""
    tsv_files = []
    for tsv_path in root.rglob("*.tsv"):
        try:
            df = read_tsv_file(tsv_path)
            tsv_files.append((tsv_path, df))
        except Exception as e:
            print(f"[WARN] Failed to read {tsv_path}: {e}.")
    return tsv_files

def read_tsv_file(tsv_path, sep="\t"):
    df = pd.read_csv(tsv_path, sep=sep)
    return df

def extract_session_towers(df):
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
    
    return towers

def expand_content_columns(df, content_col="content"):
    """Flatten JSON content into DataFrame columns."""
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
    """ Returns a datetime used to order multiple files within the same session.
    Priority:
      1) timestamp in filename: ...-YYYY-MM-DD-HHMMSS.tsv
      2) df info row start_time (if provided)
      3) file mtime (last resort)
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

def group_by_subj_and_ses(tsv_files):
    """Group DataFrames by (subject_id, session_key), and sort within-session by time."""
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

def create_subject_id(tsv_path):
    """Create a standardized subject id from metadata"""
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

def create_session_key(tsv_path):
    """Create a standardized session key from metadata"""
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

def sort_session_key(session_key):
    match = re.search(r"ses-(\d+)_date-(\d{8})", session_key)
    ses_num = int(match.group(1))
    date = int(match.group(2))
    return (date, ses_num)