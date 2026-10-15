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

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.behavior_import.problem_numbering import (
    SessionFacts,
    number_subject_sessions,
    session_sort_key,
    tower_layout,
)

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
def _sorted_session_keys(grouped_tsv_files: dict, subject_id: str) -> list:
    """The ``(subject_id, session_key)`` keys of one subject, in chronological order."""
    keys = [k for k in grouped_tsv_files.keys() if k[0] == subject_id]
    return sorted(keys, key=lambda k: sort_session_key(k[1]))


class RunStartError(ValueError):
    """A recording's first ``run_start`` row is missing or not valid JSON."""


def source_path(df) -> str:
    """The file a DataFrame was read from, for error messages."""
    return df.attrs.get("source_path", "<unknown path>")


def _build_session_record(df_list: list, towers: dict, assignment, evidence: dict) -> dict:
    """The per-session entry returned by :func:`import_data`.

    Raw and expanded representations are independent copies, so expanding
    ``data`` cannot mutate the four-column DataFrames retained under ``df``.
    A session with a single recording holds the DataFrame itself, otherwise a list.
    """
    original_dfs = [df.copy() for df in df_list]
    expanded_dfs = [expand_content_columns(df.copy()) for df in df_list]
    return {
        "data": expanded_dfs[0] if len(expanded_dfs) == 1 else expanded_dfs,
        "df": original_dfs[0] if len(original_dfs) == 1 else original_dfs,
        "problem": assignment.problem,
        "problem_flag": assignment.flag,
        "settings_evidence": evidence,
        "task_name": evidence["task_name"],
        "choice_towers": towers.get("choice_towers", set()),
        "initiation_tower": towers.get("initiation_tower"),
    }


def _session_run_start(df_list: list, subject_id: str, session_key: str) -> dict:
    """The earliest recording's parsed ``run_start``; errors name subject, session and file."""
    try:
        return read_run_start(df_list[0])
    except RunStartError as err:
        raise ValueError(f"{subject_id} {session_key} ({source_path(df_list[0])}): {err}") from err


def _session_facts(towers: dict, evidence: dict) -> SessionFacts:
    return SessionFacts(
        tower_layout(towers["choice_towers"], towers["initiation_tower"]),
        evidence["starts_found_ports_false"],
        evidence["has_port_finding_trials"],
        evidence["task_name"],
    )


def _import_subject_sessions(grouped_tsv_files: dict, subject_id: str) -> dict:
    """``{session_key: session record}`` for one subject, with problem numbers assigned."""
    keys = _sorted_session_keys(grouped_tsv_files, subject_id)
    df_lists = [grouped_tsv_files[key] for key in keys]
    run_starts = [
        _session_run_start(df_list, subject_id, key[1])
        for df_list, key in zip(df_lists, keys, strict=True)
    ]
    towers = [towers_from_run_start(run_start) for run_start in run_starts]
    evidence = [
        settings_evidence(df_list, run_start)
        for df_list, run_start in zip(df_lists, run_starts, strict=True)
    ]
    # stacklevel 4: number_subject_sessions, this function, import_data, then the caller.
    assignments = number_subject_sessions(
        subject_id,
        [key[1] for key in keys],
        [_session_facts(t, e) for t, e in zip(towers, evidence, strict=True)],
        stacklevel=4,
    )
    return {
        key[1]: _build_session_record(df_list, t, assignment, e)
        for key, df_list, t, e, assignment in zip(
            keys, df_lists, towers, evidence, assignments, strict=True
        )
    }


def import_data(root: str | Path) -> dict:
    """Import behavioral experiment data from a directory tree of TSV files.

    Recursively scans *root* for ``.tsv`` files, groups them by subject and
    session (using directory-path components that match ``sub-<N>_id-<id>`` and
    ``ses-<N>_date-<YYYYMMDD>``), and merges multiple files that belong to the
    same session.  Each session also carries a ``"problem"`` number, assigned per
    subject in session order by the rule in
    :mod:`src.behavior_import.problem_numbering` (the same rule
    ``group_sessions_by_problem`` uses): a problem is the choice towers plus the
    initiation tower; a returning layout is a new problem; a session whose layout
    change looks like a settings error gets ``"problem": None`` and
    ``"problem_flag": "suspected_settings_error"`` and a warning.  Pretraining
    sessions (flag ``"pretraining"``, no warning), port-finding sessions
    (``"port_finding"``) and other sessions without choice towers
    (``"no_choice_towers"``) are likewise skipped, with a warning for the last two.
    Flagged sessions stay in the result for single-session analyses.  Subjects are returned in
    sorted order.

    Args:
        root: Root directory containing subject/session sub-directories with
            ``.tsv`` files.

    Returns:
        Nested dict of the form::

            {
              subject_id: {
                session_key: {
                  "data": DataFrame | list[DataFrame],  # JSON-expanded
                  "df": DataFrame | list[DataFrame],    # original 4-column TSV
                  "problem": int | None,                # 1-based per subject; None if flagged
                  "problem_flag": str | None,           # "suspected_settings_error" or None
                  "settings_evidence": dict,            # facts the flag rule used
                  "task_name": str | None,              # the file header's task_name row
                  "choice_towers": set,
                  "initiation_tower": str | None,
                }
              }
            }

        where *session_key* has the form ``"ses-<N>_date-<YYYYMMDD>"``.

    Raises:
        ValueError: If a file's path has no valid ``ses-<N>_date-<YYYYMMDD>``
            folder, or a session's first ``run_start`` row is missing or not valid
            JSON (the message names subject, session and file).

    ``"problem"`` agrees with ``extract_trials.group_sessions_by_problem``, which
    applies the same function to the layouts read from the trial data, except that it
    skips sessions without trial information before numbering.
    """
    root = Path(root).resolve()

    grouped_tsv_files = group_by_subj_and_ses(collect_tsv_files(root))

    unique_subjects = sorted({subj for subj, _ in grouped_tsv_files.keys()})
    # A plain loop, not a comprehension, so the warning stacklevel in
    # _import_subject_sessions does not depend on the Python version.
    subjects = {}
    for subject_id in unique_subjects:
        subjects[subject_id] = _import_subject_sessions(grouped_tsv_files, subject_id)

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
            df.attrs["source_path"] = str(tsv_path)  # names the file in later error messages
            tsv_files.append((tsv_path, df))
        # Known issue: every read failure is reduced to a printed warning and the file is
        # skipped, so a corrupt recording silently shrinks the session instead of raising.
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

    Raises:
        RunStartError: If the ``run_start`` row is missing or unparseable.
    """
    return towers_from_run_start(read_run_start(df))


def read_run_start(df) -> dict:
    """Parse the first ``run_start`` variable row of a recording.

    Raises:
        RunStartError: If the recording has no such row, or its content is empty or
            not a JSON object.
    """
    mask = (df.get("type") == "variable") & (df.get("subtype") == "run_start")
    rows = df.loc[mask, "content"]
    if rows.empty:
        raise RunStartError("no run_start row")
    content = rows.iloc[0]
    try:
        run_start = content if isinstance(content, dict) else json.loads(content)
    except (ValueError, TypeError) as err:
        raise RunStartError(f"first run_start row is not valid JSON: {content!r}") from err
    if not isinstance(run_start, dict):
        raise RunStartError(f"first run_start row is not a JSON object: {content!r}")
    return run_start


def settings_evidence(df_list, run_start: dict) -> dict:
    """The facts the settings-error check needs about a session.

    Args:
        df_list: The session's recordings, earliest first.
        run_start: The parsed ``run_start`` of the earliest recording.

    Returns:
        ``{"starts_found_ports_false": bool, "has_port_finding_trials": bool,
        "task_name": str | None}``.
        ``has_port_finding_trials`` is true if any recording prints a trial dict with
        ``num_t_found`` (the port-finding trials); the ``num_t_found`` key of
        ``run_start`` is always present and is ignored.  ``starts_found_ports_false``
        is true only if the first ``run_start`` has ``found_ports`` equal to ``false``;
        a missing or null value is not false.  ``task_name`` is the earliest
        recording's ``task_name`` info row, or ``None``.
    """
    printed = (
        parse_content_json(content)
        for df in df_list
        for content in df.loc[(df["type"] == "variable") & (df["subtype"] == "print"), "content"]
    )
    port_finding = any(isinstance(d, dict) and "num_t_found" in d for d in printed)
    return {
        "starts_found_ports_false": run_start.get("found_ports") is False,
        "has_port_finding_trials": port_finding,
        "task_name": read_task_name(df_list[0]),
    }


def read_task_name(df) -> str | None:
    """The recording's ``task_name`` info row (pyControl header), or ``None`` if absent."""
    rows = df.loc[(df["type"] == "info") & (df["subtype"] == "task_name"), "content"]
    return str(rows.iloc[0]) if not rows.empty else None


def towers_from_run_start(run_start: dict) -> dict:
    """The towers dict of :func:`extract_session_towers` from a parsed ``run_start``."""
    towers = {"choice_towers": set(), "initiation_tower": None}

    for key in CHOICE_TOWERS_KEYS:
        if key in run_start:
            towers["choice_towers"] = set(run_start[key])
            break

    for key in INITIATION_TOWER_KEYS:
        if key in run_start and run_start[key] is not None:
            towers["initiation_tower"] = run_start[key]
            break

    if (
        towers["initiation_tower"] is None
        and "problems" in run_start
        and "num_problem" in run_start
    ):
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
    parsed = df[content_col].map(parse_content_json)
    norm = pd.json_normalize(parsed, sep=".")
    for col in norm.columns:
        df[col] = norm[col]
    return df


def parse_content_json(x):
    """Parse one pyControl ``content`` cell as JSON.

    Returns ``{}`` for missing cells and for plain-text rows: event, state and
    most info rows log text such as ``poke_A``, which is not JSON. Any other
    failure (e.g. a non-string cell) is unexpected and raises.
    """
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, float) and pd.isna(x):
        return {}
    try:
        return json.loads(x)
    except json.JSONDecodeError:
        return {}


# ========== Metadata Helper Functions ==========
def parse_file_timestamp(tsv_path, df=None):
    """Return a datetime used to order multiple files within the same session.

    Attempts the following strategies in priority order:

    1. Timestamp embedded in the filename: ``...-YYYY-MM-DD-HHMMSS.tsv``.
    2. ``start_time`` value from the DataFrame's ``info`` / ``start_time``
       row (only checked if *df* is provided).
    3. File modification time (``mtime``) as a last resort; raises
       ``FileNotFoundError`` if the file does not exist.

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
    if (
        df is not None
        and "type" in df.columns
        and "subtype" in df.columns
        and "content" in df.columns
    ):
        s = df.loc[(df["type"].eq("info")) & (df["subtype"].eq("start_time")), "content"]
        if not s.empty:
            dt = pd.to_datetime(s.iloc[0], errors="coerce")
            if pd.notna(dt):
                return dt.to_pydatetime().replace(tzinfo=None)
    # Attempt 3: File modification time
    return datetime.fromtimestamp(p.stat().st_mtime)


def group_by_subj_and_ses(tsv_files: list) -> dict:
    """Group DataFrames by (subject_id, session_key), sorted within-session by time.

    Args:
        tsv_files: List of ``(tsv_path, dataframe)`` tuples as returned by
            :func:`collect_tsv_files`.

    Returns:
        Dict keyed by ``(subject_id, session_key)`` tuples, where each value
        is a list of ``pandas.DataFrame`` objects sorted chronologically by
        file timestamp (earliest first).  Files whose subject cannot be parsed
        are skipped with a ``[WARN]`` message.

    Raises:
        ValueError: If a file's path has no valid ``ses-<N>_date-<YYYYMMDD>``
            folder (see :func:`create_session_key`); such files are not skipped.
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

    Raises:
        ValueError: If no path component matches ``ses-<N>_date-<YYYYMMDD>``, or its
            date is not a real calendar date.  The message names the path.
    """
    tsv_path = Path(tsv_path)
    session_num = None
    session_date = None

    for part in tsv_path.parts:
        if part.startswith("ses-") and "date-" in part:
            m = re.match(r"ses-(\d+)_date-(\d{8})", part)
            if m:
                session_num = int(m.group(1))
                try:
                    session_date = datetime.strptime(m.group(2), "%Y%m%d")
                except ValueError as err:
                    raise ValueError(
                        f"Invalid date in session folder {part!r} of {tsv_path}"
                    ) from err

    if session_date is None:
        raise ValueError(
            f"No ses-<N>_date-<YYYYMMDD> folder in the path of {tsv_path}; "
            "cannot assign the file to a session."
        )
    session_key = f"ses-{session_num}_date-{session_date.strftime('%Y%m%d')}"
    return session_num, session_date, session_key


# Sessions are ordered by date, then session number: the key shared with
# ``group_sessions_by_problem`` (see ``problem_numbering.session_sort_key``).
sort_session_key = session_sort_key
