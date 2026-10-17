"""Find the photometry sessions of a cohort in the lab's ``sub-*/ses-*`` folder layout.

A cohort folder holds ``sub-XX_id-<mouse>/ses-NN_date-YYYYMMDD/`` folders, each with a
``behav/`` folder (pyControl ``.tsv``, camera ``*_GPIO_data.csv``) and a ``photometry/``
folder (pyPhotometry ``.ppd``). Only the folders are inspected here; no file is opened.
"""

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

_SUBJECT_FOLDER = re.compile(r"^sub-(?P<number>[^_]+)_id-(?P<mouse>.+)$")
_SESSION_FOLDER = re.compile(r"^ses-(?P<number>\d+)_date-(?P<date>\d{8})$")
_NUMERIC_SESSION_PREFIX = re.compile(r"^ses-\d+(_|$)")


@dataclass(frozen=True)
class PhotometrySession:
    """One mouse on one day, with the files each recording system produced.

    Attributes:
        subject (str): Mouse ID from the ``sub-XX_id-<mouse>`` folder, e.g. ``"RT_64_R"``.
        session (str): Session label, e.g. ``"ses-46"``.
        date (date): Date of the session.
        folder (Path): The ``ses-NN_date-YYYYMMDD`` folder.
        ppd_paths (tuple[Path, ...]): pyPhotometry files, sorted.
        tsv_paths (tuple[Path, ...]): pyControl files, sorted.
        gpio_paths (tuple[Path, ...]): Camera ``*_GPIO_data.csv`` files, sorted.
    """

    subject: str
    session: str
    date: date
    folder: Path
    ppd_paths: tuple[Path, ...]
    tsv_paths: tuple[Path, ...]
    gpio_paths: tuple[Path, ...]


def _parse_subject(folder: Path) -> str:
    match = _SUBJECT_FOLDER.match(folder.name)
    if match is None:
        raise ValueError(f"subject folder {folder} is not named 'sub-XX_id-<mouse>'.")
    return match["mouse"]


def _parse_session(folder: Path) -> tuple[str, date]:
    match = _SESSION_FOLDER.match(folder.name)
    if match is None:
        raise ValueError(f"session folder {folder} is not named 'ses-NN_date-YYYYMMDD'.")
    try:
        session_date = datetime.strptime(match["date"], "%Y%m%d").date()
    except ValueError as error:
        raise ValueError(f"session folder {folder} has an invalid date: {error}.") from error
    return f"ses-{match['number']}", session_date


def _sorted_files(folder: Path, pattern: str) -> tuple[Path, ...]:
    return tuple(sorted(folder.glob(pattern))) if folder.is_dir() else ()


def _session_folders(subject_folder: Path) -> list[Path]:
    """Numbered session folders of one subject, in session order.

    Folders with a non-numeric label (``ses-test``) are test recordings and are left out.
    """
    numbered = [
        folder
        for folder in subject_folder.iterdir()
        if folder.is_dir() and _NUMERIC_SESSION_PREFIX.match(folder.name)
    ]
    return sorted(numbered, key=lambda f: (int(f.name.split("_")[0][4:]), f.name))


def find_photometry_sessions(cohort_dir) -> list[PhotometrySession]:
    """List the sessions of a cohort that have at least one ``.ppd`` file.

    Args:
        cohort_dir (str | Path): Folder holding the ``sub-XX_id-<mouse>`` folders.

    Returns:
        list[PhotometrySession]: Ordered by subject folder, then session number. A session
        counts when its ``photometry/`` folder holds a ``.ppd``; test sessions
        (``ses-test`` or any non-numeric label) are skipped even if they hold files.

    Raises:
        FileNotFoundError: If *cohort_dir* does not exist.
        ValueError: If a folder that qualifies as a session is not named as expected.
    """
    cohort_dir = Path(cohort_dir)
    if not cohort_dir.is_dir():
        raise FileNotFoundError(f"cohort folder {cohort_dir} does not exist.")
    sessions = []
    subject_folders = sorted(
        f for f in cohort_dir.iterdir() if f.is_dir() and f.name.startswith("sub-")
    )
    for subject_folder in subject_folders:
        for folder in _session_folders(subject_folder):
            ppd_paths = _sorted_files(folder / "photometry", "*.ppd")
            if not ppd_paths:
                continue
            session, session_date = _parse_session(folder)
            sessions.append(
                PhotometrySession(
                    subject=_parse_subject(subject_folder),
                    session=session,
                    date=session_date,
                    folder=folder,
                    ppd_paths=ppd_paths,
                    tsv_paths=_sorted_files(folder / "behav", "*.tsv"),
                    gpio_paths=_sorted_files(folder / "behav", "*_GPIO_data.csv"),
                )
            )
    return sessions
