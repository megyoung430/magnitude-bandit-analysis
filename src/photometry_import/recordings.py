"""Load the photometry recordings of a session on pyControl time.

A session folder can hold several ``.ppd`` files (a restart, a short false start) and several
pyControl ``.tsv`` files. Each ``.ppd`` is paired with the ``.tsv`` whose rsync pulse train it
shares, and every sample is given its pyControl time. No filtering or other preprocessing is
done here: the signals are as the importer returns them.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.behavior_import.import_data import read_tsv_file
from src.photometry_import.pyphotometry import import_ppd, read_ppd_header
from src.photometry_import.sessions import PhotometrySession
from src.sync.pycontrol_time import (
    NoSyncPulsesError,
    align_to_pycontrol,
    photometry_sample_times,
    pycontrol_sync_times,
    sync_quality,
)
from src.sync.rsync import RsyncError, min_pulses_required

DEFAULT_MIN_PULSES = 20
# A true match pairs nearly every pulse of the shorter train, with interval errors of the order
# of one sample period (about 8 ms at 130 Hz). A chance fit of a short train inside a long
# unrelated one pairs few pulses and is off by hundreds of ms, so both limits reject it.
MIN_FRACTION_MATCHED = 0.5
MAX_MEDIAN_DISCREPANCY_MS = 50.0


@dataclass(frozen=True, eq=False)
class PhotometryRecording:
    """One ``.ppd`` recording with every sample on pyControl time.

    Signals are in volts. Arrays that the file does not hold (no LED-on/off split outside
    pulsed mode, no clipping flags in old file versions) are None. ``eq=False`` keeps
    identity comparison, since comparing numpy fields with ``==`` is ambiguous.

    Attributes:
        subject (str): Mouse ID.
        session (str): Session label, e.g. ``"ses-46"``.
        ppd_path (Path): The pyPhotometry file.
        tsv_path (Path): The pyControl file whose rsync pulses were matched.
        fs (float): Sampling rate (Hz).
        time (np.ndarray): pyControl time (s) of each sample; NaN outside the matched pulses.
        signal (np.ndarray): ``analog_1`` (nLightG2), LED-off baseline subtracted.
        control (np.ndarray): ``analog_2`` (isosbestic control), baseline subtracted.
        signal_led_on, signal_baseline, control_led_on, control_baseline (np.ndarray | None):
            The raw LED-on and LED-off samples of each channel.
        signal_clipping, control_clipping (np.ndarray | None): Samples where the ADC clipped.
        led_current (tuple[float, ...]): LED currents (mA) from the header.
        sync_quality (dict): :func:`src.sync.pycontrol_time.sync_quality` of the match.
        n_samples_outside_sync (int): Samples whose ``time`` is NaN.
    """

    subject: str
    session: str
    ppd_path: Path
    tsv_path: Path
    fs: float
    time: np.ndarray
    signal: np.ndarray
    control: np.ndarray
    signal_led_on: np.ndarray | None
    signal_baseline: np.ndarray | None
    control_led_on: np.ndarray | None
    control_baseline: np.ndarray | None
    signal_clipping: np.ndarray | None
    control_clipping: np.ndarray | None
    led_current: tuple[float, ...]
    sync_quality: dict
    n_samples_outside_sync: int


@dataclass(frozen=True)
class SkippedRecording:
    """A ``.ppd`` that could not be put on pyControl time, and why."""

    ppd_path: Path
    reason: str


def _check_candidate(
    ppd_pulses_ms: np.ndarray, tsv_pulses_s: np.ndarray
) -> tuple[bool, str | None]:
    """Whether the ``.ppd`` pulse train genuinely aligns with one pyControl train, and why not.

    The aligner can accept a chance fit of a short train, so the alignment must also pair at
    least ``MIN_FRACTION_MATCHED`` of the pulses with a median interval error under
    ``MAX_MEDIAN_DISCREPANCY_MS``.

    Returns:
        tuple[bool, str | None]: (aligned, error). A ValueError from the aligner (a malformed
        candidate train, e.g. repeated times) makes only this candidate a non-match and its
        message is returned as *error*; an ordinary non-match has error None.
    """
    if len(tsv_pulses_s) < min_pulses_required():
        return False, None
    try:
        aligner = align_to_pycontrol(ppd_pulses_ms, tsv_pulses_s, units_ms=1)
    except RsyncError:  # no match is the expected answer for an unrelated TSV
        return False, None
    except ValueError as error:  # malformed candidate: reject it alone, keep the reason
        return False, str(error)
    quality = sync_quality(aligner)
    aligned = (
        quality["fraction_matched"] >= MIN_FRACTION_MATCHED
        and quality["median_abs_interval_discrepancy_ms"] <= MAX_MEDIAN_DISCREPANCY_MS
    )
    return aligned, None


def _match_candidates(
    ppd_pulses_ms: np.ndarray, tsv_pulse_lists: Sequence[np.ndarray]
) -> tuple[int | None, dict[int, str]]:
    """(index of the single matching candidate or None, error message per rejected candidate)."""
    matches: list[int] = []
    errors: dict[int, str] = {}
    for i, pulses in enumerate(tsv_pulse_lists):
        aligned, error = _check_candidate(ppd_pulses_ms, pulses)
        if aligned:
            matches.append(i)
        if error is not None:
            errors[i] = error
    if len(matches) > 1:
        raise ValueError(f"the .ppd pulses match more than one .tsv (candidates {matches}).")
    return (matches[0] if matches else None), errors


def match_ppd_to_tsv(
    ppd_pulses_ms: np.ndarray,
    tsv_pulse_lists: Sequence[np.ndarray],
    min_pulses: int = DEFAULT_MIN_PULSES,
) -> int | None:
    """Index of the pyControl file whose rsync pulses are the ones this ``.ppd`` recorded.

    Args:
        ppd_pulses_ms (np.ndarray): Rising edges of the sync input in the ``.ppd`` (ms).
        tsv_pulse_lists (Sequence[np.ndarray]): rsync times (s) of each candidate ``.tsv``.
        min_pulses (int): Fewest ``.ppd`` pulses to trust a match; with fewer, a short train
            can fit the wrong file by chance. Default 20.

    Returns:
        int | None: Position in *tsv_pulse_lists* of the matching file, or None if the
        ``.ppd`` has too few pulses or no file matches. A candidate counts as matching only
        if the aligner accepts it, at least half of the shorter train's pulses are paired, and
        the median interval error is under 50 ms. A malformed candidate (the aligner raises
        ValueError) is a non-match for that candidate only.

    Raises:
        ValueError: If more than one file matches, which means the pairing is ambiguous.
    """
    if len(ppd_pulses_ms) < max(min_pulses, min_pulses_required()):
        return None
    return _match_candidates(ppd_pulses_ms, tsv_pulse_lists)[0]


def read_tsv_sync_pulses(tsv_path: Path) -> np.ndarray:
    """rsync times (s) of a pyControl file; empty if the run logged none (e.g. aborted early)."""
    try:
        return pycontrol_sync_times(read_tsv_file(tsv_path))
    except NoSyncPulsesError:
        return np.empty(0)


def _session_sync_pulses(
    session: PhotometrySession, injected: Mapping[Path, np.ndarray] | None
) -> dict[Path, np.ndarray]:
    """rsync times of each ``.tsv`` of *session*, read from disk unless *injected*."""
    if injected is None:
        return {path: read_tsv_sync_pulses(path) for path in session.tsv_paths}
    missing = [p for p in session.tsv_paths if p not in injected]
    if missing:
        raise ValueError(f"tsv_pulses_s has no entry for {[str(p) for p in missing]}.")
    return {path: np.asarray(injected[path], dtype=float) for path in session.tsv_paths}


def _check_header_subject(session: PhotometrySession, ppd_path: Path) -> None:
    subject = read_ppd_header(ppd_path).get("subject_ID")
    if subject != session.subject:
        raise ValueError(
            f"{ppd_path} was recorded from subject {subject!r} but sits in the folder of "
            f"{session.subject!r}."
        )


def _build_recording(
    session: PhotometrySession,
    ppd_path: Path,
    tsv_path: Path,
    data: dict,
    sample_times: np.ndarray,
    quality: dict,
) -> PhotometryRecording:
    return PhotometryRecording(
        subject=session.subject,
        session=session.session,
        ppd_path=ppd_path,
        tsv_path=tsv_path,
        fs=data["sampling_rate"],
        time=sample_times,
        signal=data["analog_1"],
        control=data["analog_2"],
        signal_led_on=data.get("analog_1_raw_LED_on"),
        signal_baseline=data.get("analog_1_raw_baseline"),
        control_led_on=data.get("analog_2_raw_LED_on"),
        control_baseline=data.get("analog_2_raw_baseline"),
        signal_clipping=data.get("analog_1_clipping"),
        control_clipping=data.get("analog_2_clipping"),
        led_current=tuple(data["LED_current"]),
        sync_quality=quality,
        n_samples_outside_sync=int(np.isnan(sample_times).sum()),
    )


def _load_one(
    session: PhotometrySession,
    ppd_path: Path,
    tsv_pulses: Mapping[Path, np.ndarray],
    min_pulses: int,
) -> PhotometryRecording | SkippedRecording:
    """Import one ``.ppd`` and put it on the time of the ``.tsv`` it shares pulses with."""
    _check_header_subject(session, ppd_path)
    data = import_ppd(ppd_path, low_pass=None, high_pass=None)
    ppd_pulses_ms = data["pulse_times_1"]
    if len(ppd_pulses_ms) < max(min_pulses, min_pulses_required()):
        return SkippedRecording(
            ppd_path, f"{len(ppd_pulses_ms)} sync pulses (fewer than min_pulses={min_pulses})."
        )
    if not tsv_pulses:
        return SkippedRecording(ppd_path, "the session has no .tsv file.")
    tsv_paths = list(tsv_pulses)
    index, errors = _match_candidates(ppd_pulses_ms, [tsv_pulses[p] for p in tsv_paths])
    if index is None:
        reason = "no .tsv in the session has a matching rsync train."
        if errors:
            detail = "; ".join(f"{tsv_paths[i].name}: {msg}" for i, msg in errors.items())
            reason += f" Rejected as malformed: {detail}"
        return SkippedRecording(ppd_path, reason)
    tsv_path = tsv_paths[index]
    sample_times, aligner = photometry_sample_times(data, tsv_pulses[tsv_path])
    return _build_recording(session, ppd_path, tsv_path, data, sample_times, sync_quality(aligner))


def load_photometry_recordings(
    session: PhotometrySession,
    min_pulses: int = DEFAULT_MIN_PULSES,
    tsv_pulses_s: Mapping[Path, np.ndarray] | None = None,
) -> tuple[list[PhotometryRecording], list[SkippedRecording]]:
    """Load every ``.ppd`` of a session on pyControl time; report those that cannot be.

    Signals are not filtered. Samples before the first and after the last matched sync pulse
    keep a NaN ``time``; each recording counts them in ``n_samples_outside_sync``.

    Args:
        session (PhotometrySession): From :func:`find_photometry_sessions`.
        min_pulses (int): Fewest sync pulses a ``.ppd`` needs to be matched. Default 20.
        tsv_pulses_s (Mapping[Path, np.ndarray] | None): rsync times (s) of each of the
            session's ``.tsv`` files, to use instead of reading them. Default None, meaning
            read them from disk.

    Returns:
        tuple[list[PhotometryRecording], list[SkippedRecording]]: The recordings that were
        aligned, and for each other ``.ppd`` the reason it was not (too few pulses, no
        matching ``.tsv``). Both are in file order.

    Raises:
        ValueError: If a ``.ppd`` header names a different subject than the session folder,
            or a ``.ppd`` matches more than one ``.tsv``.
    """
    tsv_pulses = _session_sync_pulses(session, tsv_pulses_s)
    results = [_load_one(session, ppd, tsv_pulses, min_pulses) for ppd in session.ppd_paths]
    recordings = [r for r in results if isinstance(r, PhotometryRecording)]
    skipped = [r for r in results if isinstance(r, SkippedRecording)]
    return recordings, skipped
