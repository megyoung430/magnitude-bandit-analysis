"""One imaging run through the Zong et al. 2022 pipeline, and every run of a variant.

Composes the pure steps (``zong2022_signal``, ``zong2022_responses``, ``zong2022_motion``) with
the readers in ``zong2022_io``. The slow stages (extraction, the signal processing, the
registration metrics) are cached in the repo's disposable cache; nothing is written next to the
data.
"""

import dataclasses
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from src.behavior_analysis.outcome_events import OutcomeParseError, load_outcome_events
from src.funcimg_analysis.collect import check_same_window, variant_runs
from src.funcimg_analysis.event_frames import mean_by_category, pooled_mean_sem
from src.funcimg_analysis.frame_times import (
    FRAME_TIMES_FILE,
    RunMismatchError,
    check_run_source,
    load_frame_times,
)
from src.funcimg_analysis.run_responses import PULSE_CATEGORIES
from src.funcimg_analysis.zong2022_io import (
    aligned_labels,
    array_digest,
    cached,
    default_diameter_px,
    extract_cached,
    file_signature,
    read_frames,
    registration_offsets,
    renumber_labels,
    seg_paths,
)
from src.funcimg_analysis.zong2022_motion import (
    MotionSettings,
    registration_metrics,
    registration_settings,
    sample_frame_indices,
)
from src.funcimg_analysis.zong2022_responses import (
    ResponseSettings,
    aligned_windows,
    chance_summary,
    circular_shifts,
    pooled_responses,
    response_frames,
    responsive_cells,
    reward_frame_mask,
    valid_span,
)
from src.funcimg_analysis.zong2022_signal import SignalResult, Zong2022Settings, analyse_traces

OUTCOME_KINDS = {"reward": True, "no_reward": False}  # name: rewarded?


class RunLeftOut(Exception):
    """A run that cannot be analysed for a known reason; the message says what is missing.

    Only subclasses are raised, and :func:`analyse_runs` skips exactly these. Any other error
    (including a stray ``FileNotFoundError``) is a bug or damaged data and stops the loop.
    """


class MissingInputError(RunLeftOut):
    """A file the run needs (frame times, behaviour, Cellpose segmentation) is not there."""


class MaskAlignmentRefused(RunLeftOut):
    """mini2p refused to align the Cellpose masks onto the plane."""


class NoUsableCells(RunLeftOut):
    """Every ROI of the run was excluded (no positive F0, or too few baseline points)."""


def report_to(report: Callable[[str], None] | None) -> Callable[[str], None]:
    return report if report is not None else (lambda line: None)


# ---- stages ------------------------------------------------------------------------------------


def load_run_inputs(run_dir, variant: str, load_outcomes: Callable):
    """The run's frame times (checked against the batch's source TIFF) and outcome table.

    Raises:
        MissingInputError: If the run has no ``frame_times.npz`` or no behaviour file.
        RunMismatchError: If the variant was made from another acquisition's TIFF.
    """
    run_dir = Path(run_dir)
    if not (run_dir / FRAME_TIMES_FILE).is_file():
        raise MissingInputError(f"no {FRAME_TIMES_FILE} in {run_dir}; write the frame times first.")
    frames = load_frame_times(run_dir)
    check_run_source(frames.tiff.name, run_dir, variants=(variant,))
    try:
        outcomes = load_outcomes(frames.tsv)
    except FileNotFoundError as error:
        raise MissingInputError(f"no behaviour file {frames.tsv} for {run_dir}.") from error
    return frames, outcomes


def extract_run(
    plane_dir, run_dir, cache_dir, report: Callable[[str], None], diameter_px: int | None = None
) -> dict:
    """Align the run's Cellpose masks onto the plane and extract Fcell and Fnp (both channels).

    The aligned labels are renumbered 1..n: alignment can drop whole ROIs, leaving gaps in the
    Cellpose numbers, while the extraction gives one row per remaining ROI. After this, row
    ``i`` of the traces is label ``i + 1`` of ``labels``, and ``cellpose_labels[i]`` is the
    number it had in Cellpose.

    Args:
        diameter_px: ROI diameter for the alignment and ROI statistics; default is mini2p's.

    Raises:
        MissingInputError: If the segmentation or its image is missing.
        MaskAlignmentRefused: If mini2p refuses to align the masks.
    """
    diameter_px = default_diameter_px() if diameter_px is None else diameter_px
    plane_dir = Path(plane_dir)
    seg_file, mask_image = seg_paths(run_dir)
    for what, path in (("segmentation", seg_file), ("mask image", mask_image)):
        if not path.is_file():
            raise MissingInputError(f"no Cellpose {what}: {path}")
    ops = np.load(plane_dir / "ops.npy", allow_pickle=True).item()
    report(f"masks: aligning {seg_file.name} onto {plane_dir.parents[1].name}")
    try:
        aligned, alignment = aligned_labels(seg_file, mask_image, ops, diameter_px)
    except ValueError as error:
        raise MaskAlignmentRefused(f"mask alignment refused: {error}") from error
    labels, cellpose_labels = renumber_labels(aligned)
    report(f"extract: {len(cellpose_labels)} ROIs from the registered binaries (cached)")
    traces = extract_cached(labels, plane_dir, diameter_px, cache_dir)
    return {
        "ops": ops,
        "labels": labels,
        "cellpose_labels": cellpose_labels,
        "alignment": alignment,
        **traces,
    }


def compact(result: SignalResult) -> SignalResult:
    """*result* with its float traces stored as float32, so the cached copy is half the size."""
    changes = {
        field.name: getattr(result, field.name).astype(np.float32)
        for field in dataclasses.fields(result)
        if isinstance(getattr(result, field.name), np.ndarray)
        and getattr(result, field.name).dtype == np.float64
        and getattr(result, field.name).ndim == 2
    }
    return dataclasses.replace(result, **changes)


def signal_cached(
    f: np.ndarray, fneu: np.ndarray, period_s: float, settings: Zong2022Settings, cache_dir
) -> SignalResult:
    """:func:`analyse_traces`, cached on the traces' content and the settings."""
    parts = {
        "f": array_digest(f),
        "fneu": array_digest(fneu),
        "period_s": period_s,
        "settings": dataclasses.asdict(settings),
    }
    out = cached(
        cache_dir,
        "signal",
        parts,
        lambda: {"signal": compact(analyse_traces(f, fneu, period_s, settings))},
    )
    return out["signal"]


def _test_one_kind(trace, aligned, frame_times_s, settings: ResponseSettings, rewarded, rng):
    """The shuffle test for rewarded (or unrewarded) outcomes, or None when there are none."""
    is_rewarded = aligned.pulses > 0
    if not (is_rewarded if rewarded else ~is_rewarded).any():
        return None
    n_response = response_frames(settings.response_s, aligned.period_s)
    mask = reward_frame_mask(
        trace.shape[1], aligned.event_index, aligned.pulses, n_response, rewarded
    )
    span = valid_span(frame_times_s)
    shifts = circular_shifts(
        span.stop - span.start, aligned.period_s, settings.n_shifts, settings.min_shift_s, rng
    )
    return responsive_cells(trace[:, span], mask[span], shifts)


def reward_analysis(
    signal: SignalResult,
    frame_times_s: np.ndarray,
    outcomes: pd.DataFrame,
    settings: ResponseSettings,
) -> dict:
    """Outcome-aligned responses of ΔF/F_clear and E_clear, and the responsive-cell tests.

    Returns:
        dict: ``aligned`` (``AlignedWindows`` per trace), ``pooled`` (``{trace: {"snr_pass",
        "all"}: pooled mean ± SEM per pulse count}``), ``n_trials`` per pulse count,
        ``n_dropped`` outcomes without a complete window, and ``tests``
        (``{"reward", "no_reward"}``: the shuffle-test result with ``summary_all`` and
        ``summary_pass`` counts against chance, or None if the run has no such outcome).
    """
    traces = {"dff_clear": signal.dff_clear, "e_clear": signal.e_clear}
    aligned = {
        name: aligned_windows(trace, frame_times_s, outcomes, settings.window_s, settings.phases)
        for name, trace in traces.items()
    }
    groups = {"snr_pass": signal.passes, "all": signal.usable}
    pooled = {
        name: {group: pooled_responses(aligned[name], mask) for group, mask in groups.items()}
        for name in traces
    }
    reference = aligned["dff_clear"]
    tests = {}
    for index, (kind, rewarded) in enumerate(OUTCOME_KINDS.items()):
        rng = np.random.default_rng([settings.seed, index])
        test = _test_one_kind(
            traces[settings.score_trace], reference, frame_times_s, settings, rewarded, rng
        )
        if test is not None:
            for name, group in (("summary_all", signal.usable), ("summary_pass", signal.passes)):
                test[name] = chance_summary(test["responsive"][group], test["score"][group])
        tests[kind] = test
    n_trials = {int(p): int((reference.pulses == p).sum()) for p in np.unique(reference.pulses)}
    return {
        "aligned": aligned,
        "pooled": pooled,
        "n_trials": n_trials,
        "n_dropped": reference.n_dropped,
        "tests": tests,
    }


def check_some_usable(signal: SignalResult) -> None:
    """Raise :class:`NoUsableCells` (with the reasons and their counts) if no ROI is usable."""
    if signal.usable.any():
        return
    n = len(signal.usable)
    reasons = pd.Series(signal.excluded).value_counts().to_dict()
    raise NoUsableCells(f"{n} of {n} cells are excluded: {reasons}.")


def derivative_arrays(run: dict) -> dict:
    """The arrays saved per run: traces, the clock and each row's Cellpose label."""
    signal = run["signal"]
    return {
        "fcorr": signal.fcorr,
        "f0": signal.f0,
        "dff": signal.dff,
        "dff_clear": signal.dff_clear,
        "e_clear": signal.e_clear,
        "time_s": run["frame_times_s"],
        "cellpose_label": run["cellpose_labels"],
        "usable": signal.usable,
    }


def motion_qc(plane_dir, ops: dict, settings: MotionSettings, cache_dir, report) -> dict:
    """Registration metrics of a plane's green binary, cached on the file and the settings.

    Returns:
        dict: ``drift`` (per-PC residual drift), ``i_top``, ``i_bottom``, ``pc_time``,
        ``n_extremes``, ``n_frames_sampled``, ``crop`` (the registration's valid ``(y0, y1, x0,
        x1)`` the images are cut to) and ``offsets`` (the rigid and non-rigid shifts Suite2p
        recorded in ``ops``).
    """
    from mini2p_preprocessing.extraction import registered_binaries

    plane_dir = Path(plane_dir)
    green, _ = registered_binaries(ops, plane_dir)
    ly, lx = int(ops["Ly"]), int(ops["Lx"])
    y0, y1 = ops.get("yrange", [0, ly])
    x0, x1 = ops.get("xrange", [0, lx])
    crop = (int(y0), int(y1), int(x0), int(x1))
    parts = {
        "green": file_signature(green),
        "ops": file_signature(plane_dir / "ops.npy"),
        "settings": dataclasses.asdict(settings),
        "crop": crop,
    }

    def compute():
        indices = sample_frame_indices(int(ops["nframes"]), settings.max_frames)
        report(f"motion: reading {len(indices)} frames and computing {settings.n_pcs} spatial PCs")
        movie = read_frames(green, (int(ops["nframes"]), ly, lx), indices, crop)
        metrics = registration_metrics(
            movie, settings.n_pcs, settings.n_extremes, registration_settings(ops)
        )
        return {**metrics, "n_frames_sampled": len(indices), "crop": crop}

    metrics = cached(cache_dir, "motion", parts, compute)
    return {**metrics, "offsets": registration_offsets(ops)}


# ---- tables ------------------------------------------------------------------------------------


def build_cell_table(signal: SignalResult, tests: dict, cellpose_labels) -> pd.DataFrame:
    """One row per ROI: exclusion, SNR and event counts, and the responsive-cell tests.

    Args:
        signal (SignalResult): The run's signal processing.
        tests (dict): ``reward_analysis(...)["tests"]``; a kind that is None gives NaN columns.
        cellpose_labels: The Cellpose label of each row (``roi`` counts the rows from 0).

    Raises:
        ValueError: If there is not one label per row.
    """
    n = len(signal.usable)
    if len(cellpose_labels) != n:
        raise ValueError(f"{len(cellpose_labels)} Cellpose labels for {n} ROIs.")
    columns = {
        "roi": np.arange(n),
        "cellpose_label": np.asarray(cellpose_labels),
        "usable": signal.usable,
        "excluded": signal.excluded,
        "snr": signal.snr,
        "signal": signal.signal,
        "noise": signal.noise,
        "n_transients": signal.n_transients,
        "n_events": signal.n_events,
        "passes": signal.passes,
    }
    for kind in OUTCOME_KINDS:
        test = tests.get(kind)
        for field in ("score", "null_95th", "p_value"):
            columns[f"{kind}_{field}"] = test[field] if test else np.full(n, np.nan)
        columns[f"{kind}_responsive"] = test["responsive"] if test else np.zeros(n, dtype=bool)
    return pd.DataFrame(columns)


def top_responsive_cells(table: pd.DataFrame, kind: str, n: int) -> np.ndarray:
    """ROI numbers of the *n* highest-scoring cells that pass the SNR threshold and respond.

    Args:
        table (pd.DataFrame): :func:`build_cell_table` output.
        kind (str): ``"reward"`` or ``"no_reward"``.
        n (int): How many to return at most.
    """
    chosen = table[table["passes"] & table[f"{kind}_responsive"]]
    chosen = chosen.sort_values(f"{kind}_score", ascending=False, kind="stable").head(n)
    return chosen["roi"].to_numpy()


def pool_reward_runs(runs: list[dict], trace: str, group: str) -> dict:
    """Mean ± SEM across the cells of all *runs* of ``trace`` around the outcome, per pulse count.

    Each cell contributes its own trial-averaged window, so a run weighs by its cell count.

    Args:
        runs (list[dict]): Results of :func:`analyse_run`.
        trace (str): ``"dff_clear"`` or ``"e_clear"``.
        group (str): ``"snr_pass"`` (cells above the SNR threshold) or ``"all"`` (every cell
            with a defined ΔF/F).

    Raises:
        ValueError: If the runs' windows or frame periods differ, or *group* is unknown.
    """
    if group not in ("snr_pass", "all"):
        raise ValueError(f"group must be 'snr_pass' or 'all', not {group!r}.")
    aligned = [r["reward"]["aligned"][trace] for r in runs]
    check_same_window([{"offsets": a.offsets, "period_s": a.period_s} for a in aligned])
    per_run = []
    for run, windows in zip(runs, aligned, strict=True):
        mask = run["signal"].passes if group == "snr_pass" else run["signal"].usable
        if mask.any():
            per_run.append(
                mean_by_category(windows.windows[mask], windows.pulses, PULSE_CATEGORIES)
            )
    return pooled_mean_sem(per_run, PULSE_CATEGORIES)


# ---- a run and a folder of runs ----------------------------------------------------------------


def analyse_run(
    plane_dir,
    run_dir,
    variant: str,
    *,
    settings: Zong2022Settings,
    responses: ResponseSettings,
    motion: MotionSettings,
    cache_dir,
    load_outcomes: Callable = load_outcome_events,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Extract, process and test one run.

    Args:
        plane_dir: The variant's ``suite2p/plane0`` (registered binaries and ``ops.npy``).
        run_dir: The run's derivatives folder (frame times, and the Cellpose segmentation under
            ``suite2p/plane0``).
        variant (str): ``"without_destriping"`` or ``"with_destriping"``.
        settings (Zong2022Settings): Signal-processing parameters.
        responses (ResponseSettings): Alignment and shuffle-test parameters.
        motion (MotionSettings): Registration-metric parameters.
        cache_dir: Root of the disposable cache.
        load_outcomes (Callable): Reads a behaviour file's outcome table (for tests).
        progress (Callable[[str], None] | None): Receives a line at the start of each stage.

    Returns:
        dict: ``signal`` (``SignalResult``), ``table`` (per-ROI), ``reward`` (see
        :func:`reward_analysis`), ``motion`` (see :func:`motion_qc`), ``labels`` (renumbered
        1..n by row), ``cellpose_labels`` (each row's Cellpose label), ``alignment``,
        ``mean_image``, ``frame_times_s``, ``period_s``, ``n_frames`` and ``outcomes``.

    Raises:
        RunLeftOut: (a subclass) For a known reason the run cannot be analysed: a missing input,
            a refused mask alignment, or no usable cell.
        RunMismatchError: If the traces and the frame clock are from different acquisitions.
    """
    report = report_to(progress)
    frames, outcomes = load_run_inputs(run_dir, variant, load_outcomes)
    extracted = extract_run(plane_dir, run_dir, cache_dir, report)
    n_frames = extracted["f"].shape[1]
    if n_frames != len(frames.time_s):
        raise RunMismatchError(
            f"{plane_dir} has {n_frames} frames but there are {len(frames.time_s)} frame times; "
            "the traces and the frame clock are not from the same acquisition."
        )
    report("ΔF/F: baseline, transients, deconvolution and SNR (cached)")
    signal = signal_cached(extracted["f"], extracted["fneu"], frames.period_s, settings, cache_dir)
    check_some_usable(signal)
    report("responses: outcome-aligned averages and the circular-shift test")
    reward = reward_analysis(signal, frames.time_s, outcomes, responses)
    qc = motion_qc(plane_dir, extracted["ops"], motion, cache_dir, report)
    return {
        "signal": signal,
        "table": build_cell_table(signal, reward["tests"], extracted["cellpose_labels"]),
        "reward": reward,
        "motion": qc,
        "labels": extracted["labels"],
        "cellpose_labels": extracted["cellpose_labels"],
        "alignment": extracted["alignment"],
        "mean_image": np.asarray(extracted["ops"]["meanImg"], dtype=float),
        "frame_times_s": frames.time_s,
        "period_s": frames.period_s,
        "n_frames": n_frames,
        "outcomes": outcomes,
    }


def analyse_runs(
    subject_dir,
    first_session: int,
    last_session: int,
    variant: str,
    runs,
    *,
    report: Callable[[str], None] = print,
    **kwargs,
) -> tuple[list[dict], list[tuple[str, str, str]]]:
    """:func:`analyse_run` for every run of *variant* in the session range.

    Runs that cannot be analysed for a known reason (a missing input, a refused mask alignment,
    no usable cell, traces from another acquisition, an unreadable behaviour file) are listed
    with the reason; anything else, including a stray ``FileNotFoundError``, is a bug or damaged
    data and stops the loop.

    Args:
        subject_dir: ``derivatives/<cohort>/<subject>``.
        first_session (int): First session number.
        last_session (int): Last session number.
        variant (str): ``"without_destriping"`` or ``"with_destriping"``.
        runs: Run folder names to keep (``("run-02",)``), or None for every run.
        report (Callable[[str], None]): Receives progress lines. Default ``print``.
        **kwargs: Passed to :func:`analyse_run` (``settings``, ``responses``, ``motion``,
            ``cache_dir``, ``load_outcomes``).

    Returns:
        tuple: One result per analysed run (``analyse_run`` output plus ``session`` and ``run``),
        and ``(session, run, reason)`` for each run left out.
    """
    from mini2p_preprocessing.progress import ProgressLog

    found = variant_runs(subject_dir, first_session, last_session, variant, runs)
    included, skipped = [], []
    log = ProgressLog(f"Zong et al. 2022 pipeline ({variant})", len(found), report, 30)
    for session, run_dir, plane_dir in found:
        report(f"{session} {run_dir.name}")
        try:
            result = analyse_run(plane_dir, run_dir, variant, progress=report, **kwargs)
        except (RunLeftOut, RunMismatchError, OutcomeParseError) as error:
            skipped.append((session, run_dir.name, str(error)))
            log.step_done(succeeded=False)
            continue
        included.append({"session": session, "run": run_dir.name, "run_dir": run_dir, **result})
        log.step_done()
    return included, skipped
