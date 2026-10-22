"""Signal processing of Zong et al. 2022 (Cell 185:1240), "Neuronal activity processing".

Pure functions from neuropil-corrected fluorescence to ΔF/F, significant transients, calcium
events and SNR; nothing here reads files or draws. The steps, in the paper's order:

1. ``Fcorr = Fcell - 0.7 * Fnp``.
2. Running baseline ``F0 = Fs + m`` (Low et al. 2014): ``Fs`` is the 8th percentile of ``Fcorr``
   in a ±15 s window and ``m`` centres ``Fcorr - F0`` on zero at baseline points, the frames whose
   local SD is within the lowest 10 % of the SD range. ``ΔF/F = (Fcorr - F0) / F0``.
3. Significant transients: ``ΔF/F`` above 2 robust local SDs for more than 0.75 s.
4. OASIS non-negative deconvolution with the indicator's decay time; ``ΔF/F`` and the events
   outside significant transients are set to zero (``*_clear``) and the events are scaled so
   their maximum equals the maximum of ``ΔF/F_clear``.
5. SNR = mean 90th percentile of ``ΔF/F`` per transient over the noise of ``ΔF/F`` outside
   transients; cells need SNR > 3.
"""

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

MAD_TO_SD = 1.4826  # scales the median absolute deviation to a standard deviation (normal data)
SIGNAL_PERCENTILE = 90  # of ΔF/F within each significant transient


@dataclass(frozen=True)
class Zong2022Settings:
    """The pipeline's parameters; the defaults are the paper's, except ``tau_s``.

    Attributes:
        neuropil_coefficient (float): ``Fcorr = Fcell - coefficient * Fnp``. Paper: 0.7.
        tau_s (float): Exponential decay time of the indicator for OASIS (s). The paper used
            1.5 s for GCaMP6s; there is no paper value for jGCaMP8s, see the notebook.
        baseline_percentile (float): Percentile of the running baseline ``Fs``. Paper: 8.
        baseline_half_window_s (float): Half width of the baseline and local SD windows. Paper: 15.
        baseline_sd_fraction (float): A frame is a baseline point when its local SD is at most
            ``sd_min + fraction * (sd_max - sd_min)``. Paper: 0.1.
        transient_sd (float): A transient exceeds this many local baseline SDs. Paper: 2.
        transient_min_s (float): ... for longer than this (s). Paper: 0.75.
        min_snr (float): Cells need an SNR above this. Paper: 3.
        min_baseline_s (float): A cell needs at least this much baseline (in frames at the frame
            rate) to centre ``F0`` and to estimate its noise; fewer and the cell is excluded.
        min_events (int | None): Cells also need at least this many calcium events. The paper
            used 100 for its tuning analyses; None applies no event criterion.
    """

    neuropil_coefficient: float = 0.7
    tau_s: float = 0.7
    baseline_percentile: float = 8.0
    baseline_half_window_s: float = 15.0
    baseline_sd_fraction: float = 0.1
    transient_sd: float = 2.0
    transient_min_s: float = 0.75
    min_snr: float = 3.0
    min_baseline_s: float = 4.0
    min_events: int | None = None

    def __post_init__(self):
        checks = {
            "neuropil_coefficient": self.neuropil_coefficient >= 0,
            "tau_s": self.tau_s > 0,
            "baseline_percentile": 0 <= self.baseline_percentile <= 100,
            "baseline_half_window_s": self.baseline_half_window_s > 0,
            "baseline_sd_fraction": 0 < self.baseline_sd_fraction <= 1,
            "transient_sd": self.transient_sd > 0,
            "transient_min_s": self.transient_min_s >= 0,
            "min_snr": self.min_snr >= 0,
            "min_baseline_s": self.min_baseline_s > 0,
            "min_events": self.min_events is None or self.min_events >= 1,
        }
        bad = [name for name, ok in checks.items() if not ok]
        if bad:
            raise ValueError(f"Invalid Zong2022Settings: {bad}; got {self}.")


@dataclass(frozen=True)
class SignalResult:
    """Everything the pipeline computes for one run; rows are ROIs, columns frames.

    Cells that cannot be analysed (``usable`` False; ``excluded`` says why: F0 not positive, or
    too few baseline points) have NaN traces and no SNR; they keep their row so indices stay the
    ROI numbers.
    """

    period_s: float
    settings: Zong2022Settings
    fcorr: np.ndarray
    f0: np.ndarray
    delta_f: np.ndarray  # Fcorr - F0, the input of the deconvolution
    dff: np.ndarray
    baseline_points: np.ndarray
    baseline_sd: np.ndarray  # robust local SD of ΔF/F outside transients, the threshold's unit
    transient_mask: np.ndarray
    events: np.ndarray  # OASIS output, before filtering
    dff_clear: np.ndarray
    e_clear: np.ndarray
    usable: np.ndarray
    excluded: np.ndarray  # "" for usable cells, else the reason
    signal: np.ndarray
    noise: np.ndarray
    snr: np.ndarray
    n_transients: np.ndarray
    n_events: np.ndarray
    passes: np.ndarray


# ---- helpers ----------------------------------------------------------------------------------


def neuropil_correct(f: np.ndarray, fneu: np.ndarray, coefficient: float) -> np.ndarray:
    """``Fcell - coefficient * Fnp``.

    Raises:
        ValueError: If the shapes differ or either array has a NaN or inf (one bad frame would
            spread across a whole baseline window without being noticed).
    """
    f, fneu = np.asarray(f, dtype=float), np.asarray(fneu, dtype=float)
    if f.shape != fneu.shape:
        raise ValueError(f"F and Fneu differ in shape: {f.shape} vs {fneu.shape}.")
    if not (np.isfinite(f).all() and np.isfinite(fneu).all()):
        raise ValueError("Traces contain non-finite values (NaN or inf).")
    return f - coefficient * fneu


def half_window_frames(half_window_s: float, period_s: float) -> int:
    """Half width of a centred window, in whole frames."""
    return int(round(half_window_s / period_s))


def min_transient_frames(min_s: float, period_s: float) -> int:
    """Fewest frames whose duration is strictly longer than *min_s* (the paper says "more than").

    The ratio is rounded to 9 places first so a duration that is a whole number of frames, which
    must be exceeded, is not misread because of floating-point error.
    """
    return int(np.floor(round(min_s / period_s, 9))) + 1


def _check_window(n_frames: int, half: int) -> None:
    if 2 * half + 1 > n_frames:
        raise ValueError(
            f"A ±{half}-frame window ({2 * half + 1} frames) is longer than the "
            f"{n_frames}-frame trace; the recording is too short for this window."
        )


def _window_bounds(n: int, half: int) -> tuple[np.ndarray, np.ndarray]:
    """Start and (exclusive) stop frame of the window centred on each frame, cut at the ends."""
    index = np.arange(n)
    return np.maximum(index - half, 0), np.minimum(index + half + 1, n)


# ---- moving windows (windows are cut at the recording's ends, never padded) ---------------------


def moving_percentile(trace: np.ndarray, half: int, percentile: float) -> np.ndarray:
    """Percentile of ``trace`` in the window ``[i - half, i + half]`` centred on each frame.

    Within *half* frames of either end the window is cut to the frames that exist.
    """
    x = np.asarray(trace, dtype=float)
    n = len(x)
    _check_window(n, half)
    out = np.empty(n)
    out[half : n - half] = np.percentile(sliding_window_view(x, 2 * half + 1), percentile, axis=1)
    for i in (*range(half), *range(n - half, n)):
        out[i] = np.percentile(x[max(0, i - half) : i + half + 1], percentile)
    return out


def moving_std(trace: np.ndarray, half: int) -> np.ndarray:
    """Population SD of ``trace`` in the window centred on each frame, cut at the ends.

    Uses running sums of the centred trace, so it is O(n) rather than O(n * window).
    """
    x = np.asarray(trace, dtype=float)
    n = len(x)
    _check_window(n, half)
    centred = x - x.mean()
    sums = np.r_[0.0, np.cumsum(centred)]
    squares = np.r_[0.0, np.cumsum(centred**2)]
    lo, hi = _window_bounds(n, half)
    count = hi - lo
    mean = (sums[hi] - sums[lo]) / count
    variance = (squares[hi] - squares[lo]) / count - mean**2
    return np.sqrt(np.maximum(variance, 0.0))


# ---- baseline and ΔF/F -----------------------------------------------------------------------


def running_baseline(
    fcorr: np.ndarray, half: int, percentile: float, sd_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """The running baseline ``F0 = Fs + m`` of one trace, and its baseline points.

    ``Fs`` is the running *percentile*. The local SD in the same window picks the baseline
    points: frames at most ``sd_min + sd_fraction * (sd_max - sd_min)`` (``sd_min`` and
    ``sd_max`` over the whole trace). ``m`` is the mean of ``Fcorr - Fs`` over the baseline
    points, which is the constant that centres ``Fcorr - F0`` on zero there. Windows within
    *half* frames of either end are cut, so their SD is estimated from fewer frames and is more
    variable; baseline points are over-represented there.

    Args:
        fcorr (np.ndarray): Neuropil-corrected fluorescence of one cell, ``(n_frames,)``.
        half (int): Half window in frames.
        percentile (float): Percentile of ``Fs``.
        sd_fraction (float): Fraction of the SD range that still counts as baseline.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``F0`` and a boolean per frame, True at baseline points.
    """
    fs = moving_percentile(fcorr, half, percentile)
    sd = moving_std(fcorr, half)
    baseline = sd <= sd.min() + sd_fraction * (sd.max() - sd.min())
    offset = np.mean(np.asarray(fcorr, dtype=float)[baseline] - fs[baseline])
    return fs + offset, baseline


def min_baseline_frames(min_s: float, period_s: float) -> int:
    """Frames in *min_s* seconds, rounded up (at least one)."""
    return max(1, int(np.ceil(round(min_s / period_s, 9))))


def moving_robust_sd(
    x: np.ndarray,
    half: int,
    exclude: np.ndarray | None = None,
    min_points: int = 30,
    stride: int | None = None,
) -> np.ndarray:
    """Robust SD (1.4826 x MAD) of *x* in the window centred on each frame, without excluded frames.

    The MAD is insensitive to the few large values of a transient, and *exclude* removes the
    frames of detected transients altogether. Windows are cut at the ends of the recording. To
    stay fast the SD is evaluated every *stride* frames (default a tenth of the half window) and
    interpolated; the window changes by a few percent between neighbouring frames, so the
    interpolation error is negligible. A window with fewer than *min_points* usable frames takes
    the SD of all usable frames of the trace.

    Raises:
        ValueError: If fewer than *min_points* frames are usable in all.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    keep = np.ones(n, dtype=bool) if exclude is None else ~np.asarray(exclude, dtype=bool)
    if keep.sum() < min_points:
        raise ValueError(
            f"Only {int(keep.sum())} usable frames (need {min_points}) to measure the noise from."
        )
    stride = max(1, half // 10) if stride is None else stride
    centres = np.unique(np.r_[np.arange(0, n, stride), n - 1])
    lo, hi = _window_bounds(n, half)
    global_sd = _mad_sd(x[keep])
    values = np.empty(len(centres))
    for k, centre in enumerate(centres):
        window = x[lo[centre] : hi[centre]][keep[lo[centre] : hi[centre]]]
        values[k] = _mad_sd(window) if len(window) >= min_points else global_sd
    return np.interp(np.arange(n), centres, values)


def _mad_sd(values: np.ndarray) -> float:
    return float(MAD_TO_SD * np.median(np.abs(values - np.median(values))))


def transient_threshold(
    dff: np.ndarray, half: int, n_sd: float, min_frames: int, min_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """The local noise SD and the significant transients of one cell, found in two passes.

    The paper's "local standard deviation of the baseline fluorescence" is read here as the
    robust SD of ``ΔF/F`` over the frames of the ±15 s window that lie outside significant
    transients. The transients are not known beforehand, so they are found with the robust SD
    of all frames in the window (the MAD ignores them), then the SD is measured again without
    the detected transients and the detection repeated.

    Returns:
        tuple[np.ndarray, np.ndarray]: The SD per frame and the transient mask.
    """
    first = moving_robust_sd(dff, half, min_points=min_points)
    detected = significant_transients(dff, first, n_sd, min_frames)
    sd = moving_robust_sd(dff, half, exclude=detected, min_points=min_points)
    return sd, significant_transients(dff, sd, n_sd, min_frames)


# ---- transients and SNR ----------------------------------------------------------------------


def transient_bounds(mask: np.ndarray) -> list[tuple[int, int]]:
    """``(start, stop)`` frames (stop exclusive) of each run of True in *mask*."""
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(int)
    steps = np.diff(padded)
    return [
        (int(a), int(b))
        for a, b in zip(np.flatnonzero(steps == 1), np.flatnonzero(steps == -1), strict=True)
    ]


def significant_transients(
    dff: np.ndarray, local_sd: np.ndarray, n_sd: float, min_frames: int
) -> np.ndarray:
    """Frames inside runs of at least *min_frames* where ``ΔF/F`` exceeds ``n_sd * local_sd``.

    NaN ``ΔF/F`` is never above threshold.
    """
    with np.errstate(invalid="ignore"):
        above = np.asarray(dff, dtype=float) > n_sd * np.asarray(local_sd, dtype=float)
    mask = np.zeros(above.shape, dtype=bool)
    for start, stop in transient_bounds(above):
        if stop - start >= min_frames:
            mask[start:stop] = True
    return mask


def transient_snr(dff: np.ndarray, mask: np.ndarray) -> dict:
    """Signal, noise and SNR of one cell from its significant transients.

    Signal is the mean over transients of the 90th percentile of ``ΔF/F`` within each. Noise is
    the mean absolute difference between consecutive frames of ``ΔF/F``, taken only over pairs
    where both frames lie outside transients (a step into a transient is signal, not noise).

    Returns:
        dict: ``signal``, ``noise``, ``snr`` and ``n_transients``. ``signal`` and ``snr`` are NaN
        without a transient; ``snr`` is NaN when the noise is zero or has no frame pairs.
    """
    x = np.asarray(dff, dtype=float)
    bounds = transient_bounds(mask)
    outside = ~np.asarray(mask, dtype=bool)
    pairs = outside[1:] & outside[:-1]
    signal = (
        float(np.mean([np.percentile(x[a:b], SIGNAL_PERCENTILE) for a, b in bounds]))
        if bounds
        else np.nan
    )
    noise = float(np.mean(np.abs(np.diff(x))[pairs])) if pairs.any() else np.nan
    snr = signal / noise if noise > 0 else np.nan
    return {"signal": signal, "noise": noise, "snr": snr, "n_transients": len(bounds)}


# ---- deconvolution and clear traces ----------------------------------------------------------


def deconvolve(delta_f: np.ndarray, fs_hz: float, tau_s: float, batch_size: int = 500):
    """OASIS non-negative deconvolution (Suite2p's implementation) of baseline-subtracted traces.

    Args:
        delta_f (np.ndarray): ``(n_cells, n_frames)`` fluorescence with its baseline removed
            (OASIS has no baseline term, so a raw trace would be explained by one huge event).
        fs_hz (float): Frame rate.
        tau_s (float): Exponential decay time of the indicator (s).
        batch_size (int): Cells deconvolved per batch.

    Returns:
        np.ndarray: ``(n_cells, n_frames)`` non-negative event amplitudes.
    """
    from suite2p.extraction import oasis

    return oasis(
        F=np.asarray(delta_f, dtype=np.float32), batch_size=batch_size, tau=tau_s, fs=fs_hz
    )


def clear_traces(
    dff: np.ndarray, events: np.ndarray, transient_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``ΔF/F_clear`` and ``E_clear``: zero outside significant transients.

    Each cell's events are scaled so their maximum equals the maximum of its ``ΔF/F_clear``; a
    cell with no event inside a transient keeps zeros.
    """
    dff_clear = np.where(transient_mask, dff, 0.0)
    e_clear = np.where(transient_mask, events, 0.0)
    peak_e = e_clear.max(axis=1, keepdims=True)
    peak_dff = dff_clear.max(axis=1, keepdims=True)
    scale = np.divide(peak_dff, peak_e, out=np.zeros_like(peak_e, dtype=float), where=peak_e > 0)
    return dff_clear, e_clear * scale


def event_counts_by_tau(
    delta_f: np.ndarray, transient_mask: np.ndarray, fs_hz: float, taus: tuple[float, ...]
) -> dict[float, np.ndarray]:
    """Calcium events per cell inside significant transients, for each assumed decay time.

    Cells with a NaN ``delta_f`` (excluded) get zero.
    """
    usable = np.isfinite(delta_f).all(axis=1)
    counts = {}
    for tau in taus:
        events = np.zeros(delta_f.shape, dtype=np.float32)
        events[usable] = deconvolve(delta_f[usable], fs_hz, tau)
        counts[tau] = ((events > 0) & transient_mask).sum(axis=1)
    return counts


# ---- the pipeline ----------------------------------------------------------------------------


def _baseline_traces(fcorr: np.ndarray, half: int, settings: Zong2022Settings):
    pairs = [
        running_baseline(row, half, settings.baseline_percentile, settings.baseline_sd_fraction)
        for row in fcorr
    ]
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def _cell_transients(dff: np.ndarray, half: int, settings: Zong2022Settings, period_s: float):
    sd, mask = transient_threshold(
        dff,
        half,
        settings.transient_sd,
        min_transient_frames(settings.transient_min_s, period_s),
        min_baseline_frames(settings.min_baseline_s, period_s),
    )
    return sd, mask, transient_snr(dff, mask)


def _passes(snr: np.ndarray, n_events: np.ndarray, settings: Zong2022Settings) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        ok = snr > settings.min_snr
    if settings.min_events is not None:
        ok &= n_events >= settings.min_events
    return ok


def _exclusion_reasons(f0: np.ndarray, baseline: np.ndarray, min_points: int) -> np.ndarray:
    """Per cell: "" if analysable, else why not (a non-positive F0 or too little baseline)."""
    return np.where(
        (f0 <= 0).any(axis=1),
        "F0 <= 0",
        np.where(baseline.sum(axis=1) < min_points, "too few baseline points", ""),
    )


def analyse_traces(
    f: np.ndarray, fneu: np.ndarray, period_s: float, settings: Zong2022Settings
) -> SignalResult:
    """Run the paper's signal processing on every ROI of one run.

    Args:
        f (np.ndarray): ``Fcell``, ``(n_cells, n_frames)``.
        fneu (np.ndarray): ``Fnp``, same shape.
        period_s (float): Frame period (s).
        settings (Zong2022Settings): Parameters.

    Returns:
        SignalResult: All intermediate and final traces and the per-cell table. ROIs whose F0
        is not positive at every frame, or that have too few baseline points, cannot be analysed;
        they are marked ``usable`` False with the reason in ``excluded`` and left as NaN rather
        than dropped or raised on.

    Raises:
        ValueError: If the traces are not finite, differ in shape, or are shorter than the
            baseline window.
    """
    fcorr = neuropil_correct(f, fneu, settings.neuropil_coefficient)
    half = half_window_frames(settings.baseline_half_window_s, period_s)
    _check_window(fcorr.shape[1], half)
    f0, baseline = _baseline_traces(fcorr, half, settings)
    excluded = _exclusion_reasons(
        f0, baseline, min_baseline_frames(settings.min_baseline_s, period_s)
    )
    usable = excluded == ""
    nan = np.full(fcorr.shape, np.nan)
    delta_f = np.where(usable[:, None], fcorr - f0, nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        dff = np.where(usable[:, None], delta_f / f0, nan)
    ok = np.flatnonzero(usable)
    per_cell = [_cell_transients(dff[i], half, settings, period_s) for i in ok]
    baseline_sd, mask = nan.copy(), np.zeros(fcorr.shape, dtype=bool)
    stats = {key: np.full(len(fcorr), np.nan) for key in ("signal", "noise", "snr")}
    n_transients = np.zeros(len(fcorr), dtype=int)
    for i, (sd, cell_mask, cell_stats) in zip(ok, per_cell, strict=True):
        baseline_sd[i], mask[i], n_transients[i] = sd, cell_mask, cell_stats["n_transients"]
        for key in stats:
            stats[key][i] = cell_stats[key]
    events, dff_clear, e_clear = nan.copy(), nan.copy(), nan.copy()
    if len(ok):
        events[ok] = deconvolve(delta_f[ok], 1 / period_s, settings.tau_s)
        dff_clear[ok], e_clear[ok] = clear_traces(dff[ok], events[ok], mask[ok])
    n_events = np.where(usable, (np.nan_to_num(e_clear) > 0).sum(axis=1), 0)
    return SignalResult(
        period_s=period_s,
        settings=settings,
        fcorr=fcorr,
        f0=f0,
        delta_f=delta_f,
        dff=dff,
        baseline_points=baseline,
        baseline_sd=baseline_sd,
        transient_mask=mask,
        events=events,
        dff_clear=dff_clear,
        e_clear=e_clear,
        usable=usable,
        excluded=excluded,
        n_transients=n_transients,
        n_events=n_events,
        passes=_passes(stats["snr"], n_events, settings),
        **stats,
    )
