"""Reward-responsive cells of every run of one destriping variant (I/O around the pure analysis).

Per run, the green ΔF/F0 of Suite2p and the background-corrected green ΔF/F0 (from
:mod:`src.funcimg_analysis.collect_residual`) are ranked by their reward response with the
cross-validation of :mod:`src.funcimg_analysis.reward_cells`. A run that cannot be analysed is
listed with the reason, never dropped silently.
"""

import math
from collections.abc import Callable, Collection

import numpy as np

from src.behavior_analysis.outcome_events import load_outcome_events
from src.funcimg_analysis.collect import collect_each_run, variant_runs
from src.funcimg_analysis.collect_shared import run_seed
from src.funcimg_analysis.event_frames import frame_period_s
from src.funcimg_analysis.reward_cells import (
    CONTRASTS,
    RANK_ON,
    TooFewTrialsError,
    analyse_reward_cells,
)
from src.funcimg_analysis.run_responses import plane_views
from src.funcimg_analysis.shared_signal_run import (
    SharedSignalUnavailableError,
    TooShortForSharedSignalError,
)

SUITE2P = "dF/F0"
BG_CORRECTED = "bg-corrected dF/F0"
MEASURES = (SUITE2P, BG_CORRECTED)


class MissingResidualError(Exception):
    """A run has no background-corrected ΔF/F0 (it was left out of that analysis)."""


def check_measures(measures: tuple[str, ...]) -> None:
    """Raise unless *measures* is a non-empty tuple of names in ``MEASURES``."""
    if isinstance(measures, str) or not measures or any(m not in MEASURES for m in measures):
        raise ValueError(f"measures must be a non-empty tuple of {MEASURES}, not {measures!r}.")


def collect_reward_cells(
    subject_dir,
    first_session: int,
    last_session: int,
    variant: str,
    load_outcomes: Callable = load_outcome_events,
    *,
    phases: tuple[str, ...],
    plot_window_s: tuple[float, float],
    baseline_s: tuple[float, float],
    response_s: tuple[float, float],
    contrast: str,
    rank_on: str,
    n_shifts: int,
    min_shift_s: float,
    measures: tuple[str, ...] = MEASURES,
    residual_runs: list[dict] | None = None,
    runs: Collection[str] | None = None,
    seed: int = 0,
) -> tuple[list[dict], list[tuple[str, str, str]]]:
    """Rank each run's green cells by reward response, for Suite2p's and the corrected ΔF/F0.

    Args:
        subject_dir (str | Path): ``derivatives/<cohort>/<subject>``.
        first_session (int): First session number to include.
        last_session (int): Last session number to include.
        variant (str): ``"without_destriping"`` or ``"with_destriping"``.
        load_outcomes (Callable): Reads a behaviour file's outcome table (for tests).
        phases (tuple[str, ...]): Trial phases to include.
        plot_window_s (tuple[float, float]): Window of the plotted responses (s).
        baseline_s (tuple[float, float]): Baseline range of a trial (s).
        response_s (tuple[float, float]): Response range of a trial (s).
        contrast (str): ``"rewarded"`` or ``"4_vs_0"``.
        rank_on (str): ``"split_half"`` (cross-validated) or ``"all"`` (biased, for exploration).
        n_shifts (int): Circular shifts of the outcomes for the null.
        min_shift_s (float): Smallest circular shift (s).
        measures (tuple[str, ...]): ``"dF/F0"`` (Suite2p's) and/or ``"bg-corrected dF/F0"``.
        residual_runs (list[dict] | None): ``collect_residual_dff`` results, needed for
            ``"bg-corrected dF/F0"``. A run not among them is left out.
        runs (Collection[str] | None): Run folder names to include. Default None, every run.
        seed (int): Seeds the split and the shifts; each run draws its own from the seed and
            its name.

    Returns:
        tuple: One result per run (``analyse_reward_cells`` output plus ``session``, ``run``,
        ``p_cell`` (Suite2p's classifier probability), ``labels``, ``mean_image`` and ``rois``)
        and ``(session, run, reason)`` for each run left out.

    Raises:
        ValueError: If *measures*, *contrast* or *rank_on* is invalid, or the corrected measure
            is asked for without *residual_runs*.
    """
    check_measures(measures)
    if contrast not in CONTRASTS or rank_on not in RANK_ON:
        raise ValueError(f"contrast must be in {CONTRASTS} and rank_on in {RANK_ON}.")
    if BG_CORRECTED in measures and residual_runs is None:
        raise ValueError(f"{BG_CORRECTED!r} needs residual_runs from collect_residual_dff.")
    corrected = {(r["session"], r["run"]): r for r in residual_runs or []}

    def analyse(session, run_dir, plane_dir, frames, outcomes):
        n_frames, period = len(frames.time_s), frame_period_s(frames.time_s)
        if n_frames < 2 * math.ceil(min_shift_s / period):
            raise TooShortForSharedSignalError(
                f"{n_frames} frames are too few for circular shifts of at least {min_shift_s} s."
            )
        views, traces = plane_views(plane_dir)
        sources = {SUITE2P: lambda: views[("green", "dF/F0")]}
        if BG_CORRECTED in measures:
            if (session, run_dir.name) not in corrected:
                raise MissingResidualError("no background-corrected ΔF/F0 for this run.")
            sources[BG_CORRECTED] = lambda: corrected[(session, run_dir.name)]["channels"]["green"][
                "dff"
            ]
        result = analyse_reward_cells(
            {name: sources[name]() for name in measures},
            frames.time_s,
            outcomes,
            phases=phases,
            plot_window_s=plot_window_s,
            baseline_s=baseline_s,
            response_s=response_s,
            contrast=contrast,
            rank_on=rank_on,
            n_shifts=n_shifts,
            min_shift_s=min_shift_s,
            rng=np.random.default_rng(run_seed(seed, session, run_dir.name)),
        )
        return {
            **result,
            "p_cell": np.asarray(traces.iscell, dtype=float)[:, 1],
            "labels": traces.labels,
            "mean_image": np.asarray(traces.ops["meanImg"], dtype=float),
            "rois": np.arange(len(traces.f)),
        }

    found = variant_runs(subject_dir, first_session, last_session, variant, runs)
    return collect_each_run(
        found,
        variant,
        load_outcomes,
        analyse,
        also_skip=(SharedSignalUnavailableError, TooFewTrialsError, MissingResidualError),
    )
