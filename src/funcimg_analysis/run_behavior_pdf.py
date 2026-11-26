"""A review PDF of every aligned imaging run: its mean images over its behavior.

Each page shows one run's registered green and red mean images (Suite2p ``meanImg`` and
``meanImg_chan2``, at a fixed contrast, scaled as mini2p's exports scale them), and under
them the top panel of the single-session plot (:func:`plot_magnitudes_and_choices`: each
tower's reward magnitude, the choices and the reversals) for the run's behavior file.

Which behavior file goes with which TIFF comes from the aligned-runs listing
(``scripts/funcimg/list_aligned_runs.py``), the pairing the batch preprocessing and the
frame-times step also use. The behavior file is imported on its own, so the trials drawn are
that recording's; a reversal logged between it and an earlier recording of the session is not
marked at its first trial, as it would be in the whole session's plot. A run without a usable
mean image gets no page; it is listed instead, with the reason.

Layers: :func:`load_mean_images` and :func:`load_behavior` read, :func:`draw_run_page` draws
one page, :func:`write_run_behavior_pdf` loads, draws and saves one run at a time.
"""

from __future__ import annotations

import logging
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from mini2p_preprocessing.contrast import scale_contrast
from mini2p_preprocessing.exports import Contrast

import src.behavior_visualization.common.plot_style  # noqa: F401  # applies rcParams
from src.behavior_analysis.task_statistics.single_session import reversal_event_indices
from src.behavior_import.extract_trials import extract_trials
from src.behavior_import.import_data import import_data
from src.behavior_visualization.task_statistics.plot_single_session import (
    plot_magnitudes_and_choices,
)
from src.funcimg_import.planes import registered_plane

log = logging.getLogger(__name__)
LISTING_COLUMNS = ("session", "run", "tiff", "tsv", "aligned")
PAGE_SIZE = (11.69, 8.27)  # A4 landscape
# The display contrast of the imaging review figures: 1st-99.8th percentile, gamma 1
DEFAULT_CONTRAST = Contrast(black_percentile=1.0, white_percentile=99.8, gamma=1.0)
# What reading a registration's ops.npy can raise for a broken or unreadable file
UNREADABLE_OPS = (OSError, EOFError, pickle.UnpicklingError, ValueError)


@dataclass(frozen=True)
class ListedRun:
    """One aligned run of the listing."""

    session: str  # the session folder, e.g. ses-220_date-20260301
    run: str  # run-01, ...
    tiff: Path
    tsv: Path


@dataclass(frozen=True)
class RunPage:
    """What one page shows: the run, its mean images and its behavior (or why not)."""

    run: ListedRun
    green: np.ndarray
    red: np.ndarray
    behavior: dict | None  # the behavior file's session dict (extract_trials)
    behavior_problem: str  # why there is no behavior; "" when there is


@dataclass(frozen=True)
class RunOutcome:
    """What became of one aligned run: a page (with or without behavior) or none."""

    run: ListedRun
    page: bool
    problem: str  # why there is no page, or no behavior on it; "" when neither


class NoMeanImage(Exception):
    """A run whose registration gives no usable mean image: it gets no page."""


# ---- I/O ---------------------------------------------------------------------------------


def load_listing(path: Path) -> list[ListedRun]:
    """The aligned runs of an aligned-runs CSV, in its order.

    Raises:
        ValueError: If a column is missing, ``aligned`` is not true or false, an aligned row
            has no run or behavior file, or two aligned rows share a session and run.
    """
    table = pd.read_csv(path, dtype={"run": str, "tsv": str}, keep_default_na=False)
    missing = [c for c in LISTING_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"{path} lacks the columns {missing} of list_aligned_runs.py")
    flags = table["aligned"].astype(str).str.strip().str.lower()
    unknown = sorted(set(flags) - {"true", "false"})
    if unknown:
        raise ValueError(f"{path}: 'aligned' must be True or False, not {unknown}")
    aligned = table[flags == "true"]
    incomplete = aligned[(aligned["run"] == "") | (aligned["tsv"] == "")]
    if not incomplete.empty:
        raise ValueError(
            f"{path}: aligned rows without a run or behavior file: {incomplete['tiff'].tolist()}"
        )
    duplicated = aligned[aligned.duplicated(["session", "run"], keep=False)]
    if not duplicated.empty:
        raise ValueError(f"{path} lists these runs twice: {duplicated[['session', 'run']].values}")
    return [
        ListedRun(row.session, row.run, Path(row.tiff), Path(row.tsv))
        for row in aligned.itertuples()
    ]


def check_image(image: np.ndarray, name: str, contrast: Contrast) -> np.ndarray:
    """*image* if it can be shown at *contrast* (2-D, some finite pixels, a black level below
    its white level).

    Raises:
        NoMeanImage: Otherwise.
    """
    image = np.asarray(image)
    if image.ndim != 2 or not np.isfinite(image).any():
        raise NoMeanImage(f"{name} is not a 2-D image with finite pixels (shape {image.shape})")
    try:
        scale_contrast(image, contrast.black_percentile, contrast.white_percentile, contrast.gamma)
    except ValueError as flat:  # e.g. a constant image: black and white levels are equal
        raise NoMeanImage(f"{name} cannot be shown: {flat}") from flat
    return image


def load_mean_images(variant_dir: Path, contrast: Contrast) -> tuple[np.ndarray, np.ndarray]:
    """The green (``meanImg``) and red (``meanImg_chan2``) mean images of a run variant's
    registration (:func:`registered_plane`), checked with :func:`check_image`.

    Raises:
        NoMeanImage: If the variant has no registration (or two, aligned on each channel),
            its ``ops.npy`` cannot be read or lacks a mean image, or one cannot be shown.
    """
    try:
        plane = registered_plane(variant_dir)
    except ValueError as ambiguous:  # both channels registered, or an unreadable record
        raise NoMeanImage(f"registration not found: {ambiguous}") from ambiguous
    if plane is None:
        raise NoMeanImage(f"no registration in {variant_dir}")
    try:
        ops = np.load(plane / "ops.npy", allow_pickle=True).item()
    except UNREADABLE_OPS as broken:
        raise NoMeanImage(f"cannot read {plane / 'ops.npy'}: {broken!r}") from broken
    missing = [key for key in ("meanImg", "meanImg_chan2") if key not in ops]
    if missing:
        raise NoMeanImage(f"{plane / 'ops.npy'} has no {missing}")
    return (
        check_image(ops["meanImg"], "meanImg", contrast),
        check_image(ops["meanImg_chan2"], "meanImg_chan2", contrast),
    )


def load_behavior(tsv: Path) -> dict:
    """The session dict of one behavior file on its own (``import_data`` then
    ``extract_trials``), so its trials are that recording's.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the import gives no session (the file cannot be read, or is not in a
            ``sub-<N>_id-<mouse>/ses-<N>_date-<YYYYMMDD>`` folder) or no bandit trials; the
            import's own warnings about the file are part of the message.
    """
    tsv = Path(tsv)
    if not tsv.is_file():
        raise FileNotFoundError(f"no behavior file {tsv}")
    with warnings.catch_warnings(record=True) as caught:  # all about this one file
        warnings.simplefilter("always")
        data = extract_trials(import_data(tsv.parent, paths=[tsv]))
    notes = list(dict.fromkeys(str(w.message) for w in caught))
    for note in notes:
        log.info("%s: %s", tsv.name, note)
    because = f": {'; '.join(notes)}" if notes else ""
    sessions = [session for subject in data.values() for session in subject.values()]
    if len(sessions) != 1:
        raise ValueError(f"{tsv.name} gives {len(sessions)} sessions{because}")
    session = sessions[0]
    if not session.get("trial_info") or not session.get("reward_magnitudes_by_tower"):
        raise ValueError(f"{tsv.name} has no bandit trials{because}")
    return session


def load_run_page(
    run: ListedRun, derivatives_subject_dir: Path, variant: str, contrast: Contrast
) -> RunPage:
    """The page of one run. A behavior file that cannot be read is shown on the page, so the
    images are still seen (and logged).

    Raises:
        NoMeanImage: If the run's registration gives no usable mean image (no page).
    """
    variant_dir = Path(derivatives_subject_dir) / run.session / "funcimg" / run.run / variant
    green, red = load_mean_images(variant_dir, contrast)
    try:
        return RunPage(run, green, red, load_behavior(run.tsv), "")
    except (OSError, ValueError) as problem:
        log.warning("%s %s: no behavior: %s", run.session, run.run, problem)
        return RunPage(run, green, red, None, str(problem))


# ---- drawing -----------------------------------------------------------------------------


def _draw_image(ax, image: np.ndarray, title: str, contrast: Contrast) -> None:
    """*image* in gray at *contrast* (mini2p's ``scale_contrast``), without axes."""
    scaled, _, _ = scale_contrast(
        image, contrast.black_percentile, contrast.white_percentile, contrast.gamma
    )
    ax.imshow(scaled, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=12)
    ax.axis("off")


def _no_behavior(ax, why: str) -> None:
    """Replace the behavior panel with why it has no behavior."""
    ax.clear()
    ax.axis("off")
    ax.text(0.5, 0.5, f"No Behavior\n{why}", ha="center", va="center", wrap=True,
            transform=ax.transAxes, fontsize=10, color="0.3")  # fmt: skip


def _draw_behavior(ax, page: RunPage) -> str:
    """The single-session top panel for the page's behavior file, its legend outside the
    data; or why there is none, which is also returned ("" when drawn)."""
    if page.behavior is None:
        _no_behavior(ax, page.behavior_problem)
        return page.behavior_problem
    session = page.behavior
    mags_by, choices_by = session["reward_magnitudes_by_tower"], session["choices_by_tower"]
    towers = list(mags_by)
    n_trials = len(mags_by[towers[0]])
    try:
        plot_magnitudes_and_choices(
            mags_by, choices_by, towers, reversal_event_indices(session, n_trials),
            title=f"Reward Magnitudes & Choices: {page.run.tsv.name}", ax=ax,
        )  # fmt: skip
    except (KeyError, ValueError) as problem:  # trials the panel cannot draw
        log.warning("%s %s: behavior not drawn: %r", page.run.session, page.run.run, problem)
        why = f"cannot be drawn: {problem!r}"
        _no_behavior(ax, why)
        return why
    ax.set_xlabel("Trial")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return ""


def draw_run_page(page: RunPage, contrast: Contrast) -> tuple[Figure, str]:
    """One run's page: green and red mean images above the reward magnitudes and choices of
    its behavior file; and why the behavior is missing ("" when it is drawn)."""
    fig = plt.figure(figsize=PAGE_SIZE, facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=(2.2, 1), left=0.05, right=0.86, bottom=0.08,
                            top=0.88, hspace=0.25, wspace=0.05)  # fmt: skip
    run = page.run
    fig.suptitle(f"{run.session} | {run.run}", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.925, f"{run.tiff.name}  |  {run.tsv.name}  |  {contrast.label()}",
             ha="center", fontsize=9, color="0.3")  # fmt: skip
    _draw_image(fig.add_subplot(grid[0, 0]), page.green, "Green Functional (meanImg)", contrast)
    _draw_image(fig.add_subplot(grid[0, 1]), page.red, "Red Structural (meanImg_chan2)", contrast)
    behavior_ax = fig.add_subplot(grid[1, :], label="behavior")
    return fig, _draw_behavior(behavior_ax, page)


def caption_text(
    outcomes: list[RunOutcome], variant: str, contrast: Contrast, listing: Path
) -> str:
    """The caption saved beside the PDF."""

    def listed(selected) -> str:
        return "".join(f"\n  {o.run.session} {o.run.run}: {o.problem}" for o in selected)

    left_out = listed(o for o in outcomes if not o.page)
    no_behavior = listed(o for o in outcomes if o.page and o.problem)
    return (
        f"One page per aligned imaging run with a usable mean image "
        f"({sum(o.page for o in outcomes)} pages), in the order of {listing}. Top: the run's "
        f"registered green (Suite2p meanImg) and red (meanImg_chan2) mean images from "
        f"run-XX/{variant}/, contrast {contrast.label()} per image. Bottom: the top panel of "
        "the single-session plot for the run's behavior file (as the listing pairs it with the "
        "TIFF): each tower's reward magnitude per trial (lines), the magnitude chosen (black "
        "dots) and the reversals (vertical lines), over that recording's trials only, imported "
        "on its own: a reversal logged between it and an earlier recording of the session is "
        "not marked at its first trial.\n"
        + (f"Aligned runs left out:{left_out}\n" if left_out else "")
        + (f"Pages without behavior:{no_behavior}\n" if no_behavior else "")
    )


def write_run_behavior_pdf(
    listing: Path,
    derivatives_subject_dir: Path,
    variant: str,
    output: Path,
    contrast: Contrast = DEFAULT_CONTRAST,
    overwrite: bool = False,
) -> list[RunOutcome]:
    """Write the PDF (and ``<name>.caption.txt``) for every aligned run of *listing* with a
    usable mean image, loading, drawing and saving one run at a time, and return what became
    of each run. The PDF is written under a temporary name and renamed when complete.

    Raises:
        FileExistsError: If *output* exists and *overwrite* is False.
        ValueError: If the listing has no aligned run, or no run has a usable mean image (or
            see :func:`load_listing`).
    """
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass overwrite to replace it")
    runs = load_listing(listing)
    if not runs:
        raise ValueError(f"{listing} has no aligned run")
    partial = output.with_name(f".{output.name}.partial")
    outcomes: list[RunOutcome] = []
    try:
        with PdfPages(partial) as pdf:
            pdf.infodict()["Title"] = f"Aligned runs with behavior: {Path(listing).name}"
            for run in runs:
                try:
                    page = load_run_page(run, derivatives_subject_dir, variant, contrast)
                except NoMeanImage as why:
                    log.warning("%s %s: no page: %s", run.session, run.run, why)
                    outcomes.append(RunOutcome(run, False, str(why)))
                    continue
                fig, behavior_problem = draw_run_page(page, contrast)
                number = sum(o.page for o in outcomes) + 1
                try:
                    fig.text(0.5, 0.015, f"Page {number}", ha="center", fontsize=8, color="0.45")
                    pdf.savefig(fig, facecolor="white")
                finally:
                    plt.close(fig)
                outcomes.append(RunOutcome(run, True, behavior_problem))
                log.info("Page %d: %s %s", number, run.session, run.run)
        if not any(o.page for o in outcomes):
            raise ValueError(
                f"No aligned run of {listing} has a usable mean image in run-XX/{variant}/"
            )
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)
    output.with_suffix(".caption.txt").write_text(
        caption_text(outcomes, variant, contrast, Path(listing))
    )
    return outcomes
