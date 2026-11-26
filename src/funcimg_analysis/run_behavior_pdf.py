"""A review PDF of every aligned imaging run: its mean images over its behavior.

Each page shows one run's registered green and red mean images (Suite2p ``meanImg`` and
``meanImg_chan2``, at a fixed contrast, scaled as mini2p's exports scale them), and under
them the top panel of the single-session plot (:func:`plot_magnitudes_and_choices`: each
tower's reward magnitude, the choices and the reversals) for the run's behavior file.

Which behavior file goes with which TIFF comes from the aligned-runs listing
(``scripts/funcimg/list_aligned_runs.py``), the pairing the batch preprocessing and the
frame-times step also use, so the trials drawn are that recording's. A run whose
registration has no mean image gets no page; it is listed instead.

Layers: :func:`load_run_page` reads (a run's plane and behavior file), :func:`draw_run_page`
draws one page, :func:`write_run_behavior_pdf` writes the PDF.
"""

from __future__ import annotations

import logging
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


# Meg's usual contrast for imaging images
DEFAULT_CONTRAST = Contrast(black_percentile=1.0, white_percentile=99.8, gamma=1.0)


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


# ---- I/O ---------------------------------------------------------------------------------


def load_listing(path: Path) -> list[ListedRun]:
    """The aligned runs of an aligned-runs CSV, in its order.

    Raises:
        ValueError: If a column is missing or two aligned rows share a session and run.
    """
    table = pd.read_csv(path, dtype={"run": str}, keep_default_na=False)
    missing = [c for c in LISTING_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"{path} lacks the columns {missing} of list_aligned_runs.py")
    aligned = table[table["aligned"].astype(str).str.lower() == "true"]
    duplicated = aligned[aligned.duplicated(["session", "run"], keep=False)]
    if not duplicated.empty:
        raise ValueError(f"{path} lists these runs twice: {duplicated[['session', 'run']].values}")
    return [
        ListedRun(row.session, row.run, Path(row.tiff), Path(row.tsv))
        for row in aligned.itertuples()
    ]


def load_mean_images(variant_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """The green (``meanImg``) and red (``meanImg_chan2``) mean images of a run variant's
    registration (:func:`registered_plane`).

    Raises:
        FileNotFoundError: If the variant has no registration.
        KeyError: If its ``ops.npy`` has no red mean image.
    """
    plane = registered_plane(variant_dir)
    if plane is None:
        raise FileNotFoundError(f"no registration in {variant_dir}")
    ops = np.load(plane / "ops.npy", allow_pickle=True).item()
    missing = [key for key in ("meanImg", "meanImg_chan2") if key not in ops]
    if missing:
        raise KeyError(f"{plane / 'ops.npy'} has no {missing}")
    return np.asarray(ops["meanImg"]), np.asarray(ops["meanImg_chan2"])


def subject_dir_of(tsv: Path) -> Path:
    """The ``sub-*`` folder a behavior file sits in.

    Raises:
        ValueError: If none of its folders is one.
    """
    for parent in tsv.parents:
        if parent.name.startswith("sub-"):
            return parent
    raise ValueError(f"{tsv} is not inside a sub-* folder")


def load_behavior(tsv: Path) -> dict:
    """The session dict of one behavior file on its own (``import_data`` then
    ``extract_trials``), so its trials are that recording's.

    Raises:
        ValueError: If the file has no bandit trials, or its import gives more than one
            session (a recording split by a configuration change).
    """
    with warnings.catch_warnings():  # the import's skip warnings concern other files
        warnings.simplefilter("ignore")
        data = extract_trials(import_data(subject_dir_of(tsv), paths=[tsv]))
    sessions = [session for subject in data.values() for session in subject.values()]
    if len(sessions) != 1:
        raise ValueError(f"{tsv.name} gives {len(sessions)} sessions, not one")
    session = sessions[0]
    if not session.get("trial_info") or not session.get("reward_magnitudes_by_tower"):
        raise ValueError(f"{tsv.name} has no bandit trials")
    return session


@dataclass(frozen=True)
class SkippedRun:
    """An aligned run with no page, and why."""

    run: ListedRun
    reason: str


def load_run_page(
    run: ListedRun, derivatives_subject_dir: Path, variant: str
) -> RunPage | SkippedRun:
    """The page of one run, or why it has none (no registration or no mean image in it). A
    behavior file that cannot be read is shown on the page, so the images are still seen;
    both are logged."""
    variant_dir = Path(derivatives_subject_dir) / run.session / "funcimg" / run.run / variant
    try:
        green, red = load_mean_images(variant_dir)
    except (FileNotFoundError, KeyError, ValueError) as problem:
        log.warning("%s %s: no page, no mean images: %s", run.session, run.run, problem)
        return SkippedRun(run, str(problem))
    behavior, behavior_problem = None, ""
    try:
        behavior = load_behavior(run.tsv)
    except (FileNotFoundError, ValueError, KeyError) as problem:
        behavior_problem = str(problem)
        log.warning("%s %s: no behavior: %s", run.session, run.run, problem)
    return RunPage(run, green, red, behavior, behavior_problem)


# ---- drawing -----------------------------------------------------------------------------


def _draw_image(ax, image: np.ndarray, title: str, contrast: Contrast) -> None:
    """*image* in gray at *contrast* (mini2p's ``scale_contrast``), without axes."""
    scaled, _, _ = scale_contrast(
        image, contrast.black_percentile, contrast.white_percentile, contrast.gamma
    )
    ax.imshow(scaled, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=12)
    ax.axis("off")


def _draw_behavior(ax, page: RunPage) -> None:
    """The single-session top panel for the page's behavior file, its legend outside the
    data; or why there is none."""
    if page.behavior is None:
        ax.axis("off")
        ax.text(0.5, 0.5, f"No Behavior\n{page.behavior_problem}", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="0.3")  # fmt: skip
        return
    session = page.behavior
    mags_by, choices_by = session["reward_magnitudes_by_tower"], session["choices_by_tower"]
    towers = list(mags_by)
    n_trials = len(mags_by[towers[0]])
    plot_magnitudes_and_choices(
        mags_by, choices_by, towers, reversal_event_indices(session, n_trials),
        title=f"Reward Magnitudes & Choices: {page.run.tsv.name}", ax=ax,
    )  # fmt: skip
    ax.set_xlabel("Trial")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)


def draw_run_page(page: RunPage, contrast: Contrast) -> Figure:
    """One run's page: green and red mean images above the reward magnitudes and choices of
    its behavior file."""
    fig = plt.figure(figsize=PAGE_SIZE, facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=(2.2, 1), left=0.05, right=0.86, bottom=0.08,
                            top=0.88, hspace=0.25, wspace=0.05)  # fmt: skip
    run = page.run
    fig.suptitle(f"{run.session} | {run.run}", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.925, f"{run.tiff.name}  |  {run.tsv.name}  |  {contrast.label()}",
             ha="center", fontsize=9, color="0.3")  # fmt: skip
    _draw_image(fig.add_subplot(grid[0, 0]), page.green, "Green Functional (meanImg)", contrast)
    _draw_image(fig.add_subplot(grid[0, 1]), page.red, "Red Structural (meanImg_chan2)", contrast)
    _draw_behavior(fig.add_subplot(grid[1, :]), page)
    return fig


def caption_text(
    n_pages: int, skipped: list[SkippedRun], variant: str, contrast: Contrast, listing: Path
) -> str:
    """The caption saved beside the PDF."""
    left_out = "".join(f"\n  {s.run.session} {s.run.run}: {s.reason}" for s in skipped)
    return (
        f"One page per aligned imaging run with a mean image ({n_pages} pages), in the order "
        f"of {listing}. Top: the "
        f"run's registered green (Suite2p meanImg) and red (meanImg_chan2) mean images from "
        f"run-XX/{variant}/, contrast {contrast.label()} per image. Bottom: the top panel of "
        "the single-session plot for the run's behavior file (as the listing pairs it with the "
        "TIFF): each tower's reward magnitude per trial (lines), the magnitude chosen (black "
        "dots) and the reversals (vertical lines), over that recording's trials only.\n"
        + (f"Aligned runs left out, with no mean image:{left_out}\n" if skipped else "")
    )


def write_run_behavior_pdf(
    listing: Path,
    derivatives_subject_dir: Path,
    variant: str,
    output: Path,
    contrast: Contrast = DEFAULT_CONTRAST,
    overwrite: bool = False,
) -> tuple[list[RunPage], list[SkippedRun]]:
    """Write the PDF (and ``<name>.caption.txt``) for every aligned run of *listing* with a
    mean image, and return its pages and the runs left out.

    Raises:
        FileExistsError: If *output* exists and *overwrite* is False.
        ValueError: If the listing has no aligned run, or no run has a mean image (or see
            :func:`load_listing`).
    """
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass overwrite to replace it")
    runs = load_listing(listing)
    if not runs:
        raise ValueError(f"{listing} has no aligned run")
    loaded = [load_run_page(run, derivatives_subject_dir, variant) for run in runs]
    pages = [p for p in loaded if isinstance(p, RunPage)]
    skipped = [p for p in loaded if isinstance(p, SkippedRun)]
    if not pages:
        raise ValueError(f"No aligned run of {listing} has a mean image in run-XX/{variant}/")
    with PdfPages(output) as pdf:
        pdf.infodict()["Title"] = f"Aligned runs with behavior: {Path(listing).name}"
        for number, page in enumerate(pages, start=1):
            fig = draw_run_page(page, contrast)
            fig.text(0.5, 0.015, f"Page {number} of {len(pages)}", ha="center", fontsize=8,
                     color="0.45")  # fmt: skip
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
    output.with_suffix(".caption.txt").write_text(
        caption_text(len(pages), skipped, variant, contrast, Path(listing))
    )
    return pages, skipped
