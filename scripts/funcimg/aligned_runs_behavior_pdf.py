"""A PDF of every aligned imaging run: its mean images over its behavior file's trials.

Usage (from the repo root, where the paths in the listing exist, e.g. on the cluster):
    uv run --extra funcimg python scripts/funcimg/aligned_runs_behavior_pdf.py \
        ALIGNED_RUNS_CSV DERIVATIVES_SUBJECT_DIR OUTPUT_PDF \
        [--variant with_distortion_correction/with_destriping] \
        [--contrast 1 99.8 1] [--overwrite]

``ALIGNED_RUNS_CSV`` is the listing ``list_aligned_runs.py`` wrote (the one mini2p's
``[batch] aligned_runs_csv`` names): it pairs each TIFF with its behavior file and numbers the
runs. For each aligned run whose ``run-XX/<variant>/`` registration has a mean image there is a
page: the green (``meanImg``) and red (``meanImg_chan2``) mean images at the fixed contrast
(black and white percentiles, gamma), and under them the single-session plot's top panel (each
tower's reward magnitude, the choices and the reversals) for that run's behavior file. Runs
without a usable mean image, and pages without behavior, are listed at the end and in
``OUTPUT.caption.txt``, with the reason.
"""

import argparse
import logging
from pathlib import Path

from src.funcimg_analysis.run_behavior_pdf import (
    DEFAULT_CONTRAST,
    Contrast,
    write_run_behavior_pdf,
)


def main(argv=None) -> None:
    """Write the PDF for the arguments (see the module docstring) and list the runs left out."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("aligned_runs_csv", type=Path)
    parser.add_argument("derivatives_subject_dir", type=Path)
    parser.add_argument("output_pdf", type=Path)
    parser.add_argument(
        "--variant", default="with_distortion_correction/with_destriping",
        help="run-XX/ subfolder whose registration to show (default: %(default)s)",
    )  # fmt: skip
    parser.add_argument(
        "--contrast", type=float, nargs=3, metavar=("BLACK", "WHITE", "GAMMA"),
        default=(DEFAULT_CONTRAST.black_percentile, DEFAULT_CONTRAST.white_percentile,
                 DEFAULT_CONTRAST.gamma),
        help="black and white percentiles and gamma (default: %(default)s)",
    )  # fmt: skip
    parser.add_argument("--overwrite", action="store_true", help="replace an existing PDF")
    args = parser.parse_args(argv)
    if args.output_pdf.suffix != ".pdf":
        parser.error(f"OUTPUT_PDF must end in .pdf, not {args.output_pdf.name}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        outcomes = write_run_behavior_pdf(
            args.aligned_runs_csv, args.derivatives_subject_dir, args.variant, args.output_pdf,
            Contrast(*args.contrast), overwrite=args.overwrite,
        )  # fmt: skip
    except (FileExistsError, ValueError) as refused:
        parser.error(str(refused))
    print(f"Wrote {args.output_pdf}: {sum(o.page for o in outcomes)} pages")
    for outcome in outcomes:
        if outcome.problem:
            what = "no behavior on" if outcome.page else "left out"
            print(f"  {what} {outcome.run.session} {outcome.run.run}: {outcome.problem}")


if __name__ == "__main__":
    main()
