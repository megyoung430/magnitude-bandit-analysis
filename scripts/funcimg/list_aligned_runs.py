"""List which imaging TIFFs of a mouse can be put on pyControl time, for mini2p's batch.

Usage (from the repo root, on the machine the batch will run on):
    uv run --extra funcimg python scripts/funcimg/list_aligned_runs.py \
        RAW_SUBJECT_DIR OUTPUT_CSV [--channels 2] [--overwrite]

For every numbered session (``ses-<number>_...``) with a ``funcimg/`` folder, every TIFF
(task, pre-task or test, whatever its name) is paired with its DAQ file by acquisition
order and number of frames, and the DAQ file with its behavior file by rsync pulses
(:func:`src.funcimg_import.runs.session_run_alignments`, which the frame-times step of
``reward_responses.ipynb`` also uses). The TIFFs that reach a behavior file are numbered
``run-01``, ``run-02``, ... per session. The CSV has one row per TIFF: ``run``, ``aligned``,
the ``reason`` when it is not, and the evidence for the pairing (start time, frame counts).
Point mini2p's ``[batch] aligned_runs_csv`` at it: the batch then processes the aligned
TIFFs into those run folders. Its paths are absolute, so run this where the batch will see
the raw data at the same paths (on the cluster, for a cluster batch).
"""

import argparse
from pathlib import Path

from src.funcimg_import.runs import alignment_table, subject_run_alignments


def main(argv=None) -> None:
    """Write the CSV for the arguments (see the module docstring) and list the TIFFs left out."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw_subject_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--channels", type=int, default=2, help="channels in each TIFF")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing CSV")
    args = parser.parse_args(argv)
    if args.output_csv.exists() and not args.overwrite:
        raise SystemExit(f"{args.output_csv} exists; pass --overwrite to replace it.")
    table = alignment_table(subject_run_alignments(args.raw_subject_dir, args.channels))
    if table.empty:
        raise SystemExit(f"No TIFF in any numbered session of {args.raw_subject_dir}.")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False)
    print(f"{int(table['aligned'].sum())} of {len(table)} TIFFs aligned; wrote {args.output_csv}")
    for _, row in table[~table["aligned"]].iterrows():
        name = Path(row["tiff"]).name
        print(f"  not aligned: {row['session']} {name}: {row['reason']}")


if __name__ == "__main__":
    main()
