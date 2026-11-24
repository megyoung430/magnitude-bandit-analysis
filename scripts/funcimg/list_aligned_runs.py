"""List which imaging TIFFs of a mouse can be put on pyControl time, for mini2p's batch.

Usage (from the repo root):
    uv run --extra funcimg python scripts/funcimg/list_aligned_runs.py \
        RAW_SUBJECT_DIR SOURCE_SUBSTRING OUTPUT_CSV [--overwrite]

For every numbered session (``ses-<number>_...``) with a ``funcimg/`` folder, each TIFF whose
name holds SOURCE_SUBSTRING (e.g. ``71rtask``) is paired with its DAQ file and, by rsync
pulses, its behavior file (:func:`src.funcimg_import.runs.session_run_alignments`, which the
frame-times step of ``reward_responses.ipynb`` also uses). The CSV has one row per TIFF, with
``aligned`` and, when it is False, the ``reason``. Point mini2p's ``[batch]
aligned_runs_csv`` at it to preprocess only the aligned TIFFs. Its paths are absolute, so run
this where the batch will see the raw data at the same paths (on the cluster, for a cluster
batch).
"""

import argparse
from pathlib import Path

from src.funcimg_import.runs import alignment_table, subject_run_alignments


def main(argv=None) -> None:
    """Write the CSV for the arguments (see the module docstring) and list the TIFFs left out."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw_subject_dir", type=Path)
    parser.add_argument("source_substring")
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing CSV")
    args = parser.parse_args(argv)
    if args.output_csv.exists() and not args.overwrite:
        raise SystemExit(f"{args.output_csv} exists; pass --overwrite to replace it.")
    table = alignment_table(subject_run_alignments(args.raw_subject_dir, args.source_substring))
    if table.empty:
        raise SystemExit(
            f"No TIFF named *{args.source_substring}* in any numbered session of "
            f"{args.raw_subject_dir}."
        )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False)
    print(f"{int(table['aligned'].sum())} of {len(table)} TIFFs aligned; wrote {args.output_csv}")
    for _, row in table[~table["aligned"]].iterrows():
        name = Path(row["tiff"]).name
        print(f"  not aligned: {row['session']} {row['run']} {name}: {row['reason']}")


if __name__ == "__main__":
    main()
