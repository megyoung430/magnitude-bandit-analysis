"""Compute summary statistics about block structure and lengths.

Block lengths are derived from the trial indices at which reversals occur.
Good, bad, or all reversals can be used as block boundaries.
"""

from collections import defaultdict
from itertools import pairwise

import numpy as np

from src.behavior_analysis.get_total_reversals import get_all_reversal_indices


def _boundary_indices(*, goods, bads, boundary):
    """Sorted reversal trial indices that bound blocks for the chosen *boundary* type."""
    if boundary == "good":
        return sorted(goods)
    if boundary == "bad":
        return sorted(bads)
    if boundary == "all":
        return sorted(set(goods) | set(bads))
    raise ValueError('boundary must be "all", "good", or "bad"')


def _boundaries_per_mouse(subjects_trials, *, all_good_idx, all_bad_idx, boundary):
    """``{subject: reversal indices}`` for subjects with at least one boundary.

    Block 1 is trial 0 up to the first reversal, so a subject needs a single
    reversal to have a block; subjects without one are left out.
    """
    # Known issue: *boundary* is validated inside this loop, so with no subjects an invalid
    # value gets the caller's "No mice..." ValueError instead of the boundary message.
    per_mouse_boundaries = {}
    for mouse in subjects_trials.keys():
        idxs = _boundary_indices(
            goods=all_good_idx.get(mouse, []), bads=all_bad_idx.get(mouse, []), boundary=boundary
        )
        if len(idxs) >= 1:
            per_mouse_boundaries[mouse] = idxs
    return per_mouse_boundaries


def _block_lengths(rev_idxs):
    """``{block number: length}`` for the blocks between 0 and each reversal index."""
    lengths = [stop - start for start, stop in pairwise([0, *rev_idxs])]
    return {block + 1: float(length) for block, length in enumerate(lengths)}


def _pool_by_block(per_mouse_blocklens):
    """``{block number: [lengths]}`` pooled across mice, skipping non-finite lengths."""
    block_to_vals = defaultdict(list)
    for block_lens in per_mouse_blocklens.values():
        for block, length in block_lens.items():
            if np.isfinite(length):
                block_to_vals[block].append(length)
    return block_to_vals


def _standard_error(vals):
    """Standard error of the mean; 0.0 when there are fewer than two values."""
    vals = np.asarray(vals, dtype=float)
    return np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) >= 2 else 0.0


def get_block_lengths(subjects_trials, boundary="all"):
    """Compute per-block trial counts from reversal boundaries across subjects.

    Defines blocks as the intervals between consecutive reversal events.
    Block 1 spans trials 0 to the first reversal, block 2 spans the first
    to the second reversal, and so on.

    Args:
        subjects_trials: Nested dict ``{subject: {session_key: session_dict}}``.
        boundary: Which reversal type to use as block boundaries.  One of:

            - ``"all"`` (default) — union of good and bad reversal indices.
            - ``"good"`` — good reversal indices only.
            - ``"bad"`` — bad reversal indices only.

    Returns:
        Dict with keys:

        - ``"boundary"`` (str): The *boundary* argument used.
        - ``"per_mouse_blocklens"`` (dict): ``{subject: {block_num: length}}``
          mapping from 1-based block number to trial count (float).
        - ``"block_to_vals"`` (dict): ``{block_num: [lengths]}`` pooled across
          all subjects.
        - ``"blocks"`` (list[int]): Sorted list of 1-based block numbers.
        - ``"meds"`` (np.ndarray): Median block length per block across mice.
        - ``"ses"`` (np.ndarray): Standard error of block length per block.
        - ``"mice"`` (list[str]): Sorted list of subject keys.

    Raises:
        ValueError: If *boundary* is not one of ``"all"``, ``"good"``,
            ``"bad"``, or if no subject has at least one reversal boundary.
    """
    all_good_idx, all_bad_idx, _ = get_all_reversal_indices(subjects_trials)

    per_mouse_boundaries = _boundaries_per_mouse(
        subjects_trials, all_good_idx=all_good_idx, all_bad_idx=all_bad_idx, boundary=boundary
    )
    if not per_mouse_boundaries:
        raise ValueError("No mice had >=1 reversal boundary with the chosen boundary type.")

    per_mouse_blocklens = {m: _block_lengths(idxs) for m, idxs in per_mouse_boundaries.items()}
    block_to_vals = _pool_by_block(per_mouse_blocklens)

    blocks = sorted(block_to_vals.keys())
    meds = np.array([np.median(block_to_vals[b]) for b in blocks], dtype=float)
    ses = np.asarray([_standard_error(block_to_vals[b]) for b in blocks], dtype=float)
    mice = sorted(per_mouse_blocklens.keys())

    return {
        "boundary": boundary,
        "per_mouse_blocklens": per_mouse_blocklens,
        "block_to_vals": {k: list(v) for k, v in block_to_vals.items()},
        "blocks": blocks,
        "meds": meds,
        "ses": ses,
        "mice": mice,
    }
