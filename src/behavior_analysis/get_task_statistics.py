"""Compute summary statistics about block structure and lengths.

Block lengths are derived from the trial indices at which reversals occur.
Good, bad, or all reversals can be used as block boundaries.
"""
import numpy as np
from collections import defaultdict
from src.behavior_analysis.get_total_reversals import get_all_reversal_indices


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

    # ---- 1) choose reversal indices per mouse based on boundary type ----
    per_mouse_boundaries = {}
    for m in subjects_trials.keys():
        goods = all_good_idx.get(m, [])
        bads  = all_bad_idx.get(m, [])

        if boundary == "good":
            idxs = sorted(goods)
        elif boundary == "bad":
            idxs = sorted(bads)
        elif boundary == "all":
            idxs = sorted(set(goods) | set(bads))
        else:
            raise ValueError('boundary must be "all", "good", or "bad"')

        # We can compute blocks as long as there is >=1 reversal boundary
        # (because block 1 is 0->rev_1). If idxs is empty, skip mouse.
        if len(idxs) >= 1:
            per_mouse_boundaries[m] = idxs

    if not per_mouse_boundaries:
        raise ValueError("No mice had >=1 reversal boundary with the chosen boundary type.")

    # ---- 2) per-mouse block lengths using [0] + reversals ----
    per_mouse_blocklens = {}
    for m, rev_idxs in per_mouse_boundaries.items():
        boundaries = [0] + list(rev_idxs)
        lens = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]
        per_mouse_blocklens[m] = {b + 1: float(L) for b, L in enumerate(lens)}

    # ---- 3) aggregate across mice by block number ----
    block_to_vals = defaultdict(list)
    for m, bl in per_mouse_blocklens.items():
        for b, L in bl.items():
            if np.isfinite(L):
                block_to_vals[b].append(L)

    blocks = sorted(block_to_vals.keys())
    meds = np.array([np.median(block_to_vals[b]) for b in blocks], dtype=float)

    ses = []
    for b in blocks:
        vals = np.asarray(block_to_vals[b], dtype=float)
        ses.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) >= 2 else 0.0)
    ses = np.asarray(ses, dtype=float)

    mice = sorted(per_mouse_blocklens.keys())

    return {
        "boundary": boundary,
        "per_mouse_blocklens": dict(per_mouse_blocklens),
        "block_to_vals": {k: list(v) for k, v in block_to_vals.items()},
        "blocks": blocks,
        "meds": meds,
        "ses": ses,
        "mice": mice,
    }
