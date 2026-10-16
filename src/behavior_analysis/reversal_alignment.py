"""Place reversal windows on a common trial axis and smooth them.

A reversal window is a ``pre`` list (trials before the reversal, oldest first) and a
``post`` list (the reversal trial and the trials after it).  Windows vary in length: a
reversal near the start of the data, or one close to its neighbouring block changes, has
fewer trials than the axis.  The helpers here put every trial in the bin of its true offset
so rows of different reversals can be stacked and averaged:

- **pre** values are right-aligned: the last one is at offset -1;
- **post** values are left-aligned: the reversal trial is at offset 0;
- bins without a value are NaN.

Window bounds.  :func:`window_indices` cuts the windows.  By default the pre window reaches
back ``pre`` trials whatever block they belong to and the post window ends at the next
reversal boundary.  With ``block_changes`` given (``stop_at_block_changes`` in
``get_good_reversal_info``, used for the drifting-value analyses) both windows stay inside
the block of the reversal.  ``get_bad_reversal_info`` keeps its own unbounded windows; it is
only used by exploratory notebooks.

Offset conventions.  A row has ``pre + (post - skip)`` bins.  With a skip ``k > 0`` the
post trials ``1 .. k`` are dropped, so the bins after the reversal trial hold trials
``k + 1, k + 2, ...``.

- :func:`bin_offsets` numbers the bins ``-pre .. -1, 0, 1, ...`` without the skip shift
  (the x axis of the choice-probability curves; the plotting layer adds ``k`` back).
- :func:`true_trial_offsets` is the same axis with the shift applied: the number of trials
  between each bin and the reversal.
"""

from collections.abc import Sequence

import numpy as np


def keep_zero_then_skip(post_list: Sequence, skip: int) -> list:
    """Drop post trials ``1 .. skip``, keeping the reversal trial (index 0).

    Args:
        post_list: Values from the reversal trial on.
        skip: Number of trials after the reversal trial to drop (``<= 0`` drops none).

    Returns:
        A list with ``len(post_list) - skip`` entries when *post_list* is long enough.
    """
    post_list = list(post_list)
    if skip <= 0:
        return post_list
    return post_list[:1] + post_list[skip + 1 :]


def _check_axis(pre: int, post: int, skip: int) -> None:
    """Raise ValueError unless ``0 <= skip < post`` and ``pre >= 0``."""
    if pre < 0:
        raise ValueError("pre must be >= 0")
    if skip < 0:
        raise ValueError("skip_n_trials_after_reversal must be >= 0")
    if skip >= post:
        raise ValueError(
            "skip_n_trials_after_reversal must be < post (otherwise post window is empty)."
        )


def bin_offsets(*, pre: int, post: int, skip: int = 0) -> np.ndarray:
    """Bin numbers ``[-pre, 0)`` followed by ``[0, post - skip)``, without the skip shift.

    Args:
        pre: Number of pre-reversal bins.
        post: Number of post-reversal trials before skipping.
        skip: Post trials ``1 .. skip`` that are dropped.

    Returns:
        Int array of length ``pre + post - skip``.

    Raises:
        ValueError: If *skip* is negative or ``>= post``, or *pre* is negative.
    """
    _check_axis(pre, post, skip)
    return np.concatenate([np.arange(-pre, 0), np.arange(0, post - skip)])


def true_trial_offsets(*, pre: int, post: int, skip: int = 0) -> np.ndarray:
    """Trials between each bin and the reversal: post bins after the first sit *skip* later.

    Args:
        pre: Number of pre-reversal bins.
        post: Number of post-reversal trials before skipping.
        skip: Post trials ``1 .. skip`` that are dropped.

    Returns:
        Int array of length ``pre + post - skip``; equals :func:`bin_offsets` for ``skip = 0``.

    Raises:
        ValueError: As :func:`bin_offsets`.
    """
    bins = bin_offsets(pre=pre, post=post, skip=skip)
    return np.where(bins >= 1, bins + skip, bins)


def align_window_values(
    pre_values: Sequence[float],
    post_values: Sequence[float],
    *,
    pre: int,
    post: int,
    skip: int = 0,
    truncate: bool = False,
) -> np.ndarray:
    """One row of ``pre + post - skip`` bins from a reversal's pre and post values.

    Args:
        pre_values: Values of the trials before the reversal, oldest first.
        post_values: Values from the reversal trial on.
        pre: Pre-reversal bins in the row.
        post: Post-reversal trials in the row, before skipping.
        skip: Post trial 0 is kept and post trials ``1 .. skip`` are dropped.
        truncate: If ``True`` a window longer than the axis is cut (the last *pre* pre
            values, the first ``post - skip`` post values); if ``False`` it raises.

    Returns:
        Float array. The last pre value is in bin ``pre - 1`` (offset -1), the first post
        value in bin ``pre`` (offset 0); bins without a value are NaN.

    Raises:
        ValueError: If *skip* is negative or leaves no post window, or if a window is longer
            than the axis and *truncate* is ``False`` (it was built with other settings).
    """
    _check_axis(pre, post, skip)
    pre_values = list(pre_values)
    kept_post = keep_zero_then_skip(post_values, skip)
    n_post = post - skip
    if not truncate:
        if len(pre_values) > pre:
            raise ValueError(f"pre window has {len(pre_values)} trials but pre={pre}")
        if len(kept_post) > n_post:
            raise ValueError(
                f"post window has {len(kept_post)} trials after skipping; max {n_post}"
            )
    pre_values = pre_values[max(len(pre_values) - pre, 0) :]
    kept_post = kept_post[:n_post]
    row = np.full(pre + n_post, np.nan)
    row[pre - len(pre_values) : pre] = pre_values
    row[pre : pre + len(kept_post)] = kept_post
    return row


def window_bounds(
    boundaries: Sequence[int],
    k: int,
    n_trials: int,
    *,
    block_changes: Sequence[int] | None = None,
) -> tuple[int, int]:
    """Earliest pre trial and one-past-last post trial allowed around boundary number *k*.

    Args:
        boundaries: Sorted reversal indices.
        k: Which boundary to window.
        n_trials: Number of trials in the data.
        block_changes: ``None`` keeps the boundary-only rule: the pre window may reach back
            to the start of the data and the post window ends at the next boundary (or the
            last trial).  A sequence of block-change indices bounds both sides to the block
            of the reversal: the pre window starts at the previous boundary or block change
            and the post window ends at the next one.

    Returns:
        ``(pre_start, post_end)``: the pre window may use trials ``pre_start .. idx - 1``
        and the post window trials ``idx .. post_end - 1``.
    """
    idx = boundaries[k]
    if block_changes is None:
        return 0, boundaries[k + 1] if k + 1 < len(boundaries) else n_trials
    stops = {*boundaries, *block_changes}
    earlier = [b for b in stops if b < idx]
    later = [b for b in stops if b > idx]
    return max(earlier, default=0), min(later, default=n_trials)


def window_indices(
    boundaries: Sequence[int],
    k: int,
    n_trials: int,
    *,
    pre: int,
    post: int | None,
    block_changes: Sequence[int] | None = None,
) -> tuple[list[int], list[int]]:
    """Absolute ``(pre_idx, post_idx)`` trial indices around boundary number *k*.

    Args:
        boundaries: Sorted reversal indices.
        k: Which boundary to window.
        n_trials: Number of trials in the data.
        pre: Maximum number of pre trials.
        post: Maximum number of post trials; ``None`` runs to the end of the allowed range.
        block_changes: As in :func:`window_bounds`; ``None`` leaves the pre window
            unbounded by earlier blocks, a sequence confines both windows to one block.

    Returns:
        ``(pre_idx, post_idx)`` lists of trial indices; the pre list may be shorter than *pre*.
    """
    idx = boundaries[k]
    pre_start, post_end = window_bounds(boundaries, k, n_trials, block_changes=block_changes)
    post_stop = post_end if post is None else min(n_trials, idx + post, post_end)
    return list(range(max(pre_start, idx - pre), idx)), list(range(idx, post_stop))


def _centered_window_sum(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Sliding sum of *kernel* width, centred, with the same length as *values*.

    Equal to ``np.convolve(values, kernel, "same")`` when *values* is at least as
    long as *kernel*; that call returns ``len(kernel)`` entries for a shorter
    input, which would not fit the segment it is written back to.
    """
    start = (len(kernel) - 1) // 2
    return np.convolve(values, kernel, mode="full")[start : start + len(values)]


def nan_moving_average(
    y: Sequence[float], window: int | None, mode: str = "centered"
) -> np.ndarray:
    """NaN-aware moving average of *window* width.

    Each output bin is the mean of the finite values inside its window; bins whose window
    holds no finite value are NaN.  A centred window of even width ``w`` covers bins
    ``[i - w/2, i + w/2 - 1]`` (two before and one after for ``w = 4``).

    Args:
        y: Values to smooth.
        window: Window width; ``None`` or ``<= 1`` returns *y* as a float array.
        mode: ``"trailing"`` averages the current and previous ``window - 1`` bins; anything
            else is centred.

    Returns:
        Float array of the same length as *y*.
    """
    y = np.asarray(y, dtype=float)
    if window is None or window <= 1:
        return y

    valid = np.isfinite(y).astype(float)
    y0 = np.nan_to_num(y, nan=0.0)
    kernel = np.ones(int(window), dtype=float)

    if mode == "trailing":
        num = np.convolve(y0, kernel, mode="full")[: len(y0)]
        den = np.convolve(valid, kernel, mode="full")[: len(y0)]
    else:
        num = _centered_window_sum(y0, kernel)
        den = _centered_window_sum(valid, kernel)

    out = np.full_like(y0, np.nan, dtype=float)
    has_data = den > 0
    out[has_data] = num[has_data] / den[has_data]
    return out


def smooth_pre_post(
    y: Sequence[float], pre_mask: np.ndarray, window: int | None, mode: str = "centered"
) -> np.ndarray:
    """Smooth *y* separately on the ``pre_mask`` and ``~pre_mask`` segments.

    Args:
        y: Values over the bins of the row.
        pre_mask: Boolean array, ``True`` for pre-reversal bins.
        window: Window width (see :func:`nan_moving_average`).
        mode: ``"centered"`` or ``"trailing"``.

    Returns:
        Float array of the length of *y*; no value crosses the pre/post boundary.
    """
    y = np.asarray(y, dtype=float)
    out = np.full_like(y, np.nan, dtype=float)
    for segment in (pre_mask, ~pre_mask):
        if segment.any():
            out[segment] = nan_moving_average(y[segment], window, mode=mode)
    return out
