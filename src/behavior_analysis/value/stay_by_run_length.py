"""P(stay) as a function of run length: does staying build up the longer a port is kept?

A *run* is the consecutive trials at one port within a session (it restarts at a session start
and after a dropped trial, like the choice history of the choice logit). For every decision (a
trial that has a previous choice) the run length is that of the run ending at the previous
trial, and the decision is a *stay* if the choice repeats it. P(stay) is reported separately
after a zero payout and after a non-zero one, pooled over the non-zero magnitudes.

Pooling magnitudes is a choice: a port is kept longer when it pays well, so the magnitude mix of
the long runs differs from the short ones. The agents are replayed on the same task, so they
carry the same mix, which makes the comparison with the mice fair; read the slope of an agent
that ignores magnitude (plain WSLS) as the part of the slope that the mix alone produces.

Everything here is pure: a trial table in, tables out.
"""

import warnings

import numpy as np
import pandas as pd

from src.agents import null_agents as na

# Run-length bins: lengths 1, 2, 3, 4-5, 6-9 and 10 or more.
BIN_EDGES = (1, 2, 3, 4, 6, 10)
BIN_NAMES = ("1", "2", "3", "4-5", "6-9", "10+")
PAYOUTS = ("zero", "non-zero")
MIN_PAIRS = 20  # decisions a bin needs to be reported
MIN_BINS_FOR_SLOPE = 3


def _problem_run_lengths(rows: pd.DataFrame) -> np.ndarray:
    port_list = sorted(rows["magnitudes"].iloc[0])
    chosen = np.array([port_list.index(c) for c in rows["choice"]], dtype=int)
    start = (rows["trial_in_session"].to_numpy() == 0) | rows["after_gap"].to_numpy(dtype=bool)
    start[0] = True
    run = np.zeros(len(chosen), dtype=int)
    for t in range(1, len(chosen)):
        if not start[t]:
            run[t] = run[t - 1] + 1 if chosen[t - 1] == chosen[t - 2] and not start[t - 1] else 1
    return run


def run_lengths(trials: pd.DataFrame) -> np.ndarray:
    """Length of the run ending at the previous trial, at every trial (0 at a segment start).

    Args:
        trials (pd.DataFrame): One mouse's rows of the drifting trial table; rows are put in
            trial order within each problem and the result follows the problems in order of
            first appearance, as :func:`src.agents.null_agents.decisions`.

    Returns:
        np.ndarray: One integer per trial.
    """
    return np.concatenate(
        [
            _problem_run_lengths(rows.sort_values("trial"))
            for _, rows in trials.groupby("problem", sort=False)
        ]
    )


def bin_labels(lengths: np.ndarray) -> np.ndarray:
    """The bin name of each run length (at least 1).

    Raises:
        ValueError: If a run length is below 1.
    """
    lengths = np.asarray(lengths)
    if np.any(lengths < 1):
        raise ValueError("run lengths must be at least 1")
    return np.array(BIN_NAMES)[np.searchsorted(BIN_EDGES, lengths, side="right") - 1]


def stay_by_run_length(trials: pd.DataFrame, min_pairs: int = MIN_PAIRS) -> pd.DataFrame:
    """P(stay) per run-length bin, after a zero and after a non-zero payout.

    Args:
        trials (pd.DataFrame): One mouse's rows of the drifting trial table.
        min_pairs (int): Decisions a bin needs; a bin with fewer has NaN ``p_stay``. Default
            ``MIN_PAIRS``.

    Returns:
        pd.DataFrame: ``mouse, payout, bin, n, p_stay`` for every payout and bin of
        ``BIN_NAMES``; ``n`` is the number of decisions.

    Raises:
        ValueError: If *trials* hold other than one mouse.

    Warns:
        UserWarning: If a bin has some decisions but fewer than *min_pairs*.
    """
    if trials["mouse"].nunique() != 1:
        raise ValueError(f"stay_by_run_length takes one mouse, got {trials['mouse'].nunique()}")
    d = na.decisions(trials, na.DEFAULT_DECAY)
    pairs = ~d.start
    payout = np.where(np.nan_to_num(d.prev_magnitude) > 0, "non-zero", "zero")[pairs]
    bins = bin_labels(run_lengths(trials)[pairs])
    decisions = pd.DataFrame({"payout": payout, "bin": bins, "stayed": d.stayed[pairs]})
    grouped = decisions.groupby(["payout", "bin"])["stayed"].agg(n="size", p_stay="mean")
    index = pd.MultiIndex.from_product([PAYOUTS, BIN_NAMES], names=["payout", "bin"])
    table = grouped.reindex(index).fillna({"n": 0}).astype({"n": int}).reset_index()
    thin = table[(table["n"] > 0) & (table["n"] < min_pairs)]
    if len(thin):
        names = ", ".join(f"{r.payout} {r.bin} (n={r.n})" for r in thin.itertuples())
        warnings.warn(
            f"mouse {trials['mouse'].iloc[0]}: fewer than {min_pairs} decisions in {names}; "
            "left out",
            stacklevel=2,
        )
    table.loc[table["n"] < min_pairs, "p_stay"] = np.nan
    table.insert(0, "mouse", trials["mouse"].iloc[0])
    return table


def run_length_slopes(table: pd.DataFrame) -> pd.DataFrame:
    """Per mouse and payout, the least-squares slope of P(stay) against the bin index.

    One unit is one bin step (1, 2, 3, 4-5, 6-9, 10+), so the slope is the average rise of
    P(stay) per bin. A slope needs at least ``MIN_BINS_FOR_SLOPE`` bins with a value.

    Args:
        table (pd.DataFrame): :func:`stay_by_run_length` of one or more mice, concatenated.

    Returns:
        pd.DataFrame: ``mouse, payout, n_bins, slope`` (NaN with too few bins).
    """
    rows = []
    for (mouse, payout), group in table.groupby(["mouse", "payout"], sort=False):
        known = group.dropna(subset=["p_stay"])
        position = known["bin"].map(BIN_NAMES.index).to_numpy(dtype=float)
        enough = len(known) >= MIN_BINS_FOR_SLOPE
        slope = float(np.polyfit(position, known["p_stay"].to_numpy(), 1)[0]) if enough else np.nan
        rows.append(dict(mouse=mouse, payout=payout, n_bins=len(known), slope=slope))
    return pd.DataFrame(rows)
