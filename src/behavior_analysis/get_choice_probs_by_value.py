import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.behavior_analysis.stats import nan_mean
from src.behavior_import.extract_trials import merge_trial_variables
from src.behavior_import.session_order import ordered_session_keys

from .get_believed_value import compute_believed_value, rank_position_shares, tie_category

BEST_VALS = [2, 3, 4]
SECOND_VALS = [1, 2, 3]

logger = logging.getLogger(__name__)

TIE_EXCLUDE = "exclude"
TIE_SPLIT = "split"
TIE_MODES = (TIE_EXCLUDE, TIE_SPLIT)

# A believed-value trial counts only once every arm of the problem has been sampled.
N_ARMS = 3

REASON_TOO_FEW_ARMS = "too_few_arms"
REASON_UNRECOGNISED_RANK = "unrecognised_rank"
REASON_FEWER_THAN_THREE_SEEN = "fewer_than_three_arms_seen"
REASON_CHOSEN_UNSEEN = "chosen_arm_unseen"
REASON_TIE_EXCLUDED = "tie_excluded"
REASON_OUT_OF_GRID = "out_of_grid"


def init_matrix(fill=np.nan):
    """Empty best-value by second-value grid, every cell set to *fill*.

    Args:
        fill (float): Value for every cell. Default ``np.nan``.

    Returns:
        pd.DataFrame: 3x3 frame with rows indexed by ``best_value`` (2, 3, 4) and columns
        by ``second_value`` (1, 2, 3).
    """
    return pd.DataFrame(
        fill,
        index=pd.Index(BEST_VALS, name="best_value"),
        columns=pd.Index(SECOND_VALS, name="second_value"),
    )


class _Contribution(NamedTuple):
    """One countable trial: its cell and how much it adds to each numerator.

    Named fields keep ``best_value`` and ``second_value`` from being swapped silently.
    """

    best_value: float
    second_value: float
    best_share: float
    second_share: float
    tie: str | None = None


class _Dropped(NamedTuple):
    """A trial that is not counted: why, its (best, second) values if out of grid, its tie."""

    reason: str
    cell: tuple[float, float] | None = None
    tie: str | None = None


# ``frozen`` only blocks reassigning the fields: the DataFrames and Counters are updated in place.
@dataclass(frozen=True)
class MouseCounts:
    """Per-cell trial counts (best value x second value) for one mouse.

    ``denom`` counts the trials in each cell; ``num_best`` and ``num_second`` count how
    much of that was a best / second choice (fractional when magnitudes tie).
    ``out_of_grid`` counts the trials whose (best, second) values are not a grid cell, by
    those values; ``dropped`` counts every other uncounted trial by reason (including
    ``tie_excluded`` in ``"exclude"`` mode). ``ties`` counts the trials with tied values by
    category (``best_second``, ``second_third``, ``all_three``), whatever happens to them next.

    Attributes:
        denom (pd.DataFrame): Trials per cell, in the grid of :func:`init_matrix`.
        num_best (pd.DataFrame): Best-choice weight per cell, same grid.
        num_second (pd.DataFrame): Second-choice weight per cell, same grid.
        dropped (Counter): ``{reason: number of uncounted trials}``.
        out_of_grid (Counter): ``{(best_value, second_value): number of trials}``.
        ties (Counter): ``{tie category: number of trials}``.
    """

    denom: pd.DataFrame
    num_best: pd.DataFrame
    num_second: pd.DataFrame
    dropped: Counter = field(default_factory=Counter)
    out_of_grid: Counter = field(default_factory=Counter)
    ties: Counter = field(default_factory=Counter)

    @classmethod
    def empty(cls) -> "MouseCounts":
        return cls(init_matrix(fill=0.0), init_matrix(fill=0.0), init_matrix(fill=0.0))


def _in_value_grid(b, s):
    """Whether ``(b, s)`` is one of the matrix cells."""
    return b in BEST_VALS and s in SECOND_VALS


def _has_tied_magnitudes(reward_magnitudes: dict) -> bool:
    """Whether two arms share a magnitude (their rank labels are then arbitrary)."""
    values = list(reward_magnitudes.values())
    return len(set(values)) < len(values)


def _true_rank_shares(reward_magnitudes: dict, chosen_rank: str, choice) -> tuple[float, float]:
    """``(best_share, second_share)`` of the chosen arm for one trial's true magnitudes.

    Without ties the recorded rank label is used. With ties that label is arbitrary, so the
    chosen arm's own magnitude decides and the tied arms share their rank positions.

    Raises:
        ValueError: If the magnitudes tie and *choice* is not one of the arms.
    """
    if not _has_tied_magnitudes(reward_magnitudes):
        return float(chosen_rank == "best"), float(chosen_rank == "second")
    if choice not in reward_magnitudes:
        raise ValueError(
            f"choice {choice!r} is not one of the arms {list(reward_magnitudes)} of a trial "
            "with tied magnitudes, so its share of the tied ranks cannot be set"
        )
    return rank_position_shares(reward_magnitudes[choice], reward_magnitudes.values())


def _true_contribution(reward_magnitudes, chosen_rank, choice=None, tie_mode=TIE_SPLIT):
    """A :class:`_Contribution` for one trial binned by true magnitudes, else a :class:`_Dropped`.

    Dropped when the trial has fewer than two arms, an unrecognised chosen rank, magnitudes
    outside the grid, or (in ``"exclude"`` mode) tied magnitudes. *choice* is only read when
    magnitudes tie and are split.
    """
    sorted_arms = sorted(reward_magnitudes.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_arms) < 2:
        return _Dropped(REASON_TOO_FEW_ARMS)
    b, s = sorted_arms[0][1], sorted_arms[1][1]
    if chosen_rank not in ("best", "second", "third"):
        return _Dropped(REASON_UNRECOGNISED_RANK)
    return _bin_trial(
        b,
        s,
        tie_category(reward_magnitudes.values()),
        tie_mode,
        lambda: _true_rank_shares(reward_magnitudes, chosen_rank, choice),
    )


def _bin_trial(b, s, tie, tie_mode, shares):
    """Shared end of both binnings: apply the tie mode and the grid, then take the shares.

    *shares* is called only for a trial that is counted.
    """
    if tie and tie_mode == TIE_EXCLUDE:
        return _Dropped(REASON_TIE_EXCLUDED, tie=tie)
    if not _in_value_grid(b, s):
        return _Dropped(REASON_OUT_OF_GRID, (b, s), tie)
    return _Contribution(b, s, *shares(), tie)


def _believed_contribution(trial, tie_mode=TIE_SPLIT):
    """A :class:`_Contribution` for one ``compute_believed_value`` trial, else a :class:`_Dropped`.

    Dropped when fewer than three arms have been seen, the chosen arm has no believed rank,
    the believed magnitudes tie in ``"exclude"`` mode, or they are outside the grid. In
    ``"split"`` mode tied believed arms share the rank positions they cover evenly (see
    ``rank_position_shares``), so the result does not depend on arm names.
    """
    if trial["n_seen"] < N_ARMS:
        return _Dropped(REASON_FEWER_THAN_THREE_SEEN)
    if trial["believed_chosen_rank"] is None:
        return _Dropped(REASON_CHOSEN_UNSEEN)
    return _bin_trial(
        trial["believed_best_mag"],
        trial["believed_second_mag"],
        trial["believed_tie"],
        tie_mode,
        lambda: (trial["believed_best_share"], trial["believed_second_share"]),
    )


def _session_trial_variables(tv_list):
    """One ``{variable: list}`` dict for a session, merging its recordings in order.

    Uses the same merge as the importer (``merge_trial_variables``); a single
    recording is used as is.
    """
    return tv_list[0] if len(tv_list) == 1 else merge_trial_variables(tv_list)


def _check_equal_lengths(tv):
    """Raise ValueError unless the three per-trial lists of one recording are equally long."""
    lengths = (
        len(tv.get("reward_magnitudes", [])),
        len(tv.get("chosen_rank", [])),
        len(tv.get("choice", [])),
    )
    if len(set(lengths)) > 1:
        raise ValueError(
            "trial lists differ in length: "
            f"reward_magnitudes={lengths[0]}, chosen_rank={lengths[1]}, choice={lengths[2]}"
        )


def session_trial_lists(sess):
    """``(reward_magnitudes, chosen_ranks, choices)`` over all recordings, or ``None``.

    The recordings of the session are merged in order (as the importer does). ``None`` when the
    session has no trial variables or no trials.

    Raises:
        ValueError: If the three lists of any recording differ in length.
    """
    tv_list = sess.get("trial_variables", [])
    if not tv_list:
        return None
    for tv in tv_list:  # per recording: equal totals could hide misaligned recordings
        _check_equal_lengths(tv)
    tv = _session_trial_variables(tv_list)
    reward_magnitudes = tv.get("reward_magnitudes", [])
    if len(reward_magnitudes) == 0:
        return None
    return reward_magnitudes, tv.get("chosen_rank", []), tv.get("choice", [])


def _problem_trial_lists(subject_id, sessions_dict):
    """The trial lists of all of a mouse's sessions in one problem, in session order.

    Beliefs about the arms carry over from one session to the next within a problem, so the
    sessions are joined chronologically (``ordered_session_keys``, never dict order).
    """
    reward_magnitudes, chosen_ranks, choices = [], [], []
    for key in ordered_session_keys(subject_id, sessions_dict):
        trial_lists = session_trial_lists(sessions_dict[key])
        if trial_lists is not None:
            for joined, new in zip(
                (reward_magnitudes, chosen_ranks, choices), trial_lists, strict=True
            ):
                joined.extend(new)
    return reward_magnitudes, chosen_ranks, choices


def _trial_outcomes(reward_magnitudes, chosen_ranks, choices, *, use_believed, tie_mode):
    """A :class:`_Contribution` or :class:`_Dropped` for every trial of one problem."""
    if use_believed:
        beliefs = compute_believed_value(choices, reward_magnitudes)
        return [_believed_contribution(trial, tie_mode) for trial in beliefs]
    return [
        _true_contribution(magnitudes, rank, choice, tie_mode)
        for magnitudes, rank, choice in zip(reward_magnitudes, chosen_ranks, choices, strict=True)
    ]


def _add_outcome(counts, outcome):
    """Add one trial to *counts*: to its cell, or to the matching dropped tally."""
    if outcome.tie:
        counts.ties[outcome.tie] += 1
    if isinstance(outcome, _Dropped):
        if outcome.reason == REASON_OUT_OF_GRID:
            counts.out_of_grid[outcome.cell] += 1
        else:
            counts.dropped[outcome.reason] += 1
        return
    b, s = outcome.best_value, outcome.second_value
    counts.denom.loc[b, s] += 1.0
    if outcome.best_share:  # zero shares leave their numerator untouched
        counts.num_best.loc[b, s] += outcome.best_share
    if outcome.second_share:
        counts.num_second.loc[b, s] += outcome.second_share


def _count_mouse_problem(counts, subject_id, sessions_dict, *, use_believed, tie_mode):
    """Accumulate one mouse's sessions of one problem into *counts* (in place)."""
    trial_lists = _problem_trial_lists(subject_id, sessions_dict)
    outcomes = _trial_outcomes(*trial_lists, use_believed=use_believed, tie_mode=tie_mode)
    for outcome in outcomes:
        _add_outcome(counts, outcome)


def _warn_out_of_grid(subject_id, counts):
    """Log a warning if any of the mouse's trials fell outside the value grid."""
    n_out = sum(counts.out_of_grid.values())
    if n_out == 0:
        return
    total = int(counts.denom.to_numpy().sum()) + n_out
    cells = ", ".join(f"({b}, {s}) x{n}" for (b, s), n in sorted(counts.out_of_grid.items()))
    logger.warning(
        "%s: %d of %d trials (counted plus out-of-grid; other skips are not included) have "
        "best/second values outside the grid (best %s, second %s) and were not counted: %s.",
        subject_id,
        n_out,
        total,
        BEST_VALS,
        SECOND_VALS,
        cells,
    )


def count_choice_probs(
    subject_trials_by_value_problem: dict,
    use_believed: bool = False,
    tie_mode: str = TIE_SPLIT,
    warn_out_of_grid: bool = True,
) -> dict[str, MouseCounts]:
    """Per-mouse cell counts, pooling a mouse's trials across all problems.

    ``subject_trials_by_value_problem[problem][subject_id][session_id]`` is a session dict
    (see :func:`accumulate_choice_probs`). Counting is done within each problem, so believed
    values start empty at every problem boundary and carry over the problem's sessions in
    session order; the counts of a mouse's problems are then added. Unless
    *warn_out_of_grid* is off, a warning is logged for each mouse with trials outside the
    value grid.

    Args:
        subject_trials_by_value_problem (dict): Sessions by problem, mouse and session id.
        use_believed (bool): Bin by believed instead of true magnitudes. Believed trials
            count only once all three arms have been sampled in the problem.
        tie_mode (str): ``"split"`` shares each tied rank position evenly among the tied arms;
            ``"exclude"`` drops every trial with a tie among the top three values. Ties are
            counted in ``MouseCounts.ties`` either way.
        warn_out_of_grid (bool): Log a warning for each mouse with out-of-grid trials.

    Returns:
        dict[str, MouseCounts]: Counts per mouse.

    Raises:
        ValueError: If *tie_mode* is unknown, a recording's trial lists differ in length, a
            session key has no ``ses-<n>`` or the session numbers and dates disagree, or (in
            ``"split"`` mode) a trial with tied true magnitudes has an unknown choice.
    """
    if tie_mode not in TIE_MODES:
        raise ValueError(f"tie_mode must be one of {TIE_MODES}, got {tie_mode!r}")
    counts_by_mouse: dict[str, MouseCounts] = {}
    for subjects_dict in subject_trials_by_value_problem.values():
        for subject_id, sessions_dict in subjects_dict.items():
            counts = counts_by_mouse.setdefault(subject_id, MouseCounts.empty())
            _count_mouse_problem(
                counts, subject_id, sessions_dict, use_believed=use_believed, tie_mode=tie_mode
            )
    if warn_out_of_grid:
        for subject_id, counts in counts_by_mouse.items():
            _warn_out_of_grid(subject_id, counts)
    return counts_by_mouse


def _probabilities(numerator, denom):
    """Numerator / denominator per cell; NaN where a cell has no trials."""
    return numerator.divide(denom.where(denom > 0))


def _nanmean_matrices(prob_dict):
    """Cell-wise mean across mice, ignoring NaN, so every mouse contributes equally per cell."""
    if not prob_dict:
        return init_matrix(fill=np.nan)
    stack = np.stack([df.to_numpy(dtype=float) for df in prob_dict.values()], axis=0)
    ref = next(iter(prob_dict.values()))
    return pd.DataFrame(nan_mean(stack, axis=0), index=ref.index, columns=ref.columns)


def choice_probs_from_counts(counts_by_mouse: dict[str, MouseCounts]) -> tuple:
    """Per-mouse and across-mouse P(best) and P(second) matrices from cell counts.

    Args:
        counts_by_mouse (dict[str, MouseCounts]): Output of :func:`count_choice_probs`.

    Returns:
        tuple: ``(overall_best, overall_second, per_mouse_best, per_mouse_second)`` as in
        :func:`accumulate_choice_probs`.
    """
    per_mouse_best_prob = {
        sid: _probabilities(c.num_best, c.denom) for sid, c in counts_by_mouse.items()
    }
    per_mouse_second_prob = {
        sid: _probabilities(c.num_second, c.denom) for sid, c in counts_by_mouse.items()
    }
    overall_best_prob = _nanmean_matrices(per_mouse_best_prob)
    overall_second_prob = _nanmean_matrices(per_mouse_second_prob)
    return overall_best_prob, overall_second_prob, per_mouse_best_prob, per_mouse_second_prob


def accumulate_choice_probs(
    subject_trials_by_value_problem, use_believed=False, tie_mode=TIE_SPLIT
):
    """Per-mouse and mean P(best) and P(second) grids over (best value, second value) cells.

    Takes ``subject_trials_by_value_problem[problem][subject_id][session_id] = session_data``.

    session_data["trial_variables"] is a list with one dict per recording file; the
    recordings are concatenated in order. Each dict holds parallel lists, one value per
    trial (a length mismatch raises ``ValueError``). Keys include:
      - "reward_magnitudes" : list[dict]  — arm->magnitude per trial
      - "chosen_rank"       : list[str]   — "best"/"second"/"third" per trial
      - "choice"            : list[str]   — chosen arm label per trial

    Trials outside the value grid are not counted; use :func:`count_choice_probs` to get their
    counts and the tie counts (a warning is logged per mouse).

    Args:
        subject_trials_by_value_problem (dict): ``{problem: {subject_id: {session_id:
            session_data}}}`` as described above.
        use_believed (bool): If False (default), bin by true reward magnitudes and true
            chosen rank. If True, bin by believed magnitudes (last observed value per arm,
            carried across the sessions of a problem and reset at each problem), with arms
            re-ranked by believed value to determine best/second and the chosen rank. Trials
            are skipped until all three arms have been visited, or where the chosen arm has
            no believed value.
        tie_mode (str): ``"split"`` (default) splits the tied rank positions of tied
            magnitudes (true or believed) evenly among the tied arms; ``"exclude"`` leaves out
            trials with any tie among the top three values.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, dict, dict]: ``(overall_best_prob_df,
        overall_second_prob_df, per_mouse_best_prob, per_mouse_second_prob)``. The overall
        frames are 3x3 (rows ``best_value`` 2-4, columns ``second_value`` 1-3) means across
        mice, ignoring empty cells; the per-mouse values are ``{subject_id: 3x3 DataFrame}``.
        A cell with no trials is NaN.

    Raises:
        ValueError: For an unknown ``tie_mode``, a length mismatch of a recording's trial
            lists, a session key without ``ses-<n>`` (or numbers and dates that disagree),
            or a split trial with tied true magnitudes and an unknown choice.
    """
    counts_by_mouse = count_choice_probs(subject_trials_by_value_problem, use_believed, tie_mode)
    return choice_probs_from_counts(counts_by_mouse)
