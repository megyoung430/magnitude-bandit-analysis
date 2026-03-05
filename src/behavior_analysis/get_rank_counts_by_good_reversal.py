"""Aggregate arm-rank choice counts per good reversal for all subjects."""
import numpy as np


def get_rank_counts_by_good_reversal(good_reversal_info, include_first_block=True):
    """Count and proportion best/second/third arm choices in each reversal's post window.

    Iterates over all good reversals for every subject and sums the one-hot
    choice counts in the post window for each rank.  The result is used
    downstream for rank-proportion plots and statistical tests.

    Args:
        good_reversal_info: ``{subject: list[reversal_dict]}`` as returned by
            :func:`src.behavior_analysis.get_good_reversal_info.get_good_reversal_info`.
            Each reversal dict must contain a ``"choices_by_rank"`` key with
            sub-keys ``"best"``, ``"second"``, ``"third"``, each holding a
            ``{"pre": [...], "post": [...]}`` one-hot list.
        include_first_block: Parameter accepted for API symmetry; not currently
            used in the computation (default: ``True``).

    Returns:
        Dict ``{subject: list[rank_count_dict]}`` with one entry per reversal.
        Each *rank_count_dict* contains:

        - ``"best"`` (int): Number of post-window trials where the best arm
          was chosen.
        - ``"second"`` (int): Number of post-window trials where the second
          arm was chosen.
        - ``"third"`` (int): Number of post-window trials where the third arm
          was chosen.
        - ``"total"`` (int): Total post-window trials (sum of the above three).
        - ``"best_prop"`` (float): Proportion of best-arm choices (``nan`` if
          ``total == 0``).
        - ``"second_prop"`` (float): Proportion of second-arm choices.
        - ``"third_prop"`` (float): Proportion of third-arm choices.
    """
    rank_counts_by_good_reversal = {}
    for subj in good_reversal_info.keys():
        rank_counts_by_good_reversal[(subj)] = []
        for i in range(0, len(good_reversal_info[subj])):
            num_best = sum(good_reversal_info[subj][i]['choices_by_rank']['best']['post'])
            num_second = sum(good_reversal_info[subj][i]['choices_by_rank']['second']['post'])
            num_third = sum(good_reversal_info[subj][i]['choices_by_rank']['third']['post'])

            total = num_best + num_second + num_third
            assert total == len(good_reversal_info[subj][i]['choices_by_rank']['best']['post']), "Total does not match number of trials"

            rank_counts_by_good_reversal[(subj)].append({
                'best': num_best,
                'second': num_second,
                'third': num_third,
                'total': total,
                'best_prop': num_best / total if total > 0 else np.nan,
                'second_prop': num_second / total if total > 0 else np.nan,
                'third_prop': num_third / total if total > 0 else np.nan
            })
    return rank_counts_by_good_reversal
