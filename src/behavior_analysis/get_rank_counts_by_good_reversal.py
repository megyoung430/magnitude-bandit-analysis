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


def get_rank_counts_by_value(subjects_trials, reversal_windows, value_basis="true"):
    """Count choices in the same reversal windows by true or believed rank.

    Beliefs are the last reward observed at each arm, carried across sessions
    within the supplied problem. Rank is evaluated BEFORE updating the chosen
    arm. As in ``compute_believed_value``, only previously seen arms are ranked;
    choices of unseen arms are excluded. Ties share weight equally among the
    occupied rank positions. ``total`` counts eligible choices, so pooling uses
    the appropriate denominator. True-value counts retain the existing logic.
    """
    if value_basis == "true":
        return get_rank_counts_by_good_reversal(reversal_windows)
    if value_basis != "believed":
        raise ValueError("value_basis must be 'true' or 'believed'")

    from src.behavior_analysis.get_variables_across_sessions import get_vars_across_all_sessions

    merged, _ = get_vars_across_all_sessions(subjects_trials)
    ranks = ("best", "second", "third")
    result = {}
    for subject, windows in reversal_windows.items():
        data = merged[subject]
        rewards = data['reward_magnitudes_by_tower']
        choices = data['choices_by_tower']
        if len(rewards) > 3:
            raise ValueError("Rank proportions support at most three arms per problem")
        last_seen, weights = {}, []
        for i in range(len(data['trial'])):
            chosen = next((arm for arm, values in choices.items() if values[i]), None)
            weight = dict.fromkeys(ranks, 0.0)
            seen = {arm: last_seen[arm] for arm in rewards if arm in last_seen}
            if chosen in seen:
                value = seen[chosen]
                above = sum(v > value for v in seen.values())
                tied = sum(v == value for v in seen.values())
                for position in range(above, above + tied):
                    weight[ranks[position]] = 1.0 / tied
            weights.append(weight)
            if chosen in rewards and rewards[chosen][i] is not None:
                last_seen[chosen] = rewards[chosen][i]

        result[subject] = []
        for window in windows:
            indices = window['trial_window_idx']['post']
            counts = {rank: sum(weights[i][rank] for i in indices) for rank in ranks}
            total = sum(counts.values())
            result[subject].append({
                **counts, 'total': total,
                **{rank + '_prop': counts[rank] / total if total else np.nan for rank in ranks},
            })
    return result
