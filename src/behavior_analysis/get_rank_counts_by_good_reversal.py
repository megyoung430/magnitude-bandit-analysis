def get_rank_counts_by_good_reversal(good_reversal_info, include_first_block=True):
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