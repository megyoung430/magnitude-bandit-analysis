"""Match problem exposure using the first N chronological trials per mouse."""
import math
import re
from copy import deepcopy


def match_trial_budget(subjects, subject_numbers, reference_numbers=(5,6,7,8), *, trial_budget=None):
    """Use a fixed ``trial_budget`` or the floored reference-mouse mean as a cap.

    Mice with fewer trials retain all available trials and are flagged in the
    report. Full sessions retain terminal reversal counts; a partial session's
    totals use the counter at the next trial boundary, including a reversal
    after its final retained trial, but excluding later reversals.
    """
    counts = {mouse: sum(len(s.get('trial', [])) for s in sessions.values())
              for mouse,sessions in subjects.items()}
    reference, mean = {}, None
    if trial_budget is None:
        reference = {m:n for m,n in counts.items() if subject_numbers[m] in reference_numbers}
        missing = set(reference_numbers) - {subject_numbers[m] for m in reference}
        if missing or not reference or any(n == 0 for n in reference.values()):
            raise ValueError(f'Reference mice must all have trials; missing subject numbers: {sorted(missing)}')
        mean = sum(reference.values()) / len(reference)
        budget = math.floor(mean)
    else:
        if not isinstance(trial_budget, int) or isinstance(trial_budget, bool):
            raise ValueError('trial_budget must be a positive integer')
        budget = trial_budget
    if budget < 1:
        raise ValueError('Trial budget must be positive')
    result, report = {}, []
    for mouse,sessions in subjects.items():
        left = budget
        kept = {}
        for key in sorted(sessions, key=lambda k: (int(re.search(r'ses-(\d+)',k)[1]),k)):
            if left <= 0:
                break
            source = sessions[key]
            n = len(source.get('trial', []))
            if not n:
                continue
            take = min(n,left)
            # Copy only analysis fields; avoid duplicating large raw dataframes.
            target = {k: deepcopy(source[k]) for k in (
                'trial','blocks','trials_in_block','good_reversals','bad_reversals',
                'reward_magnitudes_by_tower','choices_by_tower','choices_by_rank',
                'has_good','has_bad','reversal_totals','reversal_boundary_events') if k in source}
            target['trial_info'] = True
            for field in ('trial','blocks','trials_in_block','good_reversals','bad_reversals'):
                if field in target:
                    target[field] = target[field][:take]
            for field in ('reward_magnitudes_by_tower','choices_by_tower','choices_by_rank'):
                target[field] = {k:v[:take] for k,v in target.get(field,{}).items()}
            if take < n:
                target['reversal_totals'] = {}
                for kind in ('good','bad'):
                    arr = source.get(kind+'_reversals',[])
                    if arr:
                        value = next((v for v in reversed(arr[:take+1]) if v is not None),0)
                        target['reversal_totals'][kind] = value
                target['reversal_boundary_events'] = [e for e in target.get('reversal_boundary_events',[]) if e['index'] <= take]
            kept[key] = target
            left -= take
        if kept:
            result[mouse] = kept
        report.append(dict(mouse=mouse,subject_number=subject_numbers[mouse],available_trials=counts[mouse],
                           retained_trials=min(budget,counts[mouse]),below_budget=counts[mouse]<budget,
                           reference_mouse=mouse in reference))
    return result, dict(reference_trial_counts=reference,reference_mean=mean,trial_budget=budget,mice=report)
