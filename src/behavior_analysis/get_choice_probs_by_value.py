import numpy as np
import pandas as pd
from .get_believed_value import compute_believed_value

BEST_VALS = [2, 3, 4]
SECOND_VALS = [1, 2, 3]

def init_matrix(fill=np.nan):
    return pd.DataFrame(
        fill,
        index=pd.Index(BEST_VALS, name="best_value"),
        columns=pd.Index(SECOND_VALS, name="second_value"),
    )

def accumulate_choice_probs(subject_trials_by_value_problem, use_believed=False):
    """
    subject_trials_by_value_problem[problem][subject_id][session_id] = session_data

    session_data["trial_variables"] is a list with a single element: a dict of
    parallel lists, one value per trial. Keys include:
      - "reward_magnitudes" : list[dict]  — arm->magnitude per trial
      - "chosen_rank"       : list[str]   — "best"/"second"/"third" per trial
      - "choice"            : list[str]   — chosen arm label per trial

    Parameters
    ----------
    use_believed : bool
        False (default) — bin by true reward magnitudes and true chosen rank.
        True            — bin by believed magnitudes (last observed value per arm
                          before this trial), with arms re-ranked by believed
                          value to determine best/second and the chosen rank.
                          Trials where fewer than 2 arms have been visited, or
                          where the chosen arm has no believed value, are skipped.
                          Tied trials split contributions evenly across best/second.

    Returns
    -------
    overall_best_prob_df   : 3x3 DataFrame  (nanmean across mice)
    overall_second_prob_df : 3x3 DataFrame
    per_mouse_best_prob    : dict[subject_id -> 3x3 DataFrame]
    per_mouse_second_prob  : dict[subject_id -> 3x3 DataFrame]
    """
    denom_mouse      = {}
    num_best_mouse   = {}
    num_second_mouse = {}

    def ensure_mouse(sid):
        if sid not in denom_mouse:
            denom_mouse[sid]      = init_matrix(fill=0.0)
            num_best_mouse[sid]   = init_matrix(fill=0.0)
            num_second_mouse[sid] = init_matrix(fill=0.0)

    for problem_num, subjects_dict in subject_trials_by_value_problem.items():
        for subject_id, sessions_dict in subjects_dict.items():
            ensure_mouse(subject_id)

            for session_id, sess in sessions_dict.items():
                tv_list = sess.get("trial_variables", [])
                if not tv_list:
                    continue
                tv = tv_list[0]

                reward_magnitudes = tv.get("reward_magnitudes", [])
                chosen_ranks      = tv.get("chosen_rank", [])
                choices           = tv.get("choice", [])

                n = min(len(reward_magnitudes), len(chosen_ranks), len(choices))
                if n == 0:
                    continue

                if use_believed:
                    believed_trials = compute_believed_value(
                        choices[:n], reward_magnitudes[:n]
                    )

                for i in range(n):
                    if use_believed:
                        trial       = believed_trials[i]
                        b           = trial["believed_best_mag"]
                        s           = trial["believed_second_mag"]
                        chosen_rank = trial["believed_chosen_rank"]
                        tied        = trial["tied"]

                        # need at least 2 seen arms to have a meaningful b/s pair
                        if trial["n_seen"] < 2:
                            continue
                        # chosen arm must have a believed rank
                        if chosen_rank is None:
                            continue
                        if b not in BEST_VALS or s not in SECOND_VALS:
                            continue

                        if tied:
                            # rank labels are arbitrary — split contributions evenly
                            denom_mouse[subject_id].loc[b, s] += 1.0
                            if chosen_rank in ("best", "second"):
                                # chosen arm was one of the two tied arms
                                num_best_mouse[subject_id].loc[b, s]   += 0.5
                                num_second_mouse[subject_id].loc[b, s] += 0.5
                            # third: contributes to denom only (0 to numerators)
                        else:
                            denom_mouse[subject_id].loc[b, s] += 1.0
                            if chosen_rank == "best":
                                num_best_mouse[subject_id].loc[b, s]   += 1.0
                            elif chosen_rank == "second":
                                num_second_mouse[subject_id].loc[b, s] += 1.0

                    else:
                        chosen_rank = chosen_ranks[i]
                        rm          = reward_magnitudes[i]
                        sorted_arms = sorted(rm.items(), key=lambda x: x[1], reverse=True)
                        if len(sorted_arms) < 2:
                            continue
                        b = sorted_arms[0][1]
                        s = sorted_arms[1][1]

                        if chosen_rank not in ("best", "second", "third"):
                            continue
                        if b not in BEST_VALS or s not in SECOND_VALS:
                            continue

                        denom_mouse[subject_id].loc[b, s] += 1.0
                        if chosen_rank == "best":
                            num_best_mouse[subject_id].loc[b, s]   += 1.0
                        elif chosen_rank == "second":
                            num_second_mouse[subject_id].loc[b, s] += 1.0

    # per-mouse probability matrices
    per_mouse_best_prob   = {}
    per_mouse_second_prob = {}
    for sid in denom_mouse:
        d = denom_mouse[sid]
        per_mouse_best_prob[sid]   = num_best_mouse[sid].divide(d.where(d > 0))
        per_mouse_second_prob[sid] = num_second_mouse[sid].divide(d.where(d > 0))

    # overall = nanmean across mice so every mouse contributes equally per cell
    def nanmean_matrices(prob_dict):
        if not prob_dict:
            return init_matrix(fill=np.nan)
        stack = np.stack([df.to_numpy(dtype=float) for df in prob_dict.values()], axis=0)
        ref   = next(iter(prob_dict.values()))
        return pd.DataFrame(np.nanmean(stack, axis=0), index=ref.index, columns=ref.columns)

    overall_best_prob   = nanmean_matrices(per_mouse_best_prob)
    overall_second_prob = nanmean_matrices(per_mouse_second_prob)

    return overall_best_prob, overall_second_prob, per_mouse_best_prob, per_mouse_second_prob