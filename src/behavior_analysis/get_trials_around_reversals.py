import numpy as np

def aggregate_reversal_choice_probabilities(reversal_windows, pre=5, post=15):
    """
    Returns:
      x : array of relative trial indices
      probs : dict with keys
        - "prev_best"
        - "next_best"
        - "third"
    """

    T = pre + post
    sums = {
        "prev_best": np.zeros(T),
        "next_best": np.zeros(T),
        "third":     np.zeros(T),
    }
    counts = np.zeros(T)

    for subj, reversals in reversal_windows.items():
        for r in reversals:
            prev_best, next_best, third = classify_towers(r)

            def cat(tower):
                return (r["choices_by_tower"][tower]["pre"] + r["choices_by_tower"][tower]["post"])

            prev_arr = cat(prev_best)
            next_arr = cat(next_best)
            third_arr = cat(third)

            for t in range(len(prev_arr)):
                sums["prev_best"][t] += prev_arr[t]
                sums["next_best"][t] += next_arr[t]
                sums["third"][t]     += third_arr[t]
                counts[t] += 1

    probs = {k: sums[k] / counts for k in sums}
    x = np.arange(-pre, post)
    return x, probs

# ========== Classifying Towers at Good Reversals ==========
def classify_towers(reversal):
    """
    Returns:
      prev_best, next_best, third
    """
    before = reversal["reward_magnitudes_by_tower_before"]
    after  = reversal["reward_magnitudes_by_tower_after"]

    prev_best = max(before, key=before.get)
    next_best = max(after, key=after.get)

    towers = set(before.keys())
    third = list(towers - {prev_best, next_best})[0]

    return prev_best, next_best, third