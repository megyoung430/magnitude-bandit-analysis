"""Hand-built reversal dicts for the reversal-aligned arm tests."""

TOWERS = ("A", "B", "C")
BEFORE = {"A": 3.0, "B": 2.0, "C": 1.0}  # prev ranking A, B, C
AFTER = {"A": 1.0, "B": 2.0, "C": 3.0}  # new ranking C, B, A


def onehot(chosen):
    return {t: [float(c == t) for c in chosen] for t in TOWERS}


def make_reversal(
    pre_choices, post_choices, *, before=BEFORE, after=AFTER, pre_scale=1.0, post_scale=1.0
):
    """Reversal whose pre/post windows hold the given chosen towers (one-hot) and magnitudes.

    Magnitudes are the before values in the pre window and the after values in the post window,
    times the scale arguments.
    """
    return {
        "reward_magnitudes_by_tower_before": dict(before),
        "reward_magnitudes_by_tower_after": dict(after),
        "choices_by_tower": {
            t: {"pre": onehot(pre_choices)[t], "post": onehot(post_choices)[t]} for t in TOWERS
        },
        "magnitudes_by_tower": {
            t: {
                "pre": [before[t] * pre_scale] * len(pre_choices),
                "post": [after[t] * post_scale] * len(post_choices),
            }
            for t in TOWERS
        },
    }
