"""Helpers shared by the rank-proportion tests."""

import numpy as np

RANKS = ("best", "second", "third")


def make_row(best, second, third):
    """A rank-count dict as built per block: counts, their total and the proportions."""
    total = best + second + third
    return {
        "best": best,
        "second": second,
        "third": third,
        "total": total,
        "best_prop": best / total if total else np.nan,
        "second_prop": second / total if total else np.nan,
        "third_prop": third / total if total else np.nan,
    }
