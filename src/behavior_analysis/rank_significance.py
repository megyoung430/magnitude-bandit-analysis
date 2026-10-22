"""Wilcoxon signed-rank tests between the best, second and third rank proportions.

The paper-ready rank figures mark these comparisons with significance stars. Each mouse is one
paired observation (its proportion of choices of each rank), the three pairwise comparisons are
two-sided Wilcoxon signed-rank tests, and their p-values are Holm-corrected as one family. The
diagnostic figures and tables keep the paired t-tests of :mod:`.get_diagnostic_p_value`.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
from scipy.stats import wilcoxon

# (key, first rank, second rank, label), in the order the comparisons are reported
COMPARISONS = (
    ("best_vs_second", "best", "second", "Best vs Second"),
    ("best_vs_third", "best", "third", "Best vs Third"),
    ("second_vs_third", "second", "third", "Second vs Third"),
)
# (upper bound, mark): a corrected p below the bound gets the mark; checked in this order
STAR_THRESHOLDS = ((0.001, "***"), (0.01, "**"), (0.05, "*"))
NOT_SIGNIFICANT = "n.s."
ALPHA = STAR_THRESHOLDS[-1][0]
# scipy's method="auto" uses the normal approximation above this many pairs
EXACT_MAX_N = 50
# scipy's method="auto" enumerates all sign flips (an exact permutation test) for ties or zero
# differences up to this many pairs
PERMUTATION_MAX_N = 13
STARS_LEGEND = "Stars: * p < 0.05, ** p < 0.01, *** p < 0.001; n.s., not significant."
METHOD_TEXT = {
    "exact": "exact p-values",
    "permutation": "exact permutation p-values (ties or zero differences present)",
    "asymptotic": "normal-approximation p-values",
}


@dataclass(frozen=True)
class PairedTest:
    """One Wilcoxon signed-rank comparison across mice.

    Attributes:
        n (int): Mice with a finite value for both ranks.
        n_nonzero (int): Of those, the mice whose difference is not zero (scipy's default
            ``zero_method="wilcox"`` drops zero differences before ranking).
        statistic (float): Scipy's two-sided W, the smaller of the two signed-rank sums; NaN
            when not tested.
        p (float): Two-sided p-value; NaN when not tested.
        method (str | None): ``"exact"``, ``"permutation"`` or ``"asymptotic"``, as scipy's
            ``method="auto"`` chose; ``None`` when not tested.
        reason (str | None): Why the comparison was not tested; ``None`` when it was.
        p_holm (float): Holm-corrected p within the comparisons of one figure; NaN until
            corrected or when not tested.
        stars (str | None): Mark of *p_holm* (see :func:`significance_stars`).
    """

    n: int
    n_nonzero: int
    statistic: float
    p: float
    method: str | None
    reason: str | None = None
    p_holm: float = math.nan
    stars: str | None = None

    @property
    def tested(self) -> bool:
        return self.reason is None


def significance_stars(p: float) -> str | None:
    """``***`` for p < 0.001, ``**`` < 0.01, ``*`` < 0.05, else ``n.s.``; ``None`` for NaN."""
    if math.isnan(p):
        return None
    return next((mark for bound, mark in STAR_THRESHOLDS if p < bound), NOT_SIGNIFICANT)


def holm_correction(pvalues: Sequence[float]) -> np.ndarray:
    """Holm–Bonferroni adjusted p-values, in the order given.

    The family is the finite p-values; NaN entries (comparisons not tested) stay NaN and do
    not count towards its size. The adjusted values are capped at 1 and never decrease with the
    raw ones.

    Args:
        pvalues (Sequence[float]): Raw p-values of one family.

    Returns:
        np.ndarray: The adjusted p-values.
    """
    raw = np.asarray(pvalues, dtype=float)
    adjusted = np.full(raw.shape, np.nan)
    tested = np.flatnonzero(np.isfinite(raw))
    order = tested[np.argsort(raw[tested], kind="stable")]
    m = len(order)
    stepped = raw[order] * (m - np.arange(m))
    adjusted[order] = np.minimum(np.maximum.accumulate(stepped), 1.0)
    return adjusted


def scipy_auto_method(differences: np.ndarray) -> str:
    """The method scipy's ``wilcoxon(method="auto")`` uses for these paired *differences*.

    Scipy does not return it, so its rule is restated here (scipy 1.13+): the normal
    approximation above 50 pairs; otherwise the exact distribution, unless there are ties or
    zero differences, then an exact permutation test up to 13 pairs and the approximation above.
    """
    n = len(differences)
    if n > EXACT_MAX_N:
        return "asymptotic"
    magnitudes = np.abs(differences[differences != 0])
    has_ties_or_zeros = len(magnitudes) < n or len(np.unique(magnitudes)) < len(magnitudes)
    if not has_ties_or_zeros:
        return "exact"
    return "permutation" if n <= PERMUTATION_MAX_N else "asymptotic"


def wilcoxon_pair(first: Sequence[float], second: Sequence[float]) -> PairedTest:
    """Two-sided Wilcoxon signed-rank test of paired values, one pair per mouse.

    Pairs with a non-finite value are dropped. Fewer than 2 pairs, or only zero differences,
    give an untested comparison (NaN p and a reason) instead of scipy's warning and NaN.

    Args:
        first (Sequence[float]): One value per mouse.
        second (Sequence[float]): The paired value of each mouse, in the same order.

    Returns:
        PairedTest: The test, not yet Holm-corrected.
    """
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    differences = a - b
    n, n_nonzero = len(differences), int(np.count_nonzero(differences))
    if n < 2:
        return PairedTest(n, n_nonzero, math.nan, math.nan, None, reason="fewer than 2 mice")
    if n_nonzero == 0:
        return PairedTest(n, 0, math.nan, math.nan, None, reason="all differences are zero")
    result = wilcoxon(a, b)
    return PairedTest(
        n,
        n_nonzero,
        float(result.statistic),
        float(result.pvalue),
        scipy_auto_method(differences),
    )


def rank_comparison_tests(per_mouse: Mapping[object, Mapping[str, float]]) -> dict:
    """Best vs second, best vs third and second vs third across mice, Holm-corrected together.

    Args:
        per_mouse (Mapping[object, Mapping[str, float]]): ``{mouse: {"best", "second",
            "third"}}`` proportions, as from
            :func:`src.behavior_analysis.rank_proportions.proportions_by_mouse`.

    Returns:
        dict: ``{comparison key: PairedTest}`` in the order of :data:`COMPARISONS`, each with
        its Holm-corrected p and stars.
    """
    units = list(per_mouse.values())
    raw = [
        wilcoxon_pair([u[first] for u in units], [u[second] for u in units])
        for _, first, second, _ in COMPARISONS
    ]
    corrected = holm_correction([t.p for t in raw])
    return {
        key: replace(test, p_holm=float(p), stars=significance_stars(float(p)))
        for (key, *_), test, p in zip(COMPARISONS, raw, corrected, strict=True)
    }


def minimum_exact_p(n: int) -> float:
    """Smallest two-sided p an exact signed-rank test of *n* non-zero differences can give."""
    return min(1.0, 2 / 2**n)


# --- caption text ---------------------------------------------------------------------------


def _format_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def _phrase(label: str, test: PairedTest, show_n: bool) -> str:
    name = f"{label} (n = {test.n})" if show_n else label
    if not test.tested:
        return f"{name} not tested ({test.reason})"
    p = _format_p(test.p)
    p_text = f"p {p}" if p.startswith("<") else f"p = {p}"
    holm = _format_p(test.p_holm)
    return f"{name} W = {test.statistic:g}, {p_text} (corrected {holm}, {test.stars})"


def _unreachable_sentence(tested: list[PairedTest]) -> str | None:
    """Why no comparison can be significant with this few mice, or ``None`` if one can."""
    if any(t.method == "asymptotic" for t in tested):
        return None
    n = max(t.n_nonzero for t in tested)
    smallest = minimum_exact_p(n)
    after_holm = smallest * len(tested)
    if after_holm < ALPHA:
        return None
    holm = (
        ""
        if smallest >= ALPHA or len(tested) == 1
        else f" ({after_holm:.4g} after Holm correction for {len(tested)} comparisons)"
    )
    return (
        f"With n = {n} mice the smallest possible p is {smallest:.4g}{holm}, "
        "so no comparison can be significant."
    )


def describe_rank_tests(tests: Mapping[str, PairedTest]) -> str:
    """The caption sentences of :func:`rank_comparison_tests`: tests, statistics and stars.

    Args:
        tests (Mapping[str, PairedTest]): As returned by :func:`rank_comparison_tests`.

    Returns:
        str: The test, the number of mice, each comparison's W, raw and corrected p and mark,
        the p-value method, why nothing can be significant when that is so, and the star key.
    """
    labels = {key: label for key, *_, label in COMPARISONS}
    tested = [t for t in tests.values() if t.tested]
    counts = {t.n for t in tests.values()}
    one_n = len(counts) == 1
    n_text = f" (n = {counts.pop()})" if one_n else ""
    family = len(tested)
    if family:
        plural = "s" if family != 1 else ""
        head = (
            f"Two-sided Wilcoxon signed-rank tests across mice{n_text}, "
            f"Holm-corrected for {family} comparison{plural}: "
        )
    else:
        head = f"No Wilcoxon signed-rank test across mice{n_text}: "
    body = "; ".join(_phrase(labels[k], t, not one_n) for k, t in tests.items())
    sentences = [head + body + "."]
    methods = sorted({t.method for t in tested})
    if methods:
        described = ", ".join(METHOD_TEXT[m] for m in methods)
        sentences.append(f"W is the smaller signed-rank sum; {described}.")
    unreachable = _unreachable_sentence(tested) if tested else None
    if unreachable:
        sentences.append(unreachable)
    sentences.append(STARS_LEGEND)
    return " ".join(sentences)
