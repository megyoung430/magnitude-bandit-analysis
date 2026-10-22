"""Wilcoxon signed-rank tests between the rank proportions, Holm-corrected, with stars."""

import math

import numpy as np
import pytest
from scipy.stats import wilcoxon

from src.behavior_analysis.rank_significance import (
    NOT_SIGNIFICANT,
    describe_rank_tests,
    holm_correction,
    minimum_exact_p,
    rank_comparison_tests,
    significance_stars,
    wilcoxon_pair,
)

# eight mice, distinct non-zero differences: scipy uses the exact distribution
BEST = [0.61, 0.72, 0.55, 0.81, 0.66, 0.73, 0.52, 0.90]
SECOND = [0.25, 0.20, 0.30, 0.12, 0.21, 0.19, 0.33, 0.06]
THIRD = [0.14, 0.08, 0.15, 0.16, 0.13, 0.10, 0.15, 0.09]


def per_mouse(best, second, third):
    return {
        f"m{i}": {"best": b, "second": s, "third": t}
        for i, (b, s, t) in enumerate(zip(best, second, third, strict=True))
    }


# --- Holm -----------------------------------------------------------------------------------


def test_holm_on_known_values_is_monotone_in_the_original_order():
    np.testing.assert_allclose(holm_correction([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_holm_caps_at_one():
    np.testing.assert_allclose(holm_correction([0.5, 0.6, 0.2]), [1.0, 1.0, 0.6])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # sorted 0.001, 0.04, 0.2 -> x3, x2, x1 = 0.003, 0.08, 0.2
        ([0.001, 0.2, 0.04], [0.003, 0.2, 0.08]),
        # equal p-values all take the largest step-down value
        ([0.3, 0.3, 0.3], [0.9, 0.9, 0.9]),
        # 0.02 x3 = 0.06, 0.049 x2 = 0.098, 0.9 x1 = 0.9
        ([0.049, 0.02, 0.9], [0.098, 0.06, 0.9]),
    ],
)
def test_holm_on_worked_examples(raw, expected):
    np.testing.assert_allclose(holm_correction(raw), expected)


def test_holm_leaves_untested_comparisons_out_of_the_family():
    corrected = holm_correction([0.01, np.nan, 0.04])
    np.testing.assert_allclose(corrected, [0.02, np.nan, 0.04])


# --- stars ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("p", "stars"),
    [
        (0.0009, "***"),
        (0.001, "**"),
        (0.0099, "**"),
        (0.01, "*"),
        (0.0499, "*"),
        (0.05, NOT_SIGNIFICANT),
        (0.8, NOT_SIGNIFICANT),
        (1.0, NOT_SIGNIFICANT),
    ],
)
def test_star_thresholds_include_their_boundaries_on_the_less_significant_side(p, stars):
    assert significance_stars(p) == stars


def test_an_untested_comparison_gets_no_star():
    assert significance_stars(np.nan) is None


# --- one comparison -------------------------------------------------------------------------


def test_the_wrapper_matches_scipy_and_records_the_exact_method():
    result = wilcoxon_pair(BEST, SECOND)
    expected = wilcoxon(BEST, SECOND)
    assert result.n == 8
    assert result.statistic == expected.statistic
    assert result.p == pytest.approx(expected.pvalue)
    assert result.method == "exact"
    assert wilcoxon(BEST, SECOND, method="exact").pvalue == pytest.approx(result.p)


def test_a_zero_difference_switches_scipy_to_its_permutation_distribution():
    second = [*SECOND[:-1], BEST[-1]]
    result = wilcoxon_pair(BEST, second)
    assert result.method == "permutation"
    assert result.n == 8 and result.n_nonzero == 7
    assert result.p == pytest.approx(wilcoxon(BEST, second).pvalue)


def test_mice_without_both_values_are_dropped():
    best = [*BEST[:-1], np.nan]
    result = wilcoxon_pair(best, SECOND)
    assert result.n == 7
    assert result.p == pytest.approx(wilcoxon(BEST[:-1], SECOND[:-1]).pvalue)


def test_fewer_than_two_mice_are_not_tested():
    result = wilcoxon_pair([0.6], [0.3])
    assert not result.tested and math.isnan(result.p)
    assert result.reason == "fewer than 2 mice"


def test_all_zero_differences_are_not_tested():
    result = wilcoxon_pair([0.3, 0.4, 0.5], [0.3, 0.4, 0.5])
    assert not result.tested and math.isnan(result.p)
    assert result.reason == "all differences are zero"


# --- the three comparisons ------------------------------------------------------------------


def test_the_three_comparisons_are_holm_corrected_and_starred_on_the_corrected_p():
    tests = rank_comparison_tests(per_mouse(BEST, SECOND, THIRD))
    assert list(tests) == ["best_vs_second", "best_vs_third", "second_vs_third"]
    raw = [tests[k].p for k in tests]
    np.testing.assert_allclose([tests[k].p_holm for k in tests], holm_correction(raw))
    for test in tests.values():
        assert test.stars == significance_stars(test.p_holm)
    # best > second in every mouse: the smallest exact two-sided p of 8 mice, times 3
    assert tests["best_vs_second"].p == pytest.approx(2 / 2**8)
    assert tests["best_vs_second"].p_holm == pytest.approx(3 * 2 / 2**8)
    assert tests["best_vs_second"].stars == "*"


def test_a_mouse_without_a_value_is_left_out_of_that_comparison_only():
    units = per_mouse(BEST, SECOND, THIRD)
    units["m0"]["third"] = np.nan
    tests = rank_comparison_tests(units)
    assert tests["best_vs_second"].n == 8
    assert tests["best_vs_third"].n == 7


def test_one_mouse_gives_three_untested_comparisons_and_no_stars():
    tests = rank_comparison_tests(per_mouse(BEST[:1], SECOND[:1], THIRD[:1]))
    assert all(not t.tested and t.stars is None and math.isnan(t.p_holm) for t in tests.values())


def test_minimum_exact_p():
    assert minimum_exact_p(5) == 0.0625
    assert minimum_exact_p(1) == 1.0


# --- the caption ----------------------------------------------------------------------------


def test_the_caption_names_the_test_correction_statistics_and_stars():
    tests = rank_comparison_tests(per_mouse(BEST, SECOND, THIRD))
    text = describe_rank_tests(tests)
    bs = tests["best_vs_second"]
    assert text.startswith(
        "Two-sided Wilcoxon signed-rank tests across mice (n = 8), "
        "Holm-corrected for 3 comparisons: "
    )
    assert f"Best vs Second W = {bs.statistic:g}, p = 0.008 (corrected 0.023, *)" in text
    assert "Second vs Third W = " in text
    assert "exact" in text
    assert text.endswith("Stars: * p < 0.05, ** p < 0.01, *** p < 0.001; n.s., not significant.")
    assert "p(" not in text
    assert "smallest possible" not in text


def test_with_five_mice_the_caption_says_no_comparison_can_be_significant():
    tests = rank_comparison_tests(per_mouse(BEST[:5], SECOND[:5], THIRD[:5]))
    assert all(t.stars == NOT_SIGNIFICANT for t in tests.values())
    assert (
        "With n = 5 mice the smallest possible p is 0.0625, so no comparison can be significant."
        in describe_rank_tests(tests)
    )


def test_with_six_mice_the_caption_names_the_holm_bound():
    tests = rank_comparison_tests(per_mouse(BEST[:6], SECOND[:6], THIRD[:6]))
    assert (
        "With n = 6 mice the smallest possible p is 0.03125 (0.09375 after Holm correction for "
        "3 comparisons), so no comparison can be significant." in describe_rank_tests(tests)
    )


def test_untested_comparisons_say_why_in_the_caption():
    tests = rank_comparison_tests(per_mouse(BEST[:1], SECOND[:1], THIRD[:1]))
    text = describe_rank_tests(tests)
    assert "not tested (fewer than 2 mice)" in text
    assert "W =" not in text
