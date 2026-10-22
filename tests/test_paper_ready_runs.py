"""``paper_ready=True`` on the two pooled drifting runners, with the real analyses.

The synthetic cohorts (``tests/cohort_fixtures.py``, ``tests/all_cohorts_fixtures.py``) are run
twice, with and without the option. The normal outputs must be identical; the extra
``paper-ready/`` tree must hold pdf, svg, png and a caption for every pooled figure, in the
same sub-folders, including the set without excluded mice.
"""

import hashlib
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from src.behavior_visualization import plot_helpers
from src.pipelines.all_cohorts_drifting import run_all_cohorts_drifting
from src.pipelines.cohort_analysis import run_cohort
from tests.all_cohorts_fixtures import synthetic_cohort
from tests.cohort_fixtures import blocked_then_two_drifting

PAPER = "paper-ready"
FIGURE_EXTENSIONS = {".png", ".pdf", ".svg"}
pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module", autouse=True)
def low_resolution():
    """Rasterise the paper PNGs at 50 dpi: the files only need to exist, and 600 dpi is slow."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(plot_helpers, "PAPER_DPI", 50)
        yield


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def digests(root: Path, *, skip: str | None = None) -> dict[str, str]:
    """``{relative path: sha256}`` of the CSV and PNG files under *root* (not PDFs: dated)."""
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.suffix in {".csv", ".png"} and not (skip and skip in p.relative_to(root).parts)
    }


def stems(folder: Path, suffix: str) -> set[str]:
    return {str(p.relative_to(folder).with_suffix("")) for p in folder.rglob(f"*{suffix}")}


def assert_complete_paper_set(normal: Path, paper: Path) -> None:
    """Every normal figure has pdf, svg, png and a caption in the same sub-folder of *paper*."""
    expected = stems(normal, ".png")
    assert expected
    for suffix in (".png", ".pdf", ".svg"):
        assert stems(paper, suffix) == expected
    captions = {
        str(p.relative_to(paper))[: -len(".caption.txt")] for p in paper.rglob("*.caption.txt")
    }
    assert captions == expected


# --- the cohort runner ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def cohort_runs(tmp_path_factory):
    roots = {}
    for paper_ready in (False, True):
        root = tmp_path_factory.mktemp(f"cohort-{paper_ready}")
        run_cohort(
            1,
            results_root=root,
            trials_by_problem=blocked_then_two_drifting(),
            pooled_only=True,
            exclude_mice=["m2"],
            paper_ready=paper_ready,
        )
        roots[paper_ready] = root
    return roots


def test_the_normal_outputs_are_identical_with_and_without_paper_ready(cohort_runs):
    plain = digests(cohort_runs[False])
    with_paper = digests(cohort_runs[True], skip=PAPER)
    assert plain and plain == with_paper


def test_nothing_is_written_to_paper_ready_unless_asked(cohort_runs):
    assert not (cohort_runs[False] / "figures" / "open-field" / PAPER).exists()


@pytest.mark.parametrize("folder", ["all-drifting-problems", "all-drifting-problems-without-m2"])
def test_every_pooled_figure_has_a_paper_copy_in_the_same_layout(cohort_runs, folder):
    figures = cohort_runs[True] / "figures" / "open-field"
    assert_complete_paper_set(
        figures / "cohort-01" / "drifting" / folder,
        figures / PAPER / "cohort-01" / "drifting" / folder,
    )


def test_the_excluded_set_names_the_exclusion_in_its_captions(cohort_runs):
    folder = (
        cohort_runs[True]
        / "figures/open-field"
        / PAPER
        / "cohort-01/drifting/all-drifting-problems-without-m2"
    )
    captions = [p.read_text() for p in folder.rglob("*.caption.txt")]
    assert captions and all("without m2" in c for c in captions)


# --- the all-cohorts runner -----------------------------------------------------------------


@pytest.fixture(scope="module")
def all_cohorts_runs(tmp_path_factory):
    imported = {
        1: synthetic_cohort(("a1", "a2", "a3"), seed=1),
        3: synthetic_cohort(("b1", "b2"), seed=3),
    }
    roots = {}
    for paper_ready in (False, True):
        root = tmp_path_factory.mktemp(f"all-{paper_ready}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run_all_cohorts_drifting(
                [1, 3],
                results_root=root,
                trials_by_cohort=imported,
                exclude_mice=["b2"],
                paper_ready=paper_ready,
            )
        roots[paper_ready] = root
    return roots


def test_all_cohorts_normal_outputs_are_identical(all_cohorts_runs):
    plain = digests(all_cohorts_runs[False])
    with_paper = digests(all_cohorts_runs[True], skip=PAPER)
    assert plain and plain == with_paper


@pytest.mark.parametrize(
    ("normal", "paper"),
    [("all-cohorts", "all-cohorts"), ("all-cohorts-without-b2", "all-cohorts-without-b2")],
)
def test_all_cohorts_figures_have_paper_copies(all_cohorts_runs, normal, paper):
    figures = all_cohorts_runs[True] / "figures" / "open-field"
    assert_complete_paper_set(figures / normal, figures / PAPER / paper)


# --- the rank figures: stars in the paper copies, p-values in the normal ones ---------------

RANK_FIGURE = "Rank Proportions (True Value) (Pool Trials True) by Average Across Mice"


def rank_files(root: Path, suffix: str) -> list[Path]:
    return sorted(root.rglob(f"{RANK_FIGURE}{suffix}"))


@pytest.mark.parametrize("runs", ["cohort_runs", "all_cohorts_runs"])
def test_paper_rank_captions_give_the_wilcoxon_tests_instead_of_t_test_p_values(runs, request):
    paper = request.getfixturevalue(runs)[True] / "figures" / "open-field" / PAPER
    captions = [p.read_text() for p in rank_files(paper, ".caption.txt")]
    assert len(captions) == 2  # the plain pool and the set without the excluded mouse
    for caption in captions:
        # a set left with one mouse says so instead of testing
        tested = "Two-sided Wilcoxon signed-rank tests across mice (n = " in caption
        assert tested or "not tested (fewer than 2 mice)" in caption
        if tested:
            assert "Holm-corrected" in caption and "Stars: * p < 0.05" in caption
        assert "p(Best vs Second)" not in caption and "p(Second vs Third)" not in caption
    assert any("Two-sided Wilcoxon" in caption for caption in captions)


@pytest.mark.parametrize("runs", ["cohort_runs", "all_cohorts_runs"])
def test_paper_rank_figures_draw_the_marks(runs, request):
    paper = request.getfixturevalue(runs)[True] / "figures" / "open-field" / PAPER
    for svg in rank_files(paper, ".svg"):
        tested = "Two-sided" in svg.with_suffix(".caption.txt").read_text()
        text = svg.read_text()
        assert tested == ("n.s." in text or "*</text>" in text)
        assert "p(Best" not in text


def test_the_all_cohorts_paper_caption_says_five_mice_cannot_reach_significance(
    all_cohorts_runs,
):
    paper = all_cohorts_runs[True] / "figures" / "open-field" / PAPER / "all-cohorts"
    (caption,) = [p.read_text() for p in rank_files(paper, ".caption.txt")]
    assert "With n = 5 mice the smallest possible p is 0.0625" in caption


def test_the_normal_rank_figure_keeps_its_t_test_p_values(monkeypatch, tmp_path):
    from src.behavior_analysis.get_diagnostic_p_value import (
        pvalue_paired_t_best_vs_second_vs_third,
    )
    from src.behavior_analysis.rank_proportions import proportions_by_mouse
    from src.behavior_analysis.rank_significance import rank_comparison_tests
    from src.pipelines.cohort_analysis import plot_pooled_drifting_rank
    from tests.plot_capture import capture_figures
    from tests.rank_rows import make_row

    counts = {
        "m1": [make_row(6, 3, 1), make_row(9, 2, 1)],
        "m2": [make_row(8, 1, 1), make_row(7, 2, 1)],
        "m3": [make_row(9, 1, 0), make_row(5, 4, 1)],
    }
    ranks = {
        "true": {
            True: {
                "problems": [1],
                "counts": counts,
                "p_values": pvalue_paired_t_best_vs_second_vs_third(counts, True),
                "significance": rank_comparison_tests(
                    proportions_by_mouse(counts, pool_trials_within_subject=True)
                ),
            }
        }
    }
    capture = capture_figures(monkeypatch)
    plot_pooled_drifting_rank(ranks, tmp_path)
    (figure,) = {s.figure for s in capture.saved}
    texts = [t.get_text() for ax in figure.axes for t in ax.texts]
    assert any(t.startswith("p(Best vs Second)") for t in texts)
    assert not any(line.get_gid() for ax in figure.axes for line in ax.lines)
