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
