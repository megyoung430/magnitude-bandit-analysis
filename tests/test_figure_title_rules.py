"""Every figure a cohort run and an all-cohorts run save keeps short titles and has a caption.

A title is its heading, plus at most one statistics line ("mean ± SE across mice, n = 13 mice");
everything else is in ``<name>.caption.txt`` next to the figure, which every figure has.
"""

import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

import src.pipelines.cohort_analysis as cohort_analysis
from src.behavior_visualization import plot_helpers
from src.pipelines.all_cohorts_drifting import run_all_cohorts_drifting
from src.pipelines.cohort_analysis import run_cohort
from tests.all_cohorts_fixtures import synthetic_cohort
from tests.cohort_fixtures import blocked_then_two_drifting

STATS_LINE = re.compile(
    r"(mean|median|proportion)( ± (SE|SD))? across [a-z ]+"
    r"(, n = \d+ [a-z]+( [a-z]+)*)+"
)


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    """``[(base path, suptitle, [axes titles])]`` of every figure the two runs save."""
    root = tmp_path_factory.mktemp("title-rules")
    records = []
    real = plot_helpers._write_figure_files

    def spy(fig, base, formats, dpi):
        titles = [ax.get_title() for ax in fig.axes if ax.get_visible() and ax.get_title()]
        records.append((Path(str(base)), fig.get_suptitle(), titles))
        return real(fig, base, formats, dpi)

    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patch.setattr(plot_helpers, "_write_figure_files", spy)
        # the fixture's sessions carry no sub-NN folder to read the subject numbers from
        patch.setattr(
            cohort_analysis,
            "subject_numbers_from_sessions",
            lambda trials: {
                m: i + 1 for i, m in enumerate(sorted({m for p in trials.values() for m in p}))
            },
        )
        run_cohort(
            3, results_root=root, trials_by_problem=blocked_then_two_drifting(), exclude_mice=["m2"]
        )
        cohorts = {
            1: synthetic_cohort(("b1", "b2"), seed=1),
            3: synthetic_cohort(("a1", "a2", "a3")),
        }
        run_all_cohorts_drifting(
            [1, 3], results_root=root, trials_by_cohort=cohorts, exclude_mice=["a3"]
        )
    plt.close("all")
    return records


def test_the_runs_save_figures_of_every_kind(saved):
    parts = {part for base, _, _ in saved for part in base.parts}
    expected = {"choice-stats", "reversal-aligned-arms", "value-change", "sessions-per-problem"}
    assert expected | {"value-psychometrics", "port-preference", "long-pokes"} <= parts
    assert len(saved) > 100


def test_every_title_is_a_heading_and_at_most_a_statistics_line(saved):
    for base, suptitle, titles in saved:
        for title in [suptitle, *titles]:
            lines = title.split("\n") if title else []
            assert len(lines) <= 2, (base.name, title)
            if len(lines) == 2:
                assert STATS_LINE.fullmatch(lines[1]), (base.name, title)
            assert not any(line.startswith("(") for line in lines), (base.name, title)


def test_every_saved_figure_has_its_caption_file(saved):
    for base, _, _ in saved:
        assert plot_helpers.caption_path(base).is_file(), base
        for fmt in plot_helpers.FIGURE_FORMATS:
            assert Path(f"{base}.{fmt}").is_file(), (base, fmt)


def test_the_extra_set_names_its_mice_in_every_caption_and_the_file_names_are_unchanged(saved):
    extra = [base for base, _, _ in saved if "without" in base.parent.as_posix()]
    assert extra
    for base in extra:
        assert "without" in plot_helpers.caption_path(base).read_text().splitlines()[0], base
    names = {base.name for base, _, _ in saved}
    assert "Sessions per Drifting Problem" in names
    assert (
        "Choice Probabilities Around Block Changes (All) (Arms Labelled by Rank Before) (pooled)"
        in names
    )
