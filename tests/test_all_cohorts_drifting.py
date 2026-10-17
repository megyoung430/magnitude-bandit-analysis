"""The all-cohorts drifting pipeline: pooling, the output tree, and what it leaves out.

Two synthetic cohorts (1 with mice a1-a3, 3 with b1-b2) each run a blocked problem and two
drifting problems (``tests/all_cohorts_fixtures.py``). The blocked problem must never reach the
pool, and both cohorts' drifting problems are numbered 1 and 2 within the type, so only the
cohort in the key keeps them apart.
"""

import importlib.util
import inspect
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from src.behavior_analysis.get_choice_probs_by_value import count_choice_probs
from src.pipelines import all_cohorts_drifting as pipeline
from src.pipelines.all_cohorts_drifting import (
    drifting_problems,
    pool_cohorts,
    pooled_scope_label,
    run_all_cohorts_drifting,
)
from src.pipelines.cohort_analysis import (
    DRIFTING_SELECTION,
    plot_pooled_drifting_rank,
    pooled_drifting_rank_results,
)
from src.pipelines.open_field import (
    POOLED_DRIFTING_LABEL,
    run_drifting_value_analyses,
)
from src.pipelines.port_preference import run_port_preference_analysis
from src.pipelines.value_change_analysis import run_value_change_analysis
from src.pipelines.value_psychometrics import run_value_psychometric_analysis
from tests.all_cohorts_fixtures import DRIFTING_NAME, synthetic_cohort
from tests.plot_capture import capture_figures

SUBFOLDERS = {
    "choice-stats",
    "value-psychometrics",
    "reversal-aligned-arms",
    "value-change",
    "port-preference",
}
MICE_A, MICE_B = ("a1", "a2", "a3"), ("b1", "b2")
ALL_MICE = {*MICE_A, *MICE_B}
RUN_AT = datetime(2026, 10, 16, 12, 30, 5)
TABLES = {"reversal_aligned_arms_counts", "value_change_events", "value_change_bin_counts"}


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def imported():
    """``{cohort id: {overall problem: {mouse: sessions}}}``."""
    return {1: synthetic_cohort(MICE_A, seed=1), 3: synthetic_cohort(MICE_B, seed=3)}


@pytest.fixture(scope="module")
def pooled(imported):
    return pool_cohorts({c: drifting_problems(t) for c, t in imported.items()})


def figure_files(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


@pytest.fixture(scope="module")
def written(imported, tmp_path_factory):
    """One real run with files: ``(results_root, result)``."""
    results_root = tmp_path_factory.mktemp("results")
    result = run_all_cohorts_drifting(
        [1, 3], results_root=results_root, trials_by_cohort=imported, run_at=RUN_AT
    )
    return results_root, result


@pytest.fixture(scope="module")
def cohort_runner_pool(pooled, tmp_path_factory):
    """What the cohort runner writes into ``all-drifting-problems/`` for the same problems."""
    out = tmp_path_factory.mktemp("reference")
    ranks = pooled_drifting_rank_results(
        pooled.problems,
        DRIFTING_SELECTION.rank_value_bases,
        DRIFTING_SELECTION.rank_pooling_modes,
    )
    plot_pooled_drifting_rank(ranks, out / "choice-stats")
    run_drifting_value_analyses(
        pooled.problems, out, DRIFTING_SELECTION, arm_scope_label=POOLED_DRIFTING_LABEL
    )
    return out


# --- pooling ---------------------------------------------------------------------------


def test_the_scope_label_names_the_cohorts():
    assert pooled_scope_label([1, 3, 4]) == "all drifting problems, cohorts 1, 3, 4"
    assert pooled_scope_label([3]) == "all drifting problems, cohort 3"


def test_only_drifting_problems_are_pooled_and_the_blocked_one_is_ignored(imported, pooled):
    assert set(imported[1]) == {1, 2, 3}  # the blocked problem is overall problem 1
    assert set(pooled.problems) == {(1, 1), (1, 2), (3, 1), (3, 2)}
    for subjects in pooled.problems.values():
        for sessions in subjects.values():
            assert {s["task_name"] for s in sessions.values()} == {DRIFTING_NAME}


def test_drifting_problems_are_numbered_within_the_type(imported):
    drifting = drifting_problems(imported[1])
    assert set(drifting) == {1, 2}
    assert set(drifting[1]) == set(MICE_A)


def test_problems_of_different_cohorts_with_the_same_number_stay_separate(imported, pooled):
    first_a, first_b = pooled.problems[(1, 1)], pooled.problems[(3, 1)]
    assert set(first_a) == set(MICE_A) and set(first_b) == set(MICE_B)
    assert first_a["a1"] is not first_b["b1"]
    assert sum(len(s) for s in pooled.problems.values()) == 2 * len(ALL_MICE)


def test_mice_per_cohort_are_counted_over_the_pooled_problems(pooled):
    assert pooled.mice_by_cohort == {1: list(MICE_A), 3: list(MICE_B)}


def test_a_mouse_in_two_cohorts_raises_naming_it_and_both_cohorts(imported):
    clash = {
        1: drifting_problems(imported[1]),
        3: drifting_problems(synthetic_cohort(["a2", "b1"])),
    }
    with pytest.raises(ValueError, match=r"a2.*cohorts? 1.*3|a2.*cohort-01.*cohort-03"):
        pool_cohorts(clash)


def test_a_cohort_without_drifting_problems_raises_naming_it(imported):
    blocked_only = {1: imported[3][1]}
    with pytest.raises(ValueError, match="cohort 3"):
        run_all_cohorts_drifting(
            [1, 3],
            results_root=Path("unused"),
            trials_by_cohort={1: imported[1], 3: blocked_only},
        )


@pytest.mark.parametrize("cohorts", [[], [1, 1], [0], [1, -2]])
def test_bad_cohort_lists_raise(cohorts, imported):
    with pytest.raises(ValueError):
        run_all_cohorts_drifting(cohorts, results_root=Path("unused"), trials_by_cohort=imported)


def test_cohorts_missing_from_the_injected_trials_raise(imported):
    with pytest.raises(ValueError, match="cohort 4"):
        run_all_cohorts_drifting(
            [1, 4], results_root=Path("unused"), trials_by_cohort={1: imported[1]}
        )


# --- problems never mix ----------------------------------------------------------------


def test_believed_values_reset_at_every_problem_boundary_across_cohorts(pooled):
    together = count_choice_probs(pooled.problems, use_believed=True)
    for mouse in ALL_MICE:
        alone = [
            count_choice_probs({problem: subjects}, use_believed=True)[mouse].denom
            for problem, subjects in pooled.problems.items()
            if mouse in subjects
        ]
        assert len(alone) == 2  # the mouse's two problems of its own cohort
        assert together[mouse].denom.to_numpy() == pytest.approx(sum(alone).to_numpy())


def test_value_change_events_never_span_problems(monkeypatch, pooled, tmp_path):
    capture_figures(monkeypatch)  # only the tables matter here
    together = run_value_change_analysis(pooled.problems, tmp_path / "all")["contamination_table"]
    per_problem = [
        run_value_change_analysis({p: s}, tmp_path / str(p))["contamination_table"]
        for p, s in pooled.problems.items()
    ]
    assert together["n_events"].sum() > 0
    assert together["n_events"].tolist() == [
        sum(t["n_events"].iloc[i] for t in per_problem) for i in range(len(together))
    ]


def test_rank_windows_are_built_inside_each_problem(pooled):
    results = pooled_drifting_rank_results(pooled.problems, ("true", "believed"), (True,))
    per_problem = [
        pooled_drifting_rank_results({p: s}, ("true", "believed"), (True,))
        for p, s in pooled.problems.items()
    ]
    for basis in ("true", "believed"):
        for mouse, rows in results[basis][True]["counts"].items():
            assert len(rows) == sum(
                len(r[basis][True]["counts"].get(mouse, [])) for r in per_problem if basis in r
            )


# --- the output tree -------------------------------------------------------------------


def test_figures_go_to_all_cohorts_with_the_subfolders_of_all_drifting_problems(written):
    results_root, result = written
    figures = results_root / "figures" / "open-field" / "all-cohorts"
    assert result.figures_dir == figures
    folders = {p.name for p in figures.iterdir() if p.is_dir()}
    assert folders == SUBFOLDERS
    assert {p.name for p in figures.iterdir() if p.is_file()} == {"cohorts.txt"}
    for folder in SUBFOLDERS:
        assert any((figures / folder).glob("*.png")), folder


def test_the_diagnostic_tables_go_to_the_all_cohorts_table_folder(written):
    results_root, result = written
    tables = results_root / "tables" / "open-field" / "all-cohorts"
    assert {p.stem for p in tables.glob("*.csv")} == TABLES
    assert set(result.tables) == TABLES
    assert not any((results_root / "tables" / "open-field").glob("cohort-*"))


def test_it_writes_what_the_cohort_runner_pools_minus_the_per_mouse_figures(
    written, cohort_runner_pool
):
    results_root, _ = written
    figures = results_root / "figures" / "open-field" / "all-cohorts"
    ours = figure_files(figures) - {"cohorts.txt"}
    reference = figure_files(cohort_runner_pool)
    dropped = reference - ours
    assert dropped and all("Per Mouse" in name for name in dropped)
    assert ours <= reference


def test_no_figure_is_per_mouse(written):
    results_root, _ = written
    names = figure_files(results_root / "figures" / "open-field" / "all-cohorts")
    assert not [n for n in names if "Per Mouse" in n or "by Mouse" in n]


def test_the_port_preference_folder_keeps_the_x_versus_plus_figure(written):
    results_root, _ = written
    folder = results_root / "figures" / "open-field" / "all-cohorts" / "port-preference"
    assert {p.name for p in folder.glob("*.png")} == {"Port Preference X vs Plus.png"}


def test_cohorts_txt_lists_the_cohorts_their_mouse_counts_and_the_run_time(written):
    results_root, _ = written
    text = (results_root / "figures/open-field/all-cohorts/cohorts.txt").read_text()
    assert "cohorts 1, 3" in text
    assert "cohort-01: 3 mice" in text
    assert "cohort-03: 2 mice" in text
    assert "2026-10-16 12:30:05" in text


def test_a_rerun_with_other_cohorts_removes_the_old_figures_and_rewrites_cohorts_txt(
    monkeypatch, imported, tmp_path
):
    capture_figures(monkeypatch)  # saving creates the folders; the files themselves do not matter
    run_all_cohorts_drifting(
        [1, 3], results_root=tmp_path, trials_by_cohort=imported, run_at=RUN_AT
    )
    figures = tmp_path / "figures" / "open-field" / "all-cohorts"
    stale = figures / "value-change" / "left over from the first run.png"
    stale.write_bytes(b"")
    unrelated = figures / "notes.md"
    unrelated.write_text("mine")
    run_all_cohorts_drifting([3], results_root=tmp_path, trials_by_cohort=imported, run_at=RUN_AT)
    assert not stale.exists()
    assert unrelated.read_text() == "mine"  # only the folders this pipeline writes are cleared
    text = (figures / "cohorts.txt").read_text()
    assert "cohort-03: 2 mice" in text and "cohort-01" not in text


# --- legends ---------------------------------------------------------------------------


def legend_texts(figure) -> list[str]:
    return [
        t.get_text()
        for ax in figure.axes
        if (legend := ax.get_legend()) is not None
        for t in [*legend.get_texts(), legend.get_title()]
        if t.get_text()
    ]


@pytest.fixture(scope="module")
def captured(imported, tmp_path_factory):
    """The figures of one run, recorded instead of written (the run is slow, so it is shared)."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        capture = capture_figures(monkeypatch)
        run_all_cohorts_drifting(
            [1, 3],
            results_root=tmp_path_factory.mktemp("captured"),
            trials_by_cohort=imported,
            run_at=RUN_AT,
        )
        yield capture


def named_figures(capture):
    return {Path(s.path).name: s.figure for s in capture.saved}


def test_no_legend_names_a_mouse(captured):
    assert captured.saved
    for saved in captured.saved:
        texts = legend_texts(saved.figure)
        assert not ALL_MICE & set(texts), (saved.path, texts)
        assert "Mouse" not in texts, saved.path


def test_the_curve_type_legends_stay(captured):
    figures = named_figures(captured)
    psychometric = next(f for n, f in figures.items() if n.startswith("Psychometric Curves"))
    assert {"P(best)", "P(second)"} <= set(legend_texts(psychometric))
    value_change = next(f for n, f in figures.items() if "Best-Value Increases" in n)
    assert legend_texts(value_change)
    arms = next(f for n, f in figures.items() if "Block Changes" in n)
    assert legend_texts(arms)


def mouse_lines(ax):
    """The lines that carry a circle marker: one per mouse (error-bar caps use other markers)."""
    return [line for line in ax.lines if line.get_marker() == "o"]


def test_the_per_mouse_lines_stay_where_they_come_without_a_legend(captured):
    figures = named_figures(captured)
    ratios = next(f for n, f in figures.items() if n.startswith("Port Preference X vs Plus"))
    assert len(mouse_lines(ratios.axes[0])) == len(ALL_MICE)
    ranks = next(f for n, f in figures.items() if n.startswith("Rank Proportions"))
    assert len(mouse_lines(ranks.axes[0])) == len(ALL_MICE)
    psychometric = next(f for n, f in figures.items() if n.startswith("Psychometric Curves"))
    assert len(mouse_lines(psychometric.axes[0])) >= 2 * len(ALL_MICE)


# --- the cohort runner is unchanged ------------------------------------------------------


@pytest.mark.parametrize(
    ("function", "switches"),
    [
        (run_value_psychometric_analysis, {"per_mouse_heatmaps": True}),
        (run_port_preference_analysis, {"per_mouse_heatmap": True, "mouse_legend": True}),
        (
            run_drifting_value_analyses,
            {"per_mouse_figures": True, "mouse_legends": True, "x_vs_plus": None},
        ),
        (plot_pooled_drifting_rank, {"mouse_legend": True}),
    ],
)
def test_the_new_switches_default_to_the_old_behaviour(function, switches):
    parameters = inspect.signature(function).parameters
    assert {name: parameters[name].default for name in switches} == switches


def test_the_cohort_runner_pool_still_has_per_mouse_figures_and_mouse_legends(
    monkeypatch, pooled, tmp_path
):
    capture = capture_figures(monkeypatch)
    ranks = pooled_drifting_rank_results(pooled.problems, ("true",), (True,))
    plot_pooled_drifting_rank(ranks, tmp_path / "choice-stats")
    run_drifting_value_analyses(
        pooled.problems, tmp_path, DRIFTING_SELECTION, arm_scope_label=POOLED_DRIFTING_LABEL
    )
    names = capture.saved_names()
    assert sum("Per Mouse" in n for n in names) == 2 * 4 + 2  # heatmaps (png, pdf) + port ratios
    rank_figure = next(s.figure for s in capture.saved if "Rank Proportions" in s.path)
    assert set(MICE_A + MICE_B) == set(legend_texts(rank_figure))
    x_vs_plus = next(s.figure for s in capture.saved if "X vs Plus" in s.path)
    assert set(MICE_A + MICE_B) <= set(legend_texts(x_vs_plus))


def test_a_problem_folder_still_skips_x_versus_plus(monkeypatch, pooled, tmp_path):
    capture = capture_figures(monkeypatch)
    run_drifting_value_analyses(
        pooled.problems,
        tmp_path,
        DRIFTING_SELECTION.model_copy(
            update=dict(
                run_value_psychometric_analysis=False,
                run_reversal_aligned_arm_analysis=False,
                run_value_change_analysis=False,
            )
        ),
        arm_scope_label="problem-01",
    )
    assert not [n for n in capture.saved_names() if "X vs Plus" in n]


# --- the script ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def script():
    path = (
        Path(pipeline.__file__).parents[2] / "scripts/full_pipelines/run_all_drifting_problems.py"
    )
    spec = importlib.util.spec_from_file_location("run_all_drifting_problems", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_needs_cohorts(script):
    with pytest.raises(SystemExit):
        script.parse_args([])


def test_the_script_reads_cohorts_and_no_cache(script):
    args = script.parse_args(["--cohorts", "1", "3", "4"])
    assert args.cohorts == [1, 3, 4] and args.no_cache is False
    assert script.parse_args(["--cohorts", "2", "--no-cache"]).no_cache is True


def test_the_script_hands_the_cohorts_to_the_pipeline(script, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        script, "run_all_cohorts_drifting", lambda cohorts, **kw: seen.update(cohorts=cohorts, **kw)
    )
    script.main(["--cohorts", "1", "3", "--no-cache"])
    assert seen["cohorts"] == [1, 3] and seen["use_cache"] is False
    script.main(["--cohorts", "4"])
    assert seen["use_cache"] is True
