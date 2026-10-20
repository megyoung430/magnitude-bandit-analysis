# ruff: noqa: F811, F401  (fixtures imported from another test module are used as arguments)
"""The all-cohorts pool without chosen mice (``all-cohorts-without-<mice>/``).

Uses the two synthetic cohorts of ``tests/test_all_cohorts_drifting.py`` (1 with mice a1-a3, 3
with b1-b2) and its fixtures.
"""

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.behavior_analysis.exclude_mice import ExcludedMiceError
from src.behavior_visualization.plot_helpers import current_figure_banner
from src.pipelines import all_cohorts_drifting as pipeline
from src.pipelines.all_cohorts_drifting import run_all_cohorts_drifting
from tests.test_all_cohorts_drifting import (
    RUN_AT,
    SESSIONS_FIGURES,
    SUBFOLDERS,
    TABLES,
    figure_files,
    imported,
)

WITHOUT = "all-cohorts-without-a1-b2"


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def plain_and_extra(imported, tmp_path_factory):
    """Two runs of the same data, without and with the exclusion: ``(plain root, extra root)``."""
    plain, extra = tmp_path_factory.mktemp("plain"), tmp_path_factory.mktemp("extra")
    run_all_cohorts_drifting([1, 3], results_root=plain, trials_by_cohort=imported, run_at=RUN_AT)
    result = run_all_cohorts_drifting(
        [1, 3],
        results_root=extra,
        trials_by_cohort=imported,
        run_at=RUN_AT,
        exclude_mice=["b2", "a1"],
    )
    return plain, extra, result


def under(root, kind, name):
    return root / kind / "open-field" / name


def test_the_extra_set_has_its_own_folders_with_the_same_layout(plain_and_extra):
    _, extra, result = plain_and_extra
    figures = under(extra, "figures", WITHOUT)
    assert result.without.figures_dir == figures
    assert {p.name for p in figures.iterdir() if p.is_dir()} == SUBFOLDERS
    sessions = figures / "sessions-per-problem"
    assert {p.name for p in sessions.iterdir()} == SESSIONS_FIGURES
    assert {p.stem for p in under(extra, "tables", WITHOUT).glob("*.csv")} == TABLES


def test_the_extra_tables_leave_out_the_mice(plain_and_extra):
    plain, extra, result = plain_and_extra
    table = pd.read_csv(under(extra, "tables", WITHOUT) / "sessions_per_problem.csv")
    assert set(table["mouse"]) == {"a2", "a3", "b1"}
    assert set(result.without.tables["sessions_per_problem"]["mouse"]) == {"a2", "a3", "b1"}
    full = pd.read_csv(under(plain, "tables", "all-cohorts") / "sessions_per_problem.csv")
    assert set(full["mouse"]) == {"a1", "a2", "a3", "b1", "b2"}


def test_the_normal_all_cohorts_set_is_unchanged(plain_and_extra):
    plain, extra, _ = plain_and_extra
    for kind in ("figures", "tables"):
        assert figure_files(under(plain, kind, "all-cohorts")) == figure_files(
            under(extra, kind, "all-cohorts")
        )
    for csv in under(plain, "tables", "all-cohorts").glob("*.csv"):
        assert csv.read_bytes() == (under(extra, "tables", "all-cohorts") / csv.name).read_bytes()


def test_cohorts_txt_and_excluded_mice_txt_describe_the_extra_set(plain_and_extra):
    _, extra, _ = plain_and_extra
    figures = under(extra, "figures", WITHOUT)
    text = (figures / "cohorts.txt").read_text()
    assert "cohort-01: 2 mice" in text and "cohort-03: 1 mouse" in text
    excluded = (figures / "excluded_mice.txt").read_text()
    assert "a1" in excluded and "b2" in excluded and "2026-10-16 12:30:05" in excluded


def test_every_figure_of_the_extra_set_is_marked_and_titled_with_the_exclusion(
    imported, tmp_path, monkeypatch
):
    seen = []
    real = pipeline.run_drifting_value_analyses

    def spy(problems, out_dir, selection, **kwargs):
        seen.append((out_dir.name, current_figure_banner(), kwargs["arm_scope_label"]))
        return real(problems, out_dir, selection, **kwargs)

    monkeypatch.setattr(pipeline, "run_drifting_value_analyses", spy)
    run_all_cohorts_drifting(
        [1, 3],
        results_root=tmp_path,
        trials_by_cohort=imported,
        run_at=RUN_AT,
        exclude_mice=["a1", "b2"],
    )
    assert seen == [
        ("all-cohorts", None, "all drifting problems, cohorts 1, 3"),
        (WITHOUT, "without a1, b2", "all drifting problems, cohorts 1, 3 without a1, b2"),
    ]


def test_excluding_the_whole_of_a_cohort_leaves_a_one_cohort_pool(imported, tmp_path):
    result = run_all_cohorts_drifting(
        [1, 3],
        results_root=tmp_path,
        trials_by_cohort=imported,
        run_at=RUN_AT,
        exclude_mice=["b1", "b2"],
    )
    figures = result.without.figures_dir
    assert result.without.mice_by_cohort == {1: ["a1", "a2", "a3"]}
    assert "cohort-03" not in (figures / "cohorts.txt").read_text()
    # one cohort left: no figure by cohort
    assert {p.name for p in (figures / "sessions-per-problem").iterdir()} == {
        "Sessions per Drifting Problem.png",
        "Sessions per Drifting Problem.pdf",
    }


def test_a_mouse_that_is_in_no_cohort_raises_before_anything_is_written(imported, tmp_path):
    with pytest.raises(ValueError, match="zz.*a1, a2, a3, b1, b2"):
        run_all_cohorts_drifting(
            [1, 3], results_root=tmp_path, trials_by_cohort=imported, exclude_mice=["a1", "zz"]
        )
    assert not list(tmp_path.iterdir())


def test_excluding_every_mouse_raises_before_anything_is_written(imported, tmp_path):
    with pytest.raises(ValueError, match="no mice left"):
        run_all_cohorts_drifting(
            [1, 3],
            results_root=tmp_path,
            trials_by_cohort=imported,
            exclude_mice=["a1", "a2", "a3", "b1", "b2"],
        )
    assert not list(tmp_path.iterdir())


def test_a_rerun_without_the_option_leaves_the_extra_set_alone(plain_and_extra, imported):
    _, extra, _ = plain_and_extra
    before = figure_files(under(extra, "figures", WITHOUT))
    run_all_cohorts_drifting([1, 3], results_root=extra, trials_by_cohort=imported, run_at=RUN_AT)
    assert figure_files(under(extra, "figures", WITHOUT)) == before


# --- the script ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def script():
    path = (
        Path(pipeline.__file__).parents[2] / "scripts/full_pipelines/run_all_drifting_problems.py"
    )
    spec = importlib.util.spec_from_file_location("run_all_drifting_problems", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_reads_exclude_mice(script):
    assert script.parse_args(["--cohorts", "1"]).exclude_mice is None
    args = script.parse_args(["--cohorts", "1", "3", "--exclude-mice", "MY_175_N", "MY_66_LL"])
    assert args.exclude_mice == ["MY_175_N", "MY_66_LL"]
    with pytest.raises(SystemExit):
        script.parse_args(["--cohorts", "1", "--exclude-mice", "../x"])


def test_the_script_hands_the_mice_to_the_pipeline(script, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        script, "run_all_cohorts_drifting", lambda cohorts, **kw: seen.update(cohorts=cohorts, **kw)
    )
    script.main(["--cohorts", "1", "3", "--exclude-mice", "a1", "b2"])
    assert seen["exclude_mice"] == ["a1", "b2"]
    script.main(["--cohorts", "1"])
    assert seen["exclude_mice"] is None


def test_excluded_mice_txt_is_in_the_table_folder_too(plain_and_extra):
    _, extra, _ = plain_and_extra
    text = (under(extra, "tables", WITHOUT) / "excluded_mice.txt").read_text()
    assert "a1" in text and "b2" in text and "2026-10-16 12:30:05" in text


def test_a_rerun_clears_the_extra_folders_it_owns(imported, tmp_path):
    kwargs = dict(
        results_root=tmp_path, trials_by_cohort=imported, run_at=RUN_AT, exclude_mice=["a1", "b2"]
    )
    run_all_cohorts_drifting([1, 3], **kwargs)
    stale_figure = under(tmp_path, "figures", WITHOUT) / "value-change" / "left over.png"
    stale_table = under(tmp_path, "tables", WITHOUT) / "sessions_per_problem.csv"
    stale_figure.write_text("x")
    stale_table.write_text("mouse\nstale\n")
    run_all_cohorts_drifting([1, 3], **kwargs)
    assert not stale_figure.exists()
    assert set(pd.read_csv(stale_table)["mouse"]) == {"a2", "a3", "b1"}


def test_every_figure_the_pooled_path_saves_for_the_extra_set_carries_the_banner(
    imported, tmp_path, monkeypatch
):
    saved = []
    real = plt.Figure.savefig

    def spy(self, path, *args, **kwargs):
        saved.append((str(path), [t.get_text() for t in self.texts]))
        return real(self, path, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", spy)
    run_all_cohorts_drifting(
        [1, 3],
        results_root=tmp_path,
        trials_by_cohort=imported,
        run_at=RUN_AT,
        exclude_mice=["a1", "b2"],
    )
    extra = [texts for path, texts in saved if f"/{WITHOUT}/" in path]
    plain = [texts for path, texts in saved if f"/{WITHOUT}/" not in path]
    assert extra and plain
    assert all("without a1, b2" in texts for texts in extra)
    assert not any("without a1, b2" in texts for texts in plain)


def test_a_generator_of_mice_is_read_once(imported, tmp_path):
    result = run_all_cohorts_drifting(
        [1, 3],
        results_root=tmp_path,
        trials_by_cohort=imported,
        run_at=RUN_AT,
        exclude_mice=(m for m in ["a1", "b2"]),
    )
    assert result.without.figures_dir.name == WITHOUT


def test_the_script_prints_a_bad_name_on_one_line(script, monkeypatch):
    def fail(cohorts, **kw):
        raise ExcludedMiceError("cannot exclude zz: not among the mice (a1)")

    monkeypatch.setattr(script, "run_all_cohorts_drifting", fail)
    with pytest.raises(SystemExit) as stop:
        script.main(["--cohorts", "1", "--exclude-mice", "zz"])
    assert "zz" in str(stop.value.code)
