"""Port-preference panels, figures and entry point on synthetic sessions."""

import matplotlib
import matplotlib.colors

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.patches import RegularPolygon

from src.behavior_analysis.poke_events import ARMS9
from src.behavior_visualization.plot_helpers import figure_caption
from src.behavior_visualization.plot_port_preference import (
    CATEGORY_LABELS,
    UNDEFINED_COLOR,
    log2_limit,
    plot_category_ratios_panel,
    plot_port_ratio_panel,
)
from src.pipelines.cohort_analysis import DRIFTING_SELECTION
from src.pipelines.open_field import (
    DRIFTING_ONLY_SECTIONS,
    AnalysisSelection,
    check_selection_for_task_type,
    run_problem,
)
from src.pipelines.port_preference import run_port_preference_analysis
from tests.plot_capture import capture_figures
from tests.test_port_preference import TRIALS, session


@pytest.fixture(autouse=True)
def close_figures():
    plt.close("all")
    yield
    plt.close("all")


def problems(mice=("m1", "m2"), numbers=(1,)):
    return {n: {m: {"ses-1": session(TRIALS)} for m in mice} for n in numbers}


def ports_table(ratios, n_trials=10):
    return pd.DataFrame(
        {"ratio": [ratios.get(p, np.nan) for p in ARMS9], "n_trials": n_trials}, index=list(ARMS9)
    ).rename_axis("port")


def face_colors(ax):
    """{port label: facecolor} of the octagons, read in drawing order."""
    octagons = [p for p in ax.patches if isinstance(p, RegularPolygon)]
    assert len(octagons) == 9
    labels = [t.get_text() for t in ax.texts if t.get_text() in ARMS9]
    return dict(zip(labels, [o.get_facecolor() for o in octagons], strict=True))


# ---------------------------------------------------------------- panels


def test_heatmap_scale_is_centred_on_one_and_symmetric_in_log2():
    table = ports_table({"A1": 1.0, "A2": 2.0, "A3": 0.5})
    faces = face_colors(plot_port_ratio_panel(table, limit=1.0))
    assert faces["A2"] != faces["A3"]
    centre = np.array(faces["A1"][:3])
    assert np.allclose(centre, 1.0, atol=0.1)  # the diverging map is near white at 1
    up, down = np.array(faces["A2"][:3]), np.array(faces["A3"][:3])
    assert np.argmax(up) != np.argmax(down)  # one end warm, the other cool


def test_undefined_ports_are_grey():
    table = ports_table({"A1": 1.0}, n_trials=10)
    table.loc["C3", "n_trials"] = 0  # never offered
    ax = plot_port_ratio_panel(table, limit=1.0)
    grey = matplotlib.colors.to_rgba(UNDEFINED_COLOR)
    faces = face_colors(ax)
    assert faces["C3"] == grey  # never offered
    assert faces["B2"] == grey  # offered but no ratio (never best)
    assert faces["A1"] != grey


def test_cells_print_the_ratio_and_n():
    table = ports_table({"A1": 1.25})
    table.loc["A1", "n_trials"] = 123
    ax = plot_port_ratio_panel(table, limit=1.0)
    texts = {t.get_text() for t in ax.texts}
    assert {"1.25", "n=123"} <= texts
    assert "no best" in texts  # a port with n > 0 but no ratio says why


def test_never_offered_port_says_n_a():
    table = ports_table({"A1": 1.0})
    table.loc["B2", "n_trials"] = 0
    texts = {t.get_text() for t in plot_port_ratio_panel(table, limit=1.0).texts}
    assert {"n/a", "n=0"} <= texts


def test_missing_port_raises():
    with pytest.raises(KeyError, match="B2"):
        plot_port_ratio_panel(ports_table({}).drop("B2"), limit=1.0)


def test_limit_covers_the_largest_log2_ratio_and_ignores_zero_and_nan():
    assert log2_limit(np.array([1.0, 1.2, np.nan, 0.0])) == 1.0
    assert log2_limit(np.array([0.2, 3.0])) == 3.0  # |log2 0.2| = 2.32 rounds up to 3
    assert log2_limit(np.array([np.nan])) == 1.0


def test_category_panel_draws_reference_line_bars_and_one_line_per_mouse():
    per_mouse = pd.DataFrame(
        {
            "mouse": ["a", "b", "a", "b"],
            "category": ["X", "X", "plus", "plus"],
            "ratio": [0.5, 0.7, 1.5, np.nan],
        }
    )
    summary = pd.DataFrame(
        {"mean": [0.6, 1.5], "se": [0.1, np.nan], "n_mice": [2, 1]},
        index=pd.Index(["X", "plus"], name="category"),
    )
    fig, ax = plt.subplots()
    returned = plot_category_ratios_panel(per_mouse, summary, ax=ax)
    assert returned is ax
    assert [b.get_height() for b in ax.patches] == [0.6, 1.5]
    assert any(np.allclose(line.get_ydata(), 1.0) for line in ax.lines)
    mouse_lines = [line for line in ax.lines if line.get_label() in ("a", "b")]
    assert len(mouse_lines) == 2


# ---------------------------------------------------------------- entry point


def test_writes_both_figures_in_every_format_and_returns_tables(tmp_path):
    result = run_port_preference_analysis(problems(), tmp_path / "new", scope_label="problem-01")
    for stem in ("Port Preference X vs Plus", "Port Preference Per Mouse"):
        for ext in ("pdf", "png", "svg"):
            assert (tmp_path / "new" / f"{stem}.{ext}").stat().st_size > 0
    assert not list((tmp_path / "new").glob("*.csv"))
    assert set(result["summary"].index) == {"X", "plus"}
    assert result["summary"].loc["X", "n_mice"] == 2
    assert set(result["port_ratios"].mouse) == {"m1", "m2"}
    # each mouse offers C3 but never has it best
    assert result["nan_cases"] == {"ports": 2, "categories": 0}


def test_pools_problems_by_counts(tmp_path):
    result = run_port_preference_analysis(
        problems(mice=("m1",), numbers=(1, 2)), tmp_path, scope_label="all drifting problems"
    )
    a1 = result["port_ratios"].set_index("port").loc["A1"]
    assert (a1.n_trials, a1.chosen, a1.best) == (6, 4, 3)


def test_counter_format_sessions_raise(tmp_path):
    data = problems()
    data[1]["m1"]["ses-1"].update({"trial": [1, 2, 3], "has_good": True})
    with pytest.raises(ValueError, match="'m1'.*drifting task has only block changes"):
        run_port_preference_analysis(data, tmp_path, scope_label="x")


# ---------------------------------------------------------------- runner wiring


def test_the_drifting_selection_runs_it_and_the_default_does_not():
    assert DRIFTING_SELECTION.run_port_preference_analysis
    assert not AnalysisSelection().run_port_preference_analysis
    assert "run_port_preference_analysis" in DRIFTING_ONLY_SECTIONS


def test_a_blocked_problem_with_it_selected_raises():
    selection = AnalysisSelection(run_port_preference_analysis=True)
    with pytest.raises(ValueError, match="run_port_preference_analysis"):
        check_selection_for_task_type(selection, "blocked")
    with pytest.raises(ValueError, match="run_port_preference_analysis"):
        run_problem(
            1, {}, task="t", cohort="c", output_root=".", selection=selection, task_type="blocked"
        )


def test_a_drifting_problem_writes_into_its_port_preference_folder(tmp_path):
    selection = AnalysisSelection(
        run_choice_probability_analysis=False,
        run_first_leave_analysis=False,
        run_number_of_reversals_analysis=False,
        run_rank_proportion_analysis=False,
        run_single_session_analysis=False,
        run_port_preference_analysis=True,
    )
    run_problem(
        3,
        problems(numbers=(3,))[3],
        task="open-field",
        cohort="cohort-01",
        output_root=tmp_path,
        selection=selection,
        task_type="drifting",
    )
    folder = tmp_path / "open-field" / "cohort-01" / "drifting" / "problem-03" / "port-preference"
    assert (folder / "Port Preference Per Mouse.png").exists()
    assert not (folder / "Port Preference X vs Plus.png").exists()  # pooled scope only


# ---------------------------------------------------------------- labels and exclusions


def saved_figures(capture):
    return {Path(s.path).name.rsplit(".", 1)[0]: s.figure for s in capture.saved}


def plus_only_mouse():
    """A mouse offered only plus ports (from B2), so its plus ratio is 1 by construction."""
    return {"ses-1": session([({"B1": 3, "B3": 1, "C2": 0}, "B1")] * 2)}


def test_titles_say_mice_and_the_heatmap_title_is_its_file_name(tmp_path, monkeypatch):
    capture = capture_figures(monkeypatch)
    run_port_preference_analysis(problems(), tmp_path, scope_label="problem-01")
    figures = saved_figures(capture)
    category = figures["Port Preference X vs Plus"]
    assert category.axes[0].get_title() == (
        "Port Preference: X vs Plus\nmean ± SE across mice, n = 2 mice"
    )
    assert figure_caption(category) == "Data: problem-01."
    heatmaps = figures["Port Preference Per Mouse"]
    assert heatmaps._suptitle.get_text() == "Port Preference Per Mouse"
    caption = figure_caption(heatmaps).splitlines()
    assert caption[0] == "Data: problem-01; 2 mice."
    assert caption[2].startswith("Outlines:") and "blue = initiation port" in caption[2]


def test_plus_ports_are_labelled_orthogonal():
    assert "orthogonal" in CATEGORY_LABELS["plus"] and "adjacent" not in CATEGORY_LABELS["plus"]


def test_a_mouse_offered_one_category_is_excluded_noted_and_counted(tmp_path, monkeypatch, caplog):
    capture = capture_figures(monkeypatch)
    data = problems()
    data[1]["m3"] = plus_only_mouse()
    with caplog.at_level("INFO"):
        result = run_port_preference_analysis(data, tmp_path, scope_label="problem-01")
    assert result["single_category_mice"] == ["m3"]
    assert result["category_ratios"].query("mouse == 'm3'").ratio.isna().all()
    assert result["summary"].loc["plus", "n_mice"] == 2
    assert "1 mouse excluded: only one category offered" in caplog.text
    figure = saved_figures(capture)["Port Preference X vs Plus"]
    assert "1 mouse excluded: only one category offered." in figure_caption(figure)
    assert figure.axes[0].get_title().endswith("n = 2 mice")


def test_no_exclusion_note_when_every_mouse_is_kept(tmp_path, monkeypatch):
    capture = capture_figures(monkeypatch)
    result = run_port_preference_analysis(problems(), tmp_path, scope_label="x")
    assert result["single_category_mice"] == []
    figure = saved_figures(capture)["Port Preference X vs Plus"]
    assert "excluded" not in figure_caption(figure)


def test_a_scope_where_every_mouse_is_excluded_still_draws(tmp_path):
    result = run_port_preference_analysis({1: {"m1": plus_only_mouse()}}, tmp_path, scope_label="x")
    assert result["single_category_mice"] == ["m1"]
    assert (tmp_path / "Port Preference X vs Plus.png").stat().st_size > 0


def test_one_problem_heatmaps_outline_current_ports_green_and_previous_red(tmp_path, monkeypatch):
    captured = {}
    import src.pipelines.port_preference as pipeline

    def record(port_ratios, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(pipeline, "plot_port_ratio_heatmaps", record)
    monkeypatch.setattr(pipeline, "plot_category_ratios", lambda *a, **k: None)
    problems = _one_problem_for_outlines()
    (mouse,) = problems[1]
    run_port_preference_analysis(
        problems, tmp_path, scope_label="problem-01", previous_choice_towers={mouse: {"A1"}}
    )
    assert captured["current_ports"][mouse] == frozenset(
        problems[1][mouse][next(iter(problems[1][mouse]))]["choice_towers"]
    )
    assert captured["previous_ports"] == {mouse: {"A1"}}


def test_pooled_heatmaps_have_no_green_or_red_outlines(tmp_path, monkeypatch):
    captured = {}
    import src.pipelines.port_preference as pipeline

    monkeypatch.setattr(pipeline, "plot_port_ratio_heatmaps", lambda r, **k: captured.update(k))
    monkeypatch.setattr(pipeline, "plot_category_ratios", lambda *a, **k: None)
    one = _one_problem_for_outlines()
    run_port_preference_analysis(
        {1: one[1], 2: one[1]}, tmp_path, scope_label="pool", previous_choice_towers={}
    )
    assert "current_ports" not in captured and "previous_ports" not in captured


def test_panel_draws_red_and_green_rings(tmp_path):
    import pandas as pd

    from src.behavior_analysis.poke_events import ARMS9
    from src.behavior_visualization.plot_port_preference import plot_port_ratio_panel

    ports = pd.DataFrame({"ratio": [1.0] * 9, "n_trials": [5] * 9}, index=list(ARMS9))
    ax = plot_port_ratio_panel(
        ports, limit=1.0, initiation_ports={"B2"}, current_ports={"B1", "C2"},
        previous_ports={"B1", "A3"},
    )  # fmt: skip
    gids = sorted(p.get_gid() for p in ax.patches if p.get_gid())
    assert gids == sorted(
        ["ring:B1:current", "ring:C2:current", "ring:B1:previous", "ring:A3:previous",
         "ring:B2:initiation"]
    )  # fmt: skip


def _one_problem_for_outlines():
    sessions = {"ses-1": {**session(TRIALS), "choice_towers": ["B1", "B3", "C2"]}}
    return {1: {"m1": sessions}}


def test_choice_towers_stored_per_trial_give_their_union(tmp_path, monkeypatch):
    # The importer can store choice_towers as one list of ports per trial (lists are unhashable).
    captured = {}
    import src.pipelines.port_preference as pipeline

    monkeypatch.setattr(pipeline, "plot_port_ratio_heatmaps", lambda r, **k: captured.update(k))
    sessions = {"ses-1": {**session(TRIALS), "choice_towers": [["B1", "B3", "C2"]] * 3}}
    run_port_preference_analysis({1: {"m1": sessions}}, tmp_path, scope_label="problem-01")
    assert captured["current_ports"] == {"m1": frozenset({"B1", "B3", "C2"})}


def test_x_vs_plus_figure_can_be_left_out(tmp_path):
    run_port_preference_analysis(problems(), tmp_path, scope_label="problem-01", x_vs_plus=False)
    assert not list(tmp_path.glob("Port Preference X vs Plus*"))
    assert list(tmp_path.glob("Port Preference Per Mouse*.png"))
