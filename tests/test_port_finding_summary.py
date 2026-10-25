"""The port-finding summary: three windows per mouse and problem, averaged across mice.

Sessions are tiny synthetic pyControl logs (``tests.poke_fixtures``); the first-trials and
first-blocks windows are made short so a test needs only a few trials.
"""

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import src.pipelines.port_finding_summary as writers
from src.behavior_analysis.poke_events import ARMS9
from src.behavior_analysis.port_finding_summary import (
    CURRENT,
    FIRST_BLOCKS,
    FIRST_TRIALS,
    INITIATION,
    NO_PORT_FINDING,
    OTHER,
    PORT_FINDING,
    PREVIOUS,
    MousePortFinding,
    TaskTrial,
    WindowPokes,
    block_starts,
    category_fraction_table,
    category_fractions,
    exclusion_reasons,
    first_blocks,
    first_trials,
    group_by_layout,
    mean_percent_map,
    mouse_mean_fractions,
    mouse_port_finding_summary,
    percent_of_pokes,
    port_category,
)
from src.behavior_visualization.plot_maze_heatmaps import draw_maze_heatmap
from tests.poke_fixtures import TOWERS, Recording, full_trial, session_record

S1, S2, S3 = "ses-1_date-20260101", "ses-2_date-20260102", "ses-3_date-20260103"
EARLIER = "ses-0_date-20251231"


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------- builders


def session(n_port_finding=0, choices=("A1", "B1"), blocks=None, long_before=("C3",)):
    """A session: *n_port_finding* port-finding trials, then one task trial per choice.

    Every task trial long-pokes *long_before* before choosing, and pokes A3 after the choice.
    """
    rec = Recording()
    for _ in range(n_port_finding):
        rec.port_finding_trial("A1", long_before=["B3"])
    for choice in choices:
        full_trial(rec, choice=choice, long_before=list(long_before), after_choice=["A3"])
    extra = {} if blocks is None else {"blocks": list(blocks)}
    return session_record([rec.to_frame()], problem=1, **extra)


def port_finding_only(n):
    rec = Recording()
    for _ in range(n):
        rec.port_finding_trial("A1", long_before=["B3"])
    return {"df": rec.to_frame(), "choice_towers": frozenset(TOWERS), "initiation_tower": "B2"}


def counts(**by_port):
    return {port: by_port.get(port, 0) for port in ARMS9}


def window(**by_port):
    return WindowPokes(counts(**by_port), counts(**by_port), n_trials=10)


def summary(mouse="m1", current=("A1", "B1", "C1"), previous=("A1", "C3"), windows=None,
            reasons=()):  # fmt: skip
    windows = windows or {w: window(A1=3, C3=1) for w in (PORT_FINDING, FIRST_TRIALS, FIRST_BLOCKS)}
    return MousePortFinding(
        mouse, frozenset(current), frozenset(previous) if previous else None, "B2", 5, windows,
        tuple(reasons),
    )  # fmt: skip


def trial(key, block):
    return TaskTrial(key, block, None, "B2")


# ---------------------------------------------------------------------------- windows


def test_the_first_trials_need_the_full_number():
    task = [trial(S1, 1)] * 4
    assert first_trials(task, 3) == task[:3]
    assert first_trials(task, 5) is None


def test_each_session_starts_a_new_block_even_with_the_same_block_number():
    task = [trial(S1, 1), trial(S1, 1), trial(S1, 2), trial(S2, 2), trial(S2, 3)]
    assert block_starts(task) == [0, 2, 3, 4]


def test_the_first_blocks_end_where_the_next_block_starts():
    task = [trial(S1, 1), trial(S1, 2), trial(S1, 2), trial(S2, 1), trial(S2, 2)]
    assert first_blocks(task, 2) == task[:3]


def test_the_last_block_is_not_known_to_be_over_until_the_next_one_starts():
    task = [trial(S1, 1), trial(S1, 2), trial(S1, 2)]
    assert first_blocks(task, 2) is None


def test_unknown_block_numbers_give_no_block_window():
    assert first_blocks([trial(S1, 1), trial(S1, None), trial(S2, 1)], 1) is None


# ---------------------------------------------------------------------------- one mouse


def run_mouse(sessions, found=None, **kwargs):
    return mouse_port_finding_summary(
        "m1", sessions, previous_ports=["A1", "C3"], port_finding_sessions=found, **kwargs
    )


def test_port_finding_counts_the_trials_of_both_kinds_of_session():
    sessions = {S1: session(2, blocks=[1, 1])}
    result = run_mouse(sessions, {EARLIER: port_finding_only(3)}, n_trials=2, n_blocks=1)
    assert result.n_port_finding_trials == 5
    # every port-finding trial long-pokes B3 and then chooses A1
    pokes = result.windows[PORT_FINDING].after_initiation
    assert {p: n for p, n in pokes.items() if n} == {"B3": 5, "A1": 5}


def test_the_task_windows_leave_the_port_finding_trials_out():
    result = run_mouse({S1: session(2, choices=["B1", "C1"], blocks=[1, 1])}, n_trials=2,
                       n_blocks=1)  # fmt: skip
    # no block 2 starts, so the block window is not complete
    assert result.windows[FIRST_BLOCKS] is None
    pokes = result.windows[FIRST_TRIALS].after_initiation
    assert {p: n for p, n in pokes.items() if n} == {"C3": 2, "B1": 1, "C1": 1}


def test_the_first_trials_run_on_into_the_next_session():
    sessions = {S1: session(1, choices=["B1"], blocks=[1]), S2: session(0, choices=["C1"] * 3,
                blocks=[1, 1, 2])}  # fmt: skip
    result = run_mouse(sessions, n_trials=3, n_blocks=2)
    assert result.windows[FIRST_TRIALS].n_trials == 3
    assert result.windows[FIRST_TRIALS].after_initiation["B1"] == 1
    assert result.windows[FIRST_TRIALS].after_initiation["C1"] == 2
    # block 1 is S1's block, block 2 is S2's first block, block 3 starts at S2's last trial
    assert result.windows[FIRST_BLOCKS].n_trials == 3
    assert result.reasons == ()


def test_the_after_choice_window_counts_visits_until_the_next_initiation():
    result = run_mouse({S1: session(1, choices=["B1"] * 3, blocks=[1, 1, 2])}, n_trials=3,
                       n_blocks=1)  # fmt: skip
    # three trials poke A3 after the choice, but the last window never closes
    assert result.windows[FIRST_TRIALS].after_choice["A3"] == 2


def test_a_mouse_without_port_finding_or_enough_trials_says_why():
    result = run_mouse({S1: session(0, blocks=[1, 1])}, n_trials=5, n_blocks=1)
    assert result.windows[PORT_FINDING] is None
    assert NO_PORT_FINDING in result.reasons
    assert "fewer than 5 task trials" in result.reasons


def test_an_unreadable_session_warns_and_cuts_the_windows_there():
    broken = {"choice_towers": set(TOWERS), "initiation_tower": "B2"}  # no raw events
    sessions = {S1: session(1, choices=["B1"], blocks=[1]), S2: broken,
                S3: session(0, choices=["C1"] * 5, blocks=[1] * 5)}  # fmt: skip
    with pytest.warns(UserWarning, match="skipped in the port-finding summary"):
        result = run_mouse(sessions, n_trials=3, n_blocks=1)
    assert result.windows[FIRST_TRIALS] is None
    assert any(f"session {S2} could not be read" in r for r in result.reasons)


def test_the_ports_and_layout_come_from_the_sessions():
    result = run_mouse({S1: session(1, blocks=[1, 1])}, n_trials=1, n_blocks=1)
    assert result.layout == (("A1", "B1", "C1"), ("A1", "C3"), "B2")


# ---------------------------------------------------------------------------- categories


def test_a_port_of_both_problems_counts_as_current():
    assert port_category("A1", {"A1", "B1"}, {"A1", "C3"}, "B2") == CURRENT
    assert port_category("C3", {"A1", "B1"}, {"A1", "C3"}, "B2") == PREVIOUS
    assert port_category("B2", {"A1", "B1"}, None, "B2") == INITIATION
    assert port_category("A2", {"A1", "B1"}, None, "B2") == OTHER


def test_the_category_fractions_add_up_to_one():
    fractions = category_fractions(summary(windows=None), PORT_FINDING, "after_choice")
    assert fractions == {CURRENT: 0.75, PREVIOUS: 0.25, INITIATION: 0.0, OTHER: 0.0}


def test_each_mouse_weighs_once_however_many_problems_it_ran():
    entries = [({"k": 1}, summary("m1")), ({"k": 2}, summary("m1")), ({"k": 1}, summary("m2"))]
    means = mouse_mean_fractions(category_fraction_table(entries))
    one = means.query("mouse == 'm1' and kind == 'after_choice' and window == 'port_finding'")
    assert set(one["n_problems"]) == {2}
    assert len(means.query("kind == 'after_choice' and window == 'port_finding'")) == 8


def test_a_mouse_with_reasons_adds_no_rows():
    table = category_fraction_table([({"k": 1}, summary(reasons=["fewer than 50 task trials"]))])
    assert table.empty


# ---------------------------------------------------------------------------- averaging


def test_mice_are_averaged_as_percentages_of_their_own_pokes():
    few = summary("few", windows=dict.fromkeys((PORT_FINDING, FIRST_TRIALS, FIRST_BLOCKS),
                                               window(A1=10)))  # fmt: skip
    many = summary("many", windows=dict.fromkeys((PORT_FINDING, FIRST_TRIALS, FIRST_BLOCKS),
                                                 window(B1=1000)))  # fmt: skip
    mean = mean_percent_map([few, many], FIRST_TRIALS, "after_initiation")
    assert mean["A1"] == pytest.approx(50.0)
    assert mean["B1"] == pytest.approx(50.0)


def test_no_pokes_has_no_percentages_and_excludes_the_mouse():
    assert percent_of_pokes(counts()) is None
    empty = summary(windows={**{w: window(A1=1) for w in (PORT_FINDING, FIRST_TRIALS)},
                             FIRST_BLOCKS: window()})  # fmt: skip
    assert exclusion_reasons(empty, "after_choice") == (
        "no after-choice pokes in the first-blocks window",
    )
    with pytest.raises(ValueError, match="cannot be averaged"):
        mean_percent_map([empty], PORT_FINDING, "after_choice")
    with pytest.raises(ValueError, match="kind must be one of"):
        exclusion_reasons(empty, "after_lunch")


def test_layouts_with_more_mice_come_first():
    groups = group_by_layout(
        [("a", summary(current=("A2", "B2", "C2"))), ("b", summary()), ("c", summary())]
    )
    assert list(groups.values()) == [["b", "c"], ["a"]]


# ---------------------------------------------------------------------------- figures


def test_heatmap_numbers_are_rounded_to_whole_units():
    ax = draw_maze_heatmap(counts(A1=13.6, B1=2.4), colorbar_label="x")
    texts = {t.get_text() for t in ax.texts}
    assert {"14", "2"} <= texts
    colorbar = ax.figure.axes[-1]
    assert [t.get_text() for t in colorbar.get_yticklabels()] == ["0", "14"]


def test_a_problem_figure_has_a_row_per_layout_and_lists_the_left_out_mice(tmp_path):
    summaries = {
        "m1": summary("m1"),
        "m2": summary("m2", current=("A2", "B2", "C2")),
        "m3": summary("m3"),
        "m4": summary("m4", reasons=[NO_PORT_FINDING]),
    }
    written = writers.write_problem_port_finding_heatmaps(summaries, tmp_path, scope="cohort-09")
    assert [p.name for p in written] == [
        "Port Finding Summary - Long Pokes After Initiation",
        "Port Finding Summary - Pokes After Choice",
    ]
    base = written[0]
    for ext in ("pdf", "png", "svg"):
        assert base.with_name(f"{base.name}.{ext}").exists()
    caption = base.with_name(f"{base.name}.caption.txt").read_text()
    assert caption.startswith("cohort-09")
    assert f"m4: {NO_PORT_FINDING}" in caption


def test_the_rows_hold_the_layouts_and_their_mice(monkeypatch, tmp_path):
    drawn = []
    monkeypatch.setattr(writers, "plot_port_finding_heatmaps",
                        lambda rows, kind, **k: drawn.append(rows))  # fmt: skip
    summaries = {"m1": summary("m1"), "m2": summary("m2", current=("A2", "B2", "C2")),
                 "m3": summary("m3")}  # fmt: skip
    writers.write_problem_port_finding_heatmaps(summaries, tmp_path, scope="s")
    rows = drawn[0]
    assert [(r.current_ports, r.n_mice) for r in rows] == [
        (("A1", "B1", "C1"), 2),
        (("A2", "B2", "C2"), 1),
    ]


def test_problems_with_no_usable_mouse_write_nothing(tmp_path):
    written = writers.write_problem_port_finding_heatmaps(
        {"m1": summary(reasons=[NO_PORT_FINDING])}, tmp_path, scope="s"
    )
    assert written == []
    assert not list(tmp_path.iterdir())


def test_cohorts_are_pooled_by_layout_not_by_problem_number(monkeypatch, tmp_path):
    drawn = {}
    monkeypatch.setattr(writers, "plot_port_finding_heatmaps",
                        lambda rows, kind, *, title, save_path, **k: drawn.update(
                            {save_path.name: rows}))  # fmt: skip
    entries = {
        ("cohort-01", 1, "a1"): summary("a1"),
        ("cohort-03", 2, "b1"): summary("b1"),
        ("cohort-03", 2, "b2"): summary("b2"),
        ("cohort-03", 1, "b3"): summary("b3", current=("A2", "B2", "C2")),
    }
    writers.write_layout_port_finding_heatmaps(entries, tmp_path, scope="s")
    name = "Port Finding Summary - A1 B1 C1 After A1 C3 - Pokes After Choice"
    (row,) = drawn[name]
    assert row.n_mice == 3
    assert row.label == "cohort-01: Problem 1 (n=1)\ncohort-03: Problem 2 (n=2)"
    assert len(drawn) == 4  # two layouts, two poke kinds


def test_the_category_figure_is_written_with_its_table(tmp_path):
    entries = [({"cohort": "cohort-01", "k": 1}, summary("m1")),
               ({"cohort": "cohort-03", "k": 1}, summary("m2"))]  # fmt: skip
    table = writers.write_category_fractions(
        entries, tmp_path, scope="s", mouse_columns=("cohort", "mouse")
    )
    assert isinstance(table, pd.DataFrame)
    assert (tmp_path / "Port Finding Summary - Port Categories.png").exists()
    caption = (tmp_path / "Port Finding Summary - Port Categories.caption.txt").read_text()
    assert "2 mouse-problem pairs." in caption


def test_no_usable_mouse_writes_no_category_figure(tmp_path):
    table = writers.write_category_fractions(
        [({"k": 1}, summary(reasons=[NO_PORT_FINDING]))], tmp_path, scope="s"
    )
    assert table.empty
    assert not list(tmp_path.iterdir())


def test_one_mouse_has_no_se_in_the_title(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a one-mouse se must not warn either
        writers.write_category_fractions([({"k": 1}, summary("m1"))], tmp_path, scope="s")
    svg = (tmp_path / "Port Finding Summary - Port Categories.svg").read_text()
    assert "mean across mice, n=1 mouse" in svg
