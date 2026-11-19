"""Training order, stages and the stage analyses of learning across problems."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis.choice_logit import logit_by_problem as lbp
from src.behavior_analysis.data import training_stages as ts
from src.behavior_analysis.pokes import port_finding_by_stage as pfs
from src.behavior_analysis.pokes.port_finding_summary import CURRENT, INITIATION, WINDOWS
from src.behavior_analysis.reversals import reversal_curves_by_stage as rcs
from src.behavior_visualization.pokes.plot_port_finding_by_stage import (
    plot_port_finding_by_stage,
)
from src.behavior_visualization.reversals.plot_reversal_curves_by_stage import (
    plot_reversal_curves_by_stage,
)
from src.pipelines.common import training_stages as pipe_stages

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def problems(mouse, cohort, sequence):
    """*sequence*: ``[(task, k), ...]`` in training order."""
    return pd.DataFrame([dict(mouse=mouse, cohort=cohort, task=task, k=k, overall_problem=i)
                         for i, (task, k) in enumerate(sequence, start=1)])  # fmt: skip


def drifting(n):
    return [("drifting", k) for k in range(1, n + 1)]


# ---------------------------------------------------------------------------
# training order and stages
# ---------------------------------------------------------------------------


def test_training_problems_keep_structured_before_drifting_and_drifting_up_to_the_cutoff():
    table = pd.concat([
        problems("a", 1, [("blocked", 1), ("blocked", 2), *drifting(3), ("blocked", 3)]),
        problems("b", 3, [("blocked", 1), *drifting(5)]),
    ])  # fmt: skip
    kept = ts.training_problems(table, {1: 2, 3: None})
    a = kept[kept["mouse"] == "a"]
    assert list(zip(a["phase"], a["phase_problem"], strict=True)) == [
        ("structured", 1), ("structured", 2), ("drifting", 1), ("drifting", 2),
    ]  # fmt: skip
    assert (kept["mouse"] == "b").sum() == 6


def test_the_stages_are_the_first_two_structured_and_first_and_last_two_drifting():
    table = ts.training_problems(problems("a", 1, [("blocked", 1), ("blocked", 2),
                                                   ("blocked", 3), *drifting(6)]),
                                 {1: None})  # fmt: skip
    staged = ts.training_stages(table).set_index(["phase", "phase_problem"])["stage"]
    assert staged.to_dict() == {
        ("structured", 1): ts.EARLY, ("structured", 2): ts.EARLY,
        ("drifting", 1): ts.SWITCH, ("drifting", 2): ts.SWITCH,
        ("drifting", 5): ts.LATE, ("drifting", 6): ts.LATE,
    }  # fmt: skip


def test_late_never_shares_a_problem_with_switch():
    table = ts.training_problems(problems("a", 1, [("blocked", 1), *drifting(3)]), {1: None})
    staged = ts.training_stages(table).set_index(["phase", "phase_problem"])["stage"]
    assert staged[("drifting", 3)] == ts.LATE
    assert (staged == ts.LATE).sum() == 1
    short = ts.training_problems(problems("b", 1, [("blocked", 1), *drifting(2)]), {1: None})
    assert ts.LATE not in set(ts.training_stages(short)["stage"])


def test_stage_tests_pair_mice_and_need_two_for_a_test():
    per_mouse = pd.DataFrame(dict(
        mouse=["a", "a", "a", "b", "b", "b"], stage=[ts.EARLY, ts.SWITCH, ts.LATE] * 2,
        value=[0.1, 0.3, 0.5, 0.2, 0.2, 0.6],
    ))  # fmt: skip
    tests = ts.stage_tests(per_mouse, "value").set_index(["first", "second"])
    early_late = tests.loc[(ts.EARLY, ts.LATE)]
    assert early_late["mean_difference"] == pytest.approx(0.4)
    assert early_late["n_mice"] == 2 and np.isfinite(early_late["p"])
    single = ts.stage_tests(per_mouse[per_mouse["mouse"] == "a"], "value")
    assert single["p"].isna().all()


def test_stage_tests_split_by_group_columns():
    per_mouse = pd.DataFrame(dict(mouse=["a", "a", "b", "b"] * 2,
                                  stage=[ts.EARLY, ts.LATE] * 4,
                                  kind=["x"] * 4 + ["y"] * 4,
                                  value=[0, 1, 0, 2, 1, 1, 2, 2]))  # fmt: skip
    tests = ts.stage_tests(per_mouse, "value", by=["kind"])
    early_late = tests[(tests["first"] == ts.EARLY) & (tests["second"] == ts.LATE)]
    assert early_late.set_index("kind")["mean_difference"].to_dict() == {"x": 1.5, "y": 0.0}


def test_problem_table_reads_the_typed_numbering_of_imported_cohorts():
    def session(task_name):
        frame = pd.DataFrame({"type": []})
        return {"task_name": task_name, "df": frame}

    blocked, drift = "Meg\\3x3_field_blocked_value_bandit", "Meg\\3x3_field_drifting_value_bandit"
    imported = {1: {"m": {"ses-1_date-20260101": session(blocked)}},
                2: {"m": {"ses-2_date-20260102": session(drift)}}}  # fmt: skip
    table = pipe_stages.problem_table({4: imported})
    assert table.set_index("overall_problem")[["task", "k"]].to_dict("index") == {
        1: {"task": "blocked", "k": 1}, 2: {"task": "drifting", "k": 1},
    }  # fmt: skip


# ---------------------------------------------------------------------------
# early and late trials of a problem
# ---------------------------------------------------------------------------


def test_the_early_and_late_parts_split_a_problem_at_n_early():
    assert lbp._part_mask(5, lbp.EARLY_TRIALS, 2).tolist() == [True, True, False, False, False]
    assert lbp._part_mask(5, lbp.LATE_TRIALS, 2).tolist() == [False, False, True, True, True]
    assert lbp._part_mask(3, lbp.ALL, 2).all()


def test_the_gap_is_late_minus_early_for_problems_with_both():
    def summaries(weights):
        return pd.DataFrame([dict(mouse="m", cohort=1, phase="drifting", phase_problem=k,
                                  group="choice", summary=lbp.LAG_ONE, weight=w)
                             for k, w in weights.items()])  # fmt: skip

    gap = lbp.early_late_gap(summaries({1: 0.1, 2: 0.2}), summaries({1: 0.5, 3: 0.9}))
    assert gap[["phase_problem", "weight"]].to_dict("records") == [
        {"phase_problem": 1, "weight": pytest.approx(0.4)}
    ]


# ---------------------------------------------------------------------------
# reversal curves and port finding by stage
# ---------------------------------------------------------------------------


def curves_table():
    rows = []
    for mouse, shift in (("a", 0.0), ("b", 0.1)):
        for phase, k, stage_value in (("structured", 1, 0.2), ("structured", 2, 0.4),
                                      ("drifting", 9, 0.6)):  # fmt: skip
            for offset in range(-2, 12):
                for role in rcs.ROLES:
                    p = stage_value + shift if role == rcs.NEW_BEST else (1 - stage_value) / 2
                    rows.append(dict(mouse=mouse, role=role, offset=offset, probability=p,
                                     n_reversals=4, phase=phase, phase_problem=k))  # fmt: skip
    return pd.DataFrame(rows)


def stage_table():
    return pd.DataFrame([dict(mouse=m, phase=p, phase_problem=k, stage=s)
                         for m in ("a", "b")
                         for p, k, s in (("structured", 1, ts.EARLY), ("structured", 2, ts.EARLY),
                                         ("drifting", 9, ts.LATE))])  # fmt: skip


def test_stage_curves_average_a_stage_s_problems_equally():
    curves = rcs.stage_curves(curves_table(), stage_table())
    early = curves[(curves["stage"] == ts.EARLY) & (curves["role"] == rcs.NEW_BEST)
                   & (curves["mouse"] == "a")]  # fmt: skip
    assert early["probability"].to_numpy() == pytest.approx(0.3)
    assert set(early["n_problems"]) == {2} and set(early["n_reversals"]) == {8}


def test_adaptation_readouts_average_the_window_after_the_change():
    readouts = rcs.adaptation_readouts(rcs.stage_curves(curves_table(), stage_table()))
    new_best = readouts[readouts["role"] == rcs.NEW_BEST].set_index(["mouse", "stage"])
    assert new_best.loc[("b", ts.LATE), "probability"] == pytest.approx(0.7)
    assert set(readouts["role"]) == {rcs.NEW_BEST, rcs.PREVIOUS_BEST}


def test_the_reversal_figure_is_saved(tmp_path):
    curves = rcs.stage_curves(curves_table(), stage_table())
    readouts = rcs.adaptation_readouts(curves)
    tests = ts.stage_tests(readouts, "probability", by=["role"])
    plot_reversal_curves_by_stage(curves, readouts, tests, tmp_path / "curves")
    assert {p.suffix for p in tmp_path.iterdir()} >= {".png", ".pdf", ".svg", ".txt"}


def fractions_table():
    rows = []
    for mouse in ("a", "b"):
        for task_type, k in (("blocked", 1), ("drifting", 9)):
            for kind in pfs.IN_SEQUENCE:
                for window in WINDOWS:
                    for category, f in ((CURRENT, 0.5), (INITIATION, 0.3), ("previous", np.nan),
                                        ("other", 0.2)):  # fmt: skip
                        value = f + (0.1 if task_type == "drifting" and category == CURRENT
                                     else 0)  # fmt: skip
                        rows.append(dict(cohort=1, task_type=task_type, k=k, mouse=mouse,
                                         kind=kind, window=window, category=category,
                                         fraction=value))  # fmt: skip
    return pd.DataFrame(rows)


def test_port_fractions_get_their_phase_and_a_stage_mean(tmp_path):
    stages = pd.DataFrame([dict(mouse=m, phase=p, phase_problem=k, stage=s) for m in ("a", "b")
                           for p, k, s in (("structured", 1, ts.EARLY),
                                           ("drifting", 9, ts.LATE))])  # fmt: skip
    staged = pfs.stage_fractions(pfs.problem_keys(fractions_table()), stages)
    current = staged[(staged["category"] == CURRENT) & (staged["stage"] == ts.LATE)]
    assert current["fraction"].to_numpy() == pytest.approx(0.6)
    assert "previous" not in set(staged["category"])  # all NaN: left out
    in_sequence = pfs.in_sequence_fractions(staged)
    assert set(zip(in_sequence["kind"], in_sequence["category"], strict=True)) == {
        ("after_initiation", CURRENT), ("after_choice", INITIATION),
    }  # fmt: skip
    tests = ts.stage_tests(in_sequence, "fraction", by=["kind", "window"])
    plot_port_finding_by_stage(staged, tests, tmp_path / "ports")
    assert {p.suffix for p in tmp_path.iterdir()} >= {".png", ".pdf", ".svg", ".txt"}


def test_an_unknown_task_type_raises():
    with pytest.raises(ValueError, match="pretraining"):
        pfs.problem_keys(fractions_table().assign(task_type="pretraining"))
