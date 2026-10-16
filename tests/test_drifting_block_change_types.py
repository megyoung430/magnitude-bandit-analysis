"""Block changes of the drifting task: which are suspect, and the two types the arms split into.

At a drifting block change the most-chosen arm drops to magnitude 0 and the arm that was at 0
is rewarded again. That arm was the best or the second-best, so the best tower often stays the
best: it is normal in the drifting task and a defect only in the blocked task, where the best
tower must change at a good reversal. In the drifting task the check is that the zero arm moves.
"""

import matplotlib

matplotlib.use("Agg")
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.behavior_analysis.get_good_reversal_info import ReversalWarning, get_good_reversal_info
from src.behavior_analysis.get_total_reversals import get_total_reversals
from src.behavior_analysis.reversal_aligned_arms import (
    build_windows_with_magnitudes,
    compute_reversal_aligned_arms,
)
from src.behavior_analysis.suspect_reversal_table import suspect_reversal_table
from src.behavior_visualization.plot_reversal_aligned_arms import plot_reversal_aligned_arms
from src.pipelines.reversal_aligned_arms import run_reversal_aligned_arm_analysis
from tests.plot_capture import capture_figures

DRIFTING_NAME = "Meg\\3x3_field_drifting_value_bandit"
BLOCKED_NAME = "Meg\\3x3_field_blocked_reward_bandit"
BLOCK_LEN = 12
TOWERS = "ABC"

# (A, B, C) per block. Each block change drops one arm to 0 and rewards the arm that was at 0:
#   1 -> 2: best A drops, C (was 0) becomes best      best_to_third,   third_to_best
#   2 -> 3: best C drops, A (was 0) becomes second    best_to_third,   third_to_second
#   3 -> 4: second A drops, C (was 0) becomes second  second_to_third, third_to_second
#           (the best tower B stays best)
#   4 -> 5: second C drops, A (was 0) becomes best    second_to_third, third_to_best
BLOCKS = [(4, 2, 0), (0, 2, 3), (1, 2, 0), (0, 2, 1), (3, 2, 0)]


def make_session(blocks=BLOCKS, *, task_name=DRIFTING_NAME, counter=False, hole=None):
    """One session with constant magnitudes per block; the animal always picks the best arm."""
    magnitudes = {t: [m[i] for m in blocks for _ in range(BLOCK_LEN)] for i, t in enumerate(TOWERS)}
    n = BLOCK_LEN * len(blocks)
    best = [max(TOWERS, key=lambda t: magnitudes[t][i]) for i in range(n)]
    if hole is not None:
        magnitudes["B"][hole] = None
    session = {
        "trial": list(range(1, n + 1)),
        "blocks": [1 + i // BLOCK_LEN for i in range(n)],
        "reward_magnitudes_by_tower": magnitudes,
        "choices_by_tower": {t: [float(b == t) for b in best] for t in TOWERS},
        "choices_by_rank": {r: [float(r == "best")] * n for r in ("best", "second", "third")},
        "task_name": task_name,
        "has_good": counter,
        "has_bad": False,
    }
    if counter:
        session["good_reversals"] = [i // BLOCK_LEN for i in range(n)]
    return session


def data(**kwargs):
    return {"m": {"ses-1_date-20260901": make_session(**kwargs)}}


def reversal_indices(info):
    return [r["reversal_idx"] for r in info["m"]]


# ----------------------------------------------------------- suspect: drifting vs blocked


def test_drifting_keeps_a_block_change_whose_best_tower_is_unchanged():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ReversalWarning)
        info = get_good_reversal_info(data(), pre=2, post=4)
    assert reversal_indices(info) == [12, 24, 36, 48]


def test_blocked_still_drops_a_good_reversal_whose_best_tower_is_unchanged():
    with pytest.warns(ReversalWarning, match="dropped 1 good reversal"):
        info = get_good_reversal_info(data(task_name=BLOCKED_NAME, counter=True), pre=2, post=4)
    assert reversal_indices(info) == [12, 24, 48]


def test_a_session_without_a_task_name_keeps_the_best_tower_check():
    sessions = data()
    del sessions["m"]["ses-1_date-20260901"]["task_name"]
    with pytest.warns(ReversalWarning, match="dropped 1 good reversal"):
        info = get_good_reversal_info(sessions, pre=2, post=4)
    assert reversal_indices(info) == [12, 24, 48]


def test_the_task_type_set_by_the_importer_decides_before_the_name():
    sessions = data(task_name="not a task name")
    sessions["m"]["ses-1_date-20260901"]["task_type"] = "drifting"
    with warnings.catch_warnings():
        warnings.simplefilter("error", ReversalWarning)
        info = get_good_reversal_info(sessions, pre=2, post=4)
    assert reversal_indices(info) == [12, 24, 36, 48]


def test_an_unrecognised_task_name_raises():
    with pytest.raises(ValueError, match="unrecognised task name"):
        get_good_reversal_info(data(task_name="Meg\\something_else"), pre=2, post=4)


def test_a_drifting_block_change_where_the_zero_arm_does_not_move_is_suspect():
    # Block 3 keeps the zero arm on C: the drift of the other arms is not a block change.
    blocks = [(4, 2, 0), (2, 4, 0), (0, 4, 3)]
    with pytest.warns(ReversalWarning, match="zero arm unchanged"):
        info = get_good_reversal_info(data(blocks=blocks), pre=2, post=4)
    assert reversal_indices(info) == [24]
    table = suspect_reversal_table(data(blocks=blocks), pre=2, post=4)
    assert list(table["reason"]) == ["zero arm unchanged"]
    assert list(table["merged_trial"]) == [12]


def test_a_drifting_block_change_with_a_missing_magnitude_is_suspect():
    with pytest.warns(ReversalWarning, match="missing magnitude"):
        get_good_reversal_info(data(hole=12), pre=2, post=4)
    table = suspect_reversal_table(data(hole=12), pre=2, post=4)
    assert list(table["reason"]) == ["missing magnitude"]


def test_the_suspect_table_of_a_clean_drifting_session_is_empty():
    assert suspect_reversal_table(data(), pre=2, post=4).empty


def test_blocked_suspect_table_still_lists_the_unchanged_best():
    table = suspect_reversal_table(data(task_name=BLOCKED_NAME, counter=True), pre=2, post=4)
    assert list(table["reason"]) == ["best tower unchanged"]
    assert list(table["merged_trial"]) == [36]


def test_drifting_reversal_totals_have_no_suspect_best_unchanged():
    totals = get_total_reversals(data())
    assert totals.get("suspect_good_reversals", 0) == 0
    assert totals.get("uncheckable_good_reversals", 0) == 0


# ------------------------------------------------------------- the two arm splits


def arms(scheme, reversal_type):
    windows = build_windows_with_magnitudes(data(), pre=4, post=6)
    return compute_reversal_aligned_arms(
        windows, scheme=scheme, reversal_type=reversal_type, pre=4, post=6, moving_avg_window=1
    )


def test_prev_scheme_has_both_types_and_all_is_their_union():
    best, second, everything = (
        arms("prev", t) for t in ("best_to_third", "second_to_third", "all")
    )
    assert (best.num_reversals, second.num_reversals) == (2, 2)
    assert everything.num_reversals == 4


def test_new_scheme_has_both_types_and_all_is_their_union():
    best, second, everything = (arms("new", t) for t in ("third_to_best", "third_to_second", "all"))
    assert (best.num_reversals, second.num_reversals) == (2, 2)
    assert everything.num_reversals == 4


@pytest.mark.parametrize(
    ("scheme", "types"),
    [("prev", ("best_to_third", "second_to_third")), ("new", ("third_to_best", "third_to_second"))],
)
def test_all_averages_over_both_types(scheme, types):
    first, second = (arms(scheme, t) for t in types)
    everything = arms(scheme, "all")
    for role in everything.roles:
        np.testing.assert_allclose(
            everything.choice_mean[role],
            (first.choice_mean[role] + second.choice_mean[role]) / 2,
        )
        np.testing.assert_allclose(
            everything.magnitude_mean[role],
            (first.magnitude_mean[role] + second.magnitude_mean[role]) / 2,
        )


def test_prev_curves_show_which_arm_the_animal_leaves():
    # The animal picks the best arm, so before every block change it picks prev_best. After a
    # best_to_third change that arm is dropped to 0; after a second_to_third change the best
    # arm stays best in one of the two cases, where the animal stays on prev_best.
    pre, post = slice(0, 4), slice(4, 10)
    best_dropped, second_dropped = arms("prev", "best_to_third"), arms("prev", "second_to_third")
    for result in (best_dropped, second_dropped):
        np.testing.assert_allclose(result.choice_mean["prev_best"][pre], 1.0)
    np.testing.assert_allclose(best_dropped.choice_mean["prev_best"][post], 0.0)
    np.testing.assert_allclose(second_dropped.choice_mean["prev_best"][post], 0.5)


def test_a_type_of_the_other_scheme_is_refused():
    windows = build_windows_with_magnitudes(data(), pre=4, post=6)
    with pytest.raises(ValueError, match="third_to_best"):
        compute_reversal_aligned_arms(
            windows, scheme="prev", reversal_type="third_to_best", pre=4, post=6
        )
    with pytest.raises(ValueError, match="second_to_third"):
        compute_reversal_aligned_arms(
            windows, scheme="new", reversal_type="second_to_third", pre=4, post=6
        )


# -------------------------------------------------------------- titles and file names


@pytest.fixture
def capture(monkeypatch):
    capture = capture_figures(monkeypatch)
    yield capture
    plt.close("all")


EXPECTED = {
    ("prev", "all"): (
        "Choice Probabilities Around Block Changes (All)",
        "arms labelled by rank before the block change",
        "Choice Probabilities Around Block Changes (All) (Arms Labelled by Rank Before) (pooled)",
    ),
    ("prev", "best_to_third"): (
        "Choice Probabilities Around Block Changes (Prev Best to Third)",
        "arms labelled by rank before the block change",
        "Choice Probabilities Around Block Changes (Prev Best to Third) "
        "(Arms Labelled by Rank Before) (pooled)",
    ),
    ("prev", "second_to_third"): (
        "Choice Probabilities Around Block Changes (Prev Second to Third)",
        "arms labelled by rank before the block change",
        "Choice Probabilities Around Block Changes (Prev Second to Third) "
        "(Arms Labelled by Rank Before) (pooled)",
    ),
    ("new", "all"): (
        "Choice Probabilities Around Block Changes (All)",
        "arms labelled by rank after the block change",
        "Choice Probabilities Around Block Changes (All) (Arms Labelled by Rank After) (pooled)",
    ),
    ("new", "third_to_best"): (
        "Choice Probabilities Around Block Changes (New Best from Third)",
        "arms labelled by rank after the block change",
        "Choice Probabilities Around Block Changes (New Best from Third) "
        "(Arms Labelled by Rank After) (pooled)",
    ),
    ("new", "third_to_second"): (
        "Choice Probabilities Around Block Changes (New Second from Third)",
        "arms labelled by rank after the block change",
        "Choice Probabilities Around Block Changes (New Second from Third) "
        "(Arms Labelled by Rank After) (pooled)",
    ),
}


@pytest.mark.parametrize(("scheme", "reversal_type"), list(EXPECTED))
def test_the_title_names_the_analysis_the_type_and_the_labelling(scheme, reversal_type):
    heading, labelling, _ = EXPECTED[(scheme, reversal_type)]
    ax, _ = plot_reversal_aligned_arms(arms(scheme, reversal_type), scope="pooled")
    lines = ax.get_title().split("\n")
    assert lines[0].startswith(heading)
    assert lines[1] == labelling.capitalize()
    n_block_changes = 4 if reversal_type == "all" else 2
    assert f"n=1 subjects, n={n_block_changes} block changes" in lines[2]


def test_the_pipeline_names_each_figure_after_its_title_and_writes_no_csv(tmp_path, capture):
    result = run_reversal_aligned_arm_analysis(
        {1: data()}, tmp_path, pre=4, post=6, moving_avg_window=1
    )
    expected = sorted(f"{stem}.png" for (_, _, stem) in EXPECTED.values())
    assert sorted(n for n in capture.saved_names() if n.endswith(".png")) == expected
    assert not list(tmp_path.glob("*.csv"))
    assert len(result["counts"]) == 6
    assert (result["counts"]["n_reversals"] > 0).all()


def test_counts_name_the_figure_of_each_row(tmp_path, capture):
    result = run_reversal_aligned_arm_analysis(
        {1: data()}, tmp_path, pre=4, post=6, moving_avg_window=1
    )
    row = result["counts"].query("label_scheme == 'new' and reversal_type == 'third_to_best'")
    assert row.iloc[0]["figure"] == f"{EXPECTED[('new', 'third_to_best')][2]}.png"
