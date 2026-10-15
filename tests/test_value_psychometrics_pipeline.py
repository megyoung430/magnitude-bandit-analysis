"""run_value_psychometric_analysis on synthetic sessions: files, tables, problems."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from src.pipelines.value_psychometrics import run_value_psychometric_analysis


@pytest.fixture(autouse=True)
def close_figures():
    plt.close("all")
    yield
    plt.close("all")


def session(offers, choices, ranks):
    return {
        "trial_variables": [{"reward_magnitudes": offers, "chosen_rank": ranks, "choice": choices}]
    }


def mouse_sessions():
    offer = {"A": 4, "B": 3, "C": 1}
    choices = ["A", "B", "C", "A", "B", "A", "A", "B"]
    ranks = ["best", "second", "third", "best", "second", "best", "best", "second"]
    return {"ses-1": session([offer] * 8, choices, ranks)}


def data(problems=(1,), mice=("m1", "m2")):
    return {p: {m: mouse_sessions() for m in mice} for p in problems}


EXPECTED_STEMS = [
    "Heatmaps (True Value) Overall",
    "Heatmaps (True Value) Per Mouse",
    "Heatmaps (Believed Value) Overall",
    "Heatmaps (Believed Value) Per Mouse",
    "Psychometric Curves (True Value) (Pool Trials True)",
    "Psychometric Curves (True Value) (Pool Trials False)",
    "Psychometric Curves (Believed Value) (Pool Trials True)",
    "Psychometric Curves (Believed Value) (Pool Trials False)",
]


def test_writes_every_figure_as_png_and_pdf_and_returns_the_tables(tmp_path):
    results = run_value_psychometric_analysis(data(), tmp_path / "new" / "out")
    for stem in EXPECTED_STEMS:
        for ext in ("png", "pdf"):
            assert (tmp_path / "new" / "out" / f"{stem}.{ext}").exists(), stem
    assert set(results) == {"true", "believed"}
    for entry in results.values():
        assert set(entry) == {"counts", "heatmaps", "psychometric"}
        assert set(entry["psychometric"]) == {True, False}
    assert plt.get_fignums() == []


def test_true_value_table_matches_hand_counts(tmp_path):
    results = run_value_psychometric_analysis(data(), tmp_path, value_bases=("true",))
    tables = results["true"]["heatmaps"]
    # 8 trials per mouse in cell (4, 3): 4 best, 3 second, 1 third
    assert tables.p_best.loc[4, 3] == pytest.approx(4 / 8)
    assert tables.p_second.loc[4, 3] == pytest.approx(3 / 8)
    assert tables.n_trials.loc[4, 3] == 16


def test_selected_bases_and_modes_only(tmp_path):
    results = run_value_psychometric_analysis(
        data(), tmp_path, value_bases=("believed",), pooling_modes=(False,)
    )
    assert set(results) == {"believed"} and set(results["believed"]["psychometric"]) == {False}
    names = sorted(p.name for p in tmp_path.glob("*.png"))
    assert names == [
        "Heatmaps (Believed Value) Overall.png",
        "Heatmaps (Believed Value) Per Mouse.png",
        "Psychometric Curves (Believed Value) (Pool Trials False).png",
    ]


def test_several_problems_pool_per_mouse_but_believed_values_reset_per_problem(tmp_path):
    one = run_value_psychometric_analysis(data((1,)), tmp_path / "a")
    two = run_value_psychometric_analysis(data((1, 2)), tmp_path / "b")
    for basis in ("true", "believed"):
        n_one = one[basis]["heatmaps"].n_trials.to_numpy().sum()
        n_two = two[basis]["heatmaps"].n_trials.to_numpy().sum()
        assert n_two == 2 * n_one  # each problem warms up on its own, so counts exactly double
    assert one["believed"]["heatmaps"].n_trials.to_numpy().sum() < 16  # warm-up trials excluded


def test_out_of_grid_counts_are_returned_and_warned(tmp_path, caplog):
    off_grid = {"A": 1, "B": 1, "C": 0}
    sessions = {"ses-1": session([off_grid] * 2, ["A", "B"], ["best", "second"])}
    with caplog.at_level("WARNING"):
        results = run_value_psychometric_analysis(
            {1: {"m1": sessions, "m2": mouse_sessions()}}, tmp_path, value_bases=("true",)
        )
    assert results["true"]["counts"]["m1"].out_of_grid == {(1, 1): 2}
    warned = [r for r in caplog.records if r.name.endswith("get_choice_probs_by_value")]
    assert len(warned) == 1 and "m1" in warned[0].getMessage()


def test_unknown_value_basis_raises(tmp_path):
    with pytest.raises(ValueError, match="value_bases"):
        run_value_psychometric_analysis(data(), tmp_path, value_bases=("belived",))


def test_no_subjects_raises(tmp_path):
    with pytest.raises(ValueError, match="no subjects"):
        run_value_psychometric_analysis({1: {}}, tmp_path)


def test_unknown_basis_is_rejected_before_anything_is_written(tmp_path):
    with pytest.raises(ValueError):
        run_value_psychometric_analysis(data(), tmp_path / "x", value_bases=("nope",))
    assert not (tmp_path / "x").exists()
