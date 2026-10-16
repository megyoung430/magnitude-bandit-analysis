"""run_value_psychometric_analysis on synthetic sessions: files, tables, problems, tie modes."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
    tied = {"A": 3, "B": 3, "C": 1}
    offers = [offer] * 8 + [tied] * 2
    choices = ["A", "B", "C", "A", "B", "A", "A", "B", "A", "B"]
    ranks = ["best", "second", "third", "best", "second", "best", "best", "second", "best", "best"]
    return {"ses-1": session(offers, choices, ranks)}


def data(problems=(1,), mice=("m1", "m2")):
    return {p: {m: mouse_sessions() for m in mice} for p in problems}


def stems(value_basis, tie_word):
    base = f"({value_basis} Value) (Ties {tie_word})"
    return [
        f"Heatmaps {base} Overall",
        f"Heatmaps {base} Per Mouse",
        f"Psychometric Curves {base} (Pool Trials True)",
        f"Psychometric Curves {base} (Pool Trials False)",
    ]


EXPECTED_STEMS = [
    stem
    for basis in ("True", "Believed")
    for word in ("Excluded", "Split")
    for stem in stems(basis, word)
]


def test_writes_every_figure_as_png_and_pdf_and_returns_the_tables(tmp_path):
    results = run_value_psychometric_analysis(data(), tmp_path / "new" / "out")
    for stem in EXPECTED_STEMS:
        for ext in ("png", "pdf"):
            assert (tmp_path / "new" / "out" / f"{stem}.{ext}").exists(), stem
    assert set(results) == {"true", "believed"}
    for by_tie_mode in results.values():
        assert set(by_tie_mode) == {"exclude", "split"}
        for entry in by_tie_mode.values():
            assert set(entry) == {"counts", "heatmaps", "psychometric"}
            assert set(entry["psychometric"]) == {True, False}
    assert plt.get_fignums() == []


def test_true_value_tables_match_hand_counts_in_both_tie_modes(tmp_path):
    results = run_value_psychometric_analysis(data(), tmp_path, value_bases=("true",))
    split = results["true"]["split"]["heatmaps"]
    exclude = results["true"]["exclude"]["heatmaps"]
    # per mouse in cell (4, 3): 8 trials, 4 best, 3 second, 1 third
    assert split.p_best.loc[4, 3] == pytest.approx(4 / 8)
    assert split.p_second.loc[4, 3] == pytest.approx(3 / 8)
    assert split.n_trials.loc[4, 3] == 16 and exclude.n_trials.loc[4, 3] == 16
    # per mouse in the tied cell (3, 3): two trials, each choosing one tied arm
    assert split.n_trials.loc[3, 3] == 4 and exclude.n_trials.loc[3, 3] == 0
    assert split.p_best.loc[3, 3] == pytest.approx(0.5)
    assert split.ties == exclude.ties == {"best_second": 4, "second_third": 0, "all_three": 0}


def test_selected_bases_modes_and_pooling_only(tmp_path):
    results = run_value_psychometric_analysis(
        data(),
        tmp_path,
        value_bases=("believed",),
        pooling_modes=(False,),
        tie_modes=("exclude",),
    )
    assert set(results["believed"]) == {"exclude"}
    assert set(results["believed"]["exclude"]["psychometric"]) == {False}
    names = sorted(p.name for p in tmp_path.glob("*.png"))
    assert names == [
        "Heatmaps (Believed Value) (Ties Excluded) Overall.png",
        "Heatmaps (Believed Value) (Ties Excluded) Per Mouse.png",
        "Psychometric Curves (Believed Value) (Ties Excluded) (Pool Trials False).png",
    ]


def test_several_problems_pool_per_mouse_but_believed_values_reset_per_problem(tmp_path):
    kwargs = {"pooling_modes": (True,), "tie_modes": ("split",)}
    one = run_value_psychometric_analysis(data((1,)), tmp_path / "a", **kwargs)
    two = run_value_psychometric_analysis(data((1, 2)), tmp_path / "b", **kwargs)
    for basis in ("true", "believed"):
        n_one = one[basis]["split"]["heatmaps"].n_trials.to_numpy().sum()
        n_two = two[basis]["split"]["heatmaps"].n_trials.to_numpy().sum()
        assert n_two == 2 * n_one  # each problem warms up on its own, so counts exactly double
    assert one["believed"]["split"]["heatmaps"].n_trials.to_numpy().sum() < 20


def test_out_of_grid_counts_are_returned_and_warned(tmp_path, caplog):
    off_grid = {"A": 1, "B": 1, "C": 0}
    sessions = {"ses-1": session([off_grid] * 2, ["A", "B"], ["best", "second"])}
    with caplog.at_level("WARNING"):
        results = run_value_psychometric_analysis(
            {1: {"m1": sessions, "m2": mouse_sessions()}},
            tmp_path,
            value_bases=("true",),
            pooling_modes=(True,),
            tie_modes=("split",),
        )
    assert results["true"]["split"]["counts"]["m1"].out_of_grid == {(1, 1): 2}
    warned = [r for r in caplog.records if r.name.endswith("get_choice_probs_by_value")]
    assert len(warned) == 1 and "m1" in warned[0].getMessage()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"value_bases": ("belived",)},
        {"pooling_modes": ("yes",)},
        {"pooling_modes": (1,)},
        {"tie_modes": ("drop",)},
        {"tie_modes": "split"},
    ],
)
def test_unknown_options_are_rejected_before_anything_is_written(tmp_path, kwargs):
    with pytest.raises(ValueError, match="must only contain"):
        run_value_psychometric_analysis(data(), tmp_path / "x", **kwargs)
    assert not (tmp_path / "x").exists()


def test_numpy_bools_are_valid_pooling_modes(tmp_path):
    results = run_value_psychometric_analysis(
        data(), tmp_path, value_bases=("true",), pooling_modes=(np.True_,), tie_modes=("split",)
    )
    assert (
        tmp_path / "Psychometric Curves (True Value) (Ties Split) (Pool Trials True).png"
    ).exists()
    assert len(results["true"]["split"]["psychometric"]) == 1


def test_out_of_grid_warning_is_logged_once_per_mouse_and_value_basis(tmp_path, caplog):
    off_grid = {"A": 1, "B": 1, "C": 0}
    choices = ["A", "B", "C", "A", "B", "A"]
    sessions = {"ses-1": session([off_grid] * 6, choices, ["best"] * 6)}
    with caplog.at_level("WARNING"):
        run_value_psychometric_analysis(
            {1: {"m1": sessions}}, tmp_path, pooling_modes=(True,)
        )  # two value bases, two tie modes
    warned = [r.getMessage() for r in caplog.records if r.name.endswith("by_value")]
    assert len(warned) == 2 and all("m1" in m for m in warned)


def test_no_subjects_raises(tmp_path):
    with pytest.raises(ValueError, match="no subjects"):
        run_value_psychometric_analysis({1: {}}, tmp_path)
