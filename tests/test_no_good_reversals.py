"""A mouse (or a whole problem) without good reversals must not stop a cohort run.

The real analysis functions run on small synthetic windows; only the window builder, the index
lookup and the plot calls are replaced. A mouse without good reversals contributes nothing to
the figures and is named in a warning; if no mouse has any, the figure is skipped with a warning.
"""

from pathlib import Path

import pytest

import src.pipelines.open_field as pipeline
from src.behavior_analysis.get_choice_probs_around_good_reversals import (
    NoReversalsError,
    get_choice_probs_around_good_reversals,
)
from src.behavior_analysis.get_good_reversal_info import ReversalWarning
from src.pipelines.open_field import (
    ChoiceProbabilitySettings,
    run_choice_probability_analysis,
    run_first_leave_analysis,
)

OUT = Path("out")
PROBLEM = 3


def reversal(idx, pre, post):
    """A good reversal A -> B over towers A, B, C; *pre*/*post* list the chosen tower per trial."""
    towers = ("A", "B", "C")
    return {
        "reversal_idx": idx,
        "reward_magnitudes_by_tower_before": {"A": 3.0, "B": 1.0, "C": 1.0},
        "reward_magnitudes_by_tower_after": {"A": 1.0, "B": 3.0, "C": 1.0},
        "trial_window_idx": {
            "pre": list(range(idx - len(pre), idx)),
            "post": list(range(idx, idx + len(post))),
        },
        "choices_by_tower": {
            t: {"pre": [float(c == t) for c in pre], "post": [float(c == t) for c in post]}
            for t in towers
        },
        "choices_by_rank": {"best": {"pre": [1.0] * len(pre), "post": [1.0] * len(post)}},
    }


def one_reversal():
    return [reversal(10, ["A"] * 10, ["B"] * 30)]


@pytest.fixture
def figures(monkeypatch):
    """Replace the windows and the figure calls; the list collects ``(name, across)`` drawn."""
    drawn = []

    def record(x, across, *, save_path, **kwargs):
        drawn.append((Path(save_path).name, across))

    monkeypatch.setattr(
        pipeline,
        "get_reversal_indices_by_kind",
        lambda trials: dict(good={}, bad={}, unclassified={}),
    )
    monkeypatch.setattr(pipeline, "plot_choice_probs_around_good_reversals", record)
    monkeypatch.setattr(
        pipeline,
        "plot_first_leave_after_good_reversals",
        lambda mean, se, per_subject, p, *, save_path: drawn.append((Path(save_path).name, mean)),
    )
    return drawn


def use_windows(monkeypatch, windows):
    monkeypatch.setattr(pipeline, "get_good_reversal_info", lambda trials, **kw: windows)


def settings(**kwargs):
    return ChoiceProbabilitySettings(
        **{"split_by_first_two": True, "split_by_half": True, **kwargs}
    )


# ---------------------------------------------------------------- analysis layer


def test_choice_probs_without_any_reversal_raises_the_specific_error():
    with pytest.raises(NoReversalsError, match="no subject has a good reversal"):
        get_choice_probs_around_good_reversals({"m1": [], "m2": []}, pre=3, post=5)


def test_the_specific_error_is_a_value_error():
    assert issubclass(NoReversalsError, ValueError)


def test_the_pipeline_skips_only_that_error(monkeypatch, figures):
    """Another ValueError of the analysis (a bad skip) still stops the run."""
    use_windows(monkeypatch, {"m1": one_reversal()})
    bad_skip = settings(
        skip=True,
        run_all=False,
        moving_avg=False,
        remove_bad=False,
        skip_n_trials_after_reversal=99,
    )
    with pytest.raises(ValueError, match="skip") as raised:
        run_choice_probability_analysis("trials", OUT, bad_skip, problem_number=PROBLEM)
    assert not isinstance(raised.value, NoReversalsError)


# ---------------------------------------------------------------- choice probabilities


def test_a_mouse_without_reversals_is_named_and_the_others_are_drawn(monkeypatch, figures):
    use_windows(monkeypatch, {"m1": one_reversal(), "m2": []})
    with pytest.warns(ReversalWarning, match=rf"m2.*no good reversals.*problem {PROBLEM}"):
        run_choice_probability_analysis(
            "trials",
            OUT,
            settings(skip=False, moving_avg=False, remove_bad=False),
            problem_number=PROBLEM,
        )
    full = [across for name, across in figures if name.endswith("Reversals")]
    assert len(full) == 1 and full[0]["num_subjects"] == 1


def test_no_mouse_with_reversals_skips_the_figures_with_a_warning(monkeypatch, figures):
    use_windows(monkeypatch, {"m1": [], "m2": []})
    with pytest.warns(ReversalWarning) as caught:
        run_choice_probability_analysis("trials", OUT, settings(), problem_number=PROBLEM)
    assert figures == []
    texts = [str(w.message) for w in caught]
    assert any("no mouse has good reversals" in t and f"problem {PROBLEM}" in t for t in texts)
    assert any("m1" in t for t in texts) and any("m2" in t for t in texts)


def test_a_split_that_is_empty_is_skipped_but_the_full_figure_is_kept(monkeypatch, figures):
    # one reversal per mouse: the first-two split has data, the later split has none
    use_windows(monkeypatch, {"m1": one_reversal(), "m2": one_reversal()})
    with pytest.warns(ReversalWarning, match="no mouse has good reversals"):
        run_choice_probability_analysis(
            "trials",
            OUT,
            settings(skip=False, moving_avg=False, remove_bad=False),
            problem_number=PROBLEM,
        )
    names = [name for name, _ in figures]
    assert names == [
        "Choice Probabilities Around Good Reversals",
        "Choice Probabilities Around Good Reversals (Early)",
    ]


@pytest.mark.parametrize(
    "options",
    [
        dict(skip=True, run_all=False, moving_avg=False, remove_bad=False),
        dict(moving_avg=True, run_all=False, skip=False, remove_bad=False),
        dict(remove_bad=True, run_all=False, skip=False, moving_avg=False),
        dict(
            split_by_best_change=True,
            split_by_best_change_and_half=True,
            moving_avg=True,
            run_all=False,
            skip=False,
            remove_bad=True,
        ),
    ],
)
def test_every_figure_family_survives_a_problem_without_reversals(monkeypatch, figures, options):
    use_windows(monkeypatch, {"m1": []})
    with pytest.warns(ReversalWarning, match="no mouse has good reversals"):
        run_choice_probability_analysis("trials", OUT, settings(**options), problem_number=PROBLEM)
    assert figures == []


# ---------------------------------------------------------------- first leave


def test_first_leave_names_the_mouse_without_reversals(monkeypatch, figures):
    use_windows(monkeypatch, {"m1": one_reversal(), "m2": []})
    with pytest.warns(ReversalWarning, match=rf"m2.*no good reversals.*problem {PROBLEM}"):
        run_first_leave_analysis("trials", OUT, problem_number=PROBLEM)
    assert [name for name, _ in figures][0] == "First Leave After Good Reversals"


def test_first_leave_skips_when_no_mouse_has_reversals(monkeypatch, figures):
    use_windows(monkeypatch, {"m1": [], "m2": []})
    with pytest.warns(ReversalWarning, match="no mouse has good reversals"):
        run_first_leave_analysis("trials", OUT, problem_number=PROBLEM)
    assert figures == []
