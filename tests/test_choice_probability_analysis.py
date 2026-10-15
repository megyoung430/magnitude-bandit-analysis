"""run_choice_probability_analysis: which figures are drawn, from which windows, in what order.

The analysis steps are replaced by fakes that return tagged tuples, so each plotted
figure carries the full provenance of the data it was computed from.
"""

from pathlib import Path

import pytest

import src.pipelines.open_field as pipeline
from src.pipelines.open_field import (
    ChoiceProbabilitySettings,
    run_choice_probability_analysis,
)

OUT = Path("out")
NAME = "Choice Probabilities Around Good Reversals"
BEST = "Choice Probs Good Reversals"
IDX = ("good_idx", "bad_idx")
UNCLASSIFIED = "unclassified_idx"


class Recorder:
    def __init__(self, monkeypatch):
        self.plots, self.calls = [], []
        fakes = dict(
            get_good_reversal_info=self.windows,
            get_reversal_indices_by_kind=self.indices,
            split_good_reversals_early_late=self.early_late,
            split_good_reversals_by_best_change=self.best_change,
            remove_trials_after_bad_rev=self.remove_bad,
            get_choice_probs_around_good_reversals=self.choice_probs,
            apply_moving_average_to_choice_probs=self.moving_average,
            plot_choice_probs_around_good_reversals=self.plot,
        )
        for name, fake in fakes.items():
            monkeypatch.setattr(pipeline, name, fake)

    def windows(self, trials, pre, post, include_first_block):
        self.calls.append("windows")
        assert include_first_block is False
        return ("windows", trials, pre, post)

    def indices(self, trials):
        self.calls.append("indices")
        return {"good": IDX[0], "bad": IDX[1], "unclassified": UNCLASSIFIED}

    def early_late(self, windows, first_n=None):
        return ("early", windows, first_n), ("late", windows, first_n)

    def best_change(self, windows):
        return ("best2", windows), ("best3", windows)

    def remove_bad(self, windows, good_idx, bad_idx, include_bad_trial):
        assert (good_idx, bad_idx, include_bad_trial) == (*IDX, True)
        return ("no_bad", windows)

    def choice_probs(self, windows, pre, post, **extra):
        return "x", "per_subject", ("across", windows, pre, post, tuple(sorted(extra.items())))

    def moving_average(self, x, per_subject, moving_avg_window, mode):
        assert (x, per_subject, mode) == ("x", "per_subject", "centered")
        return "x_smooth", "ps_smooth", ("smoothed", moving_avg_window)

    def plot(self, x, across, **kwargs):
        self.plots.append((x, across, kwargs))


@pytest.fixture
def rec(monkeypatch):
    return Recorder(monkeypatch)


def settings(**kwargs):
    none = dict(
        run_all=False,
        skip=False,
        moving_avg=False,
        remove_bad=False,
        split_by_best_change=False,
        split_by_best_change_and_half=False,
    )
    return ChoiceProbabilitySettings(**{**none, **kwargs})


def names(rec):
    return [kw.get("save_path", kw).name for _, _, kw in rec.plots]


def test_nothing_enabled_draws_nothing(rec):
    run_choice_probability_analysis("trials", OUT, settings())
    assert rec.plots == [] and rec.calls == []


def test_defaults_draw_all_four_families_in_order(rec):
    run_choice_probability_analysis("trials", OUT, ChoiceProbabilitySettings())
    assert names(rec) == (
        [f"{NAME}{s}" for s in ("", " (Early)", " (Late)")]
        + [f"{NAME}, Skipping Initial Trials{s}" for s in ("", " (Early)", " (Late)")]
        + [
            f"{NAME} (Moving Average){s}"
            for s in ("", " (First Half)", " (Second Half)", " (Early)", " (Late)")
        ]
        + [f"{NAME} (Remove Bad){s}" for s in ("", " (Early)", " (Late)")]
    )


def test_window_and_index_computation_per_family(rec):
    run_choice_probability_analysis("trials", OUT, ChoiceProbabilitySettings())
    # run_all, skip, moving_avg and remove_bad each recompute windows and indices
    assert rec.calls == ["windows", "indices"] * 4


def test_all_reversal_figures(rec):
    run_choice_probability_analysis("trials", OUT, settings(run_all=True, pre=4, post=9))
    windows = ("windows", "trials", 4, 9)
    sources = [windows, ("early", windows, 2), ("late", windows, 2)]
    assert [kw["windows_for_cumulative_axis"] for _, _, kw in rec.plots] == sources
    for (x, across, kw), source in zip(rec.plots, sources, strict=True):
        assert x == "x" and across == ("across", source, 4, 9, ())
        assert kw == dict(
            add_cumulative_axis=True,
            only_good=True,
            use_total=False,
            windows_for_cumulative_axis=source,
            all_good_idx="good_idx",
            all_bad_idx="bad_idx",
            all_unclassified_idx=UNCLASSIFIED,
            save_path=kw["save_path"],
        )


def test_skip_figures_use_skip_option_and_no_cumulative_axis(rec):
    run_choice_probability_analysis(
        "trials", OUT, settings(skip=True, pre=3, post=8, skip_n_trials_after_reversal=6)
    )
    windows = ("windows", "trials", 3, 8)
    sources = [windows, ("early", windows, 2), ("late", windows, 2)]
    assert rec.calls == ["windows", "indices"]
    for (_, across, kw), source in zip(rec.plots, sources, strict=True):
        assert across == ("across", source, 3, 8, (("skip_n_trials_after_reversal", 6),))
        assert kw == dict(
            add_cumulative_axis=False, skip_n_trials_after_reversal=6, save_path=kw["save_path"]
        )


@pytest.mark.parametrize(
    ("toggles", "suffixes"),
    [
        (dict(), ["", " (First Half)", " (Second Half)", " (Early)", " (Late)"]),
        (dict(split_by_half=False), ["", " (Early)", " (Late)"]),
        (dict(split_by_first_two=False), ["", " (First Half)", " (Second Half)"]),
        (dict(split_by_half=False, split_by_first_two=False), [""]),
    ],
)
def test_moving_average_families(rec, toggles, suffixes):
    run_choice_probability_analysis(
        "trials", OUT, settings(moving_avg=True, moving_avg_window=5, **toggles)
    )
    assert names(rec) == [f"{NAME} (Moving Average){s}" for s in suffixes]
    windows = ("windows", "trials", 10, 30)
    expected = {
        "": windows,
        " (First Half)": ("early", windows, None),
        " (Second Half)": ("late", windows, None),
        " (Early)": ("early", windows, 2),
        " (Late)": ("late", windows, 2),
    }
    for suffix, (x, across, kw) in zip(suffixes, rec.plots, strict=True):
        assert (x, across) == ("x_smooth", ("smoothed", 5))
        assert kw["windows_for_cumulative_axis"] == expected[suffix]
        assert kw["only_good"] is True and kw["use_total"] is False


def test_moving_average_smooths_with_the_configured_window(rec):
    run_choice_probability_analysis("trials", OUT, settings(moving_avg=True, moving_avg_window=7))
    assert {(x, across) for x, across, _ in rec.plots} == {("x_smooth", ("smoothed", 7))}


def test_remove_bad_figures(rec):
    run_choice_probability_analysis("trials", OUT, settings(remove_bad=True))
    windows = ("windows", "trials", 10, 30)
    dropped = ("no_bad", windows)
    sources = [
        dropped,
        ("no_bad", ("early", dropped, 2)),
        ("no_bad", ("late", dropped, 2)),
    ]
    assert names(rec) == [f"{NAME} (Remove Bad){s}" for s in ("", " (Early)", " (Late)")]
    for (_, _, kw), source in zip(rec.plots, sources, strict=True):
        assert kw["windows_for_cumulative_axis"] == source
        assert kw["only_good"] is False and "use_total" not in kw


def test_best_change_needs_moving_average(rec):
    run_choice_probability_analysis("trials", OUT, settings(split_by_best_change=True))
    assert rec.plots == []
    run_choice_probability_analysis(
        "trials", OUT, settings(split_by_best_change=True, remove_bad=True)
    )
    assert names(rec) == [f"{NAME} (Remove Bad){s}" for s in ("", " (Early)", " (Late)")]


def test_best_change_figures(rec):
    run_choice_probability_analysis(
        "trials", OUT, settings(moving_avg=True, split_by_best_change=True)
    )
    windows = ("windows", "trials", 10, 30)
    plotted = [
        (kw["save_path"].name, kw["windows_for_cumulative_axis"], kw["only_good"])
        for _, _, kw in rec.plots[5:]
    ]
    assert plotted == [
        (f"{BEST} (Best->Second)", ("best2", windows), True),
        (f"{BEST} (Best->Third)", ("best3", windows), True),
    ]


def test_best_change_with_halves_and_remove_bad(rec):
    run_choice_probability_analysis(
        "trials",
        OUT,
        settings(
            moving_avg=True,
            remove_bad=True,
            split_by_best_change=True,
            split_by_best_change_and_half=True,
        ),
    )
    windows = ("windows", "trials", 10, 30)
    early, late = ("early", windows, 2), ("late", windows, 2)
    best2, best3 = ("best2", windows), ("best3", windows)
    best = [
        (kw["save_path"].name, kw["windows_for_cumulative_axis"], kw["only_good"])
        for _, _, kw in rec.plots[5 + 3 :]
    ]
    assert best == [
        (f"{BEST} (Best->Second)", best2, True),
        (f"{BEST} (Best->Third)", best3, True),
        (f"{BEST} (Best->Second) (Early)", ("best2", early), True),
        (f"{BEST} (Best->Third) (Early)", ("best3", early), True),
        (f"{BEST} (Best->Second) (Late)", ("best2", late), True),
        (f"{BEST} (Best->Third) (Late)", ("best3", late), True),
        # The full-problem Remove Bad figures keep the unfiltered best2/best3; only the
        # early/late splits have the trials after a bad reversal removed.
        (f"{BEST} (Best->Second) (Remove Bad)", best2, False),
        (f"{BEST} (Best->Third) (Remove Bad)", best3, False),
        (f"{BEST} (Best->Second) (Remove Bad) (Early)", ("no_bad", ("best2", early)), False),
        (f"{BEST} (Best->Third) (Remove Bad) (Early)", ("no_bad", ("best3", early)), False),
        (f"{BEST} (Best->Second) (Remove Bad) (Late)", ("no_bad", ("best2", late)), False),
        (f"{BEST} (Best->Third) (Remove Bad) (Late)", ("no_bad", ("best3", late)), False),
    ]


def test_best_change_remove_bad_without_halves_has_two_figures(rec):
    run_choice_probability_analysis(
        "trials",
        OUT,
        settings(moving_avg=True, remove_bad=True, split_by_best_change=True),
    )
    assert names(rec)[8:] == [
        f"{BEST} (Best->Second)",
        f"{BEST} (Best->Third)",
        f"{BEST} (Best->Second) (Remove Bad)",
        f"{BEST} (Best->Third) (Remove Bad)",
    ]


def test_figures_are_saved_under_the_given_directory(rec):
    run_choice_probability_analysis("trials", Path("some/dir"), ChoiceProbabilitySettings())
    assert {kw["save_path"].parent for _, _, kw in rec.plots} == {Path("some/dir")}


def test_settings_are_immutable_and_strict():
    with pytest.raises(ValueError):
        ChoiceProbabilitySettings(skip_trials=3)
    with pytest.raises(ValueError):
        ChoiceProbabilitySettings().run_all = False
