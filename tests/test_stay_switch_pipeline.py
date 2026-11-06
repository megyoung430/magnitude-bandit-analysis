"""run_stay_switch_analysis and its figures on synthetic sessions: files, titles, captions."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis.stay_switch_psychometrics import (
    stay_switch_curves_by_row,
    switch_target_curves_by_row,
)
from src.behavior_analysis.value_psychometrics import Curve
from src.behavior_visualization.plot_helpers import caption_path
from src.behavior_visualization.plot_stay_switch import (
    plot_stay_heatmap_panel,
    plot_stay_switch_curves,
    plot_stay_switch_panel,
    plot_switch_target_curves,
)
from src.pipelines.stay_switch_psychometrics import run_stay_switch_analysis
from tests.plot_capture import capture_figures


@pytest.fixture(autouse=True)
def close_figures():
    plt.close("all")
    yield
    plt.close("all")


def session(offers, choices):
    ranks = ["best"] * len(choices)
    return {
        "trial_variables": [{"reward_magnitudes": offers, "chosen_rank": ranks, "choice": choices}]
    }


def mouse_sessions(seed):
    """Two sessions of 60 trials with drifting values (including ties) and random choices."""
    rng = np.random.default_rng(seed)
    sessions = {}
    for n in (1, 2):
        values = rng.integers(0, 5, size=(60, 3))
        offers = [dict(zip("ABC", map(int, row), strict=True)) for row in values]
        choices = [str(c) for c in rng.choice(list("ABC"), size=60)]
        sessions[f"ses-{n}_date-2026010{n}"] = session(offers, choices)
    return sessions


def data(mice=("m1", "m2", "m3")):
    return {1: {m: mouse_sessions(i) for i, m in enumerate(mice)}}


def stems(basis, ties):
    base = f"({basis} Value) (Ties {ties})"
    return [
        f"Stay Heatmap {base}",
        f"Stay-Switch Curves {base} (Pool Trials True)",
        f"Stay-Switch Curves {base} (Pool Trials False)",
        f"Switch Target Curves {base} (Pool Trials True)",
        f"Switch Target Curves {base} (Pool Trials False)",
    ]


EXPECTED = [s for b in ("True", "Believed") for t in ("Excluded", "Split") for s in stems(b, t)]


def test_writes_every_figure_in_every_format_and_returns_the_results(tmp_path):
    out = tmp_path / "new" / "out"
    results = run_stay_switch_analysis(data(), out)
    written = {p.name for p in out.iterdir()}
    assert written == {
        f"{stem}.{ext}" for stem in EXPECTED for ext in ("pdf", "png", "svg", "caption.txt")
    }
    assert set(results) == {"true", "believed"}
    for by_tie in results.values():
        assert set(by_tie) == {"exclude", "split"}
        for entry in by_tie.values():
            assert set(entry) == {"pairs", "heatmap", "curves", "switch_target_curves"}
            assert set(entry["switch_target_curves"]) == {True, False}
            for rows in entry["switch_target_curves"].values():
                assert set(rows) == {"chosen_zero", "chosen_nonzero"}
                for curve in rows.values():
                    assert curve is None or isinstance(curve, Curve)
            assert set(entry["curves"]) == {True, False}
            for rows in entry["curves"].values():
                assert set(rows) == {"chosen_zero", "chosen_nonzero"}
    assert plt.get_fignums() == []


def test_exclude_mode_has_no_tied_pairs_and_split_mode_keeps_them(tmp_path):
    results = run_stay_switch_analysis(data(), tmp_path, value_bases=("true",))
    exclude, split = results["true"]["exclude"], results["true"]["split"]
    assert exclude["pairs"].pairs["tie"].isna().all()
    assert split["pairs"].pairs["tie"].notna().any()

    def diffs(entry):
        rows = entry["curves"][True].values()
        return {x for curves in rows if curves for x in curves["stay"]["diff"].x}

    assert 0 not in diffs(exclude)
    assert 0 in diffs(split)
    assert exclude["pairs"].ties == split["pairs"].ties


def test_captions_define_the_pairs_and_name_the_ties(tmp_path):
    run_stay_switch_analysis(data(), tmp_path, value_bases=("believed",), tie_modes=("split",))
    caption = caption_path(tmp_path / "Stay-Switch Curves (Believed Value) (Ties Split) "
                           "(Pool Trials True)").read_text()  # fmt: skip
    assert "the chosen option is the arm chosen on t-1" in caption
    assert "V(alt) the higher value of the two other arms" in caption
    assert "Believed value: what the arm paid the last time it was chosen" in caption
    assert "Pairs across a block or value change are kept" in caption
    assert "Ties split" in caption and "Tied pairs - V(chosen) = V(alt):" in caption
    assert "Pairs pooled within mouse" in caption


def test_panel_titles_are_a_heading_and_the_statistics_line(tmp_path):
    results = run_stay_switch_analysis(
        data(), tmp_path, value_bases=("true",), tie_modes=("split",)
    )
    curves = results["true"]["split"]["curves"][True]["chosen_nonzero"]
    ax = plot_stay_switch_panel(
        {"stay": curves["stay"]["diff"], "switch": curves["switch"]["diff"]},
        "Chose Non-0 Option",
        "Value of Chosen - Best Alternative",
    )
    assert ax.get_title() == "Chose Non-0 Option\nmean ± se across mice, n=3 mice"
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Stay", "Switch"]
    assert ax.get_xlabel() == "Value of Chosen - Best Alternative"
    assert ax.get_ylim() == (0, 1)


def test_the_heatmap_prints_each_cell_with_its_pairs(tmp_path):
    results = run_stay_switch_analysis(
        data(), tmp_path, value_bases=("true",), tie_modes=("split",)
    )
    tables = results["true"]["split"]["heatmap"]
    ax = plot_stay_heatmap_panel(tables)
    assert ax.get_title() == ""  # the figure title and the colour-bar label name it
    assert plot_stay_heatmap_panel(tables, "Mouse m1").get_title() == "Mouse m1"
    texts = [t.get_text() for t in ax.texts]
    assert len(texts) == tables.p_stay.size
    top_left = tables.p_stay.index[-1], tables.p_stay.columns[0]  # highest chosen value on top
    n = tables.n_pairs.loc[top_left]
    p = tables.p_stay.loc[top_left]
    assert texts[0] == ("NA" if np.isnan(p) else f"{p:.2f}") + f"\nn={n:g}"
    assert ax.get_ylabel() == "Value of Chosen" and ax.get_xlabel() == "Value of Best Alternative"


def test_the_saved_heatmap_title_has_the_statistics_line(tmp_path):
    run_stay_switch_analysis(data(), tmp_path, value_bases=("true",), tie_modes=("exclude",))
    caption = caption_path(tmp_path / "Stay Heatmap (True Value) (Ties Excluded)").read_text()
    assert "n=pairs summed over mice" in caption and "Ties excluded" in caption


def test_a_condition_without_pairs_warns_and_draws_nothing(tmp_path):
    one_trial = {1: {"m1": {"ses-1_date-20260101": session([{"A": 1, "B": 2, "C": 3}], ["A"])}}}
    with pytest.warns(UserWarning, match="no trial pairs"):
        results = run_stay_switch_analysis(one_trial, tmp_path, value_bases=("true",))
    assert set(results["true"]["exclude"]) == {"pairs"}
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("option", "value"),
    [("value_bases", ("model",)), ("pooling_modes", (1,)), ("tie_modes", ("average",))],
)
def test_unknown_options_raise(tmp_path, option, value):
    with pytest.raises(ValueError, match=option):
        run_stay_switch_analysis(data(), tmp_path, **{option: value})


def test_no_mice_raises(tmp_path):
    with pytest.raises(ValueError, match="no subjects"):
        run_stay_switch_analysis({1: {}}, tmp_path)


def row_test_pairs():
    """m1, m2: pairs with V(chosen) = 0 and above; m3 only above, so the rows differ in n."""
    rows = [
        ("m1", 0, 4, 2, 0.0, 1.0, 0.0),
        ("m1", 0, 4, 2, 1.0, 0.0, 0.0),
        ("m1", 4, 2, 0, 1.0, 0.0, 0.0),
        ("m1", 3, 4, 0, 0.0, 0.0, 1.0),
        ("m2", 0, 2, 1, 0.0, 0.0, 1.0),
        ("m2", 3, 2, 0, 0.0, 1.0, 0.0),
        ("m2", 4, 3, 2, 1.0, 0.0, 0.0),
        ("m3", 2, 4, 4, 0.0, 0.5, 0.5),
        ("m3", 4, 3, 2, 1.0, 0.0, 0.0),
    ]
    columns = ["mouse", "v_chosen", "v_alt", "v_other", "stay", "switch_best_alt", "switch_other"]
    return pd.DataFrame(rows, columns=columns)


NO_TIES = {"chosen_alternative": 0, "alternatives": 0, "all_three": 0}


def titled_axes(fig):
    return [ax for ax in fig.axes if ax.get_title()]


def test_the_stay_switch_figure_has_a_row_per_chosen_value_with_its_own_n(monkeypatch):
    capture = capture_figures(monkeypatch)
    curves = stay_switch_curves_by_row(row_test_pairs(), pool_trials=True)
    plot_stay_switch_curves(curves, "True", True, NO_TIES, "split", None)
    (fig,) = capture.shown[0]
    assert fig.get_suptitle() == "Stay-Switch Curves (True Value)"
    axes = titled_axes(fig)
    assert len(axes) == 6
    top, bottom = axes[:3], axes[3:]
    for ax in top:
        first, stats = ax.get_title().split("\n")
        assert first == "Chose 0-Value Option" and "across mice, n=2 mice" in stats
    for ax in bottom:
        first, stats = ax.get_title().split("\n")
        assert first == "Chose Non-0 Option" and "across mice, n=3 mice" in stats
    assert [ax.get_xlabel() for ax in top] == [
        "Value of Chosen - Best Alternative",
        "Value of Chosen",
        "Value of Best Alternative",
    ]
    assert list(top[1].get_xticks()) == [0]  # the V(chosen) panel of the 0 row: one point
    lo, hi = top[1].get_xlim()
    assert lo < 0 < hi and hi - lo >= 1
    drawn = {line.get_label() for ax in fig.axes for line in ax.lines}
    assert {"Stay", "Switch"} <= drawn


def test_the_single_point_of_the_zero_row_has_an_error_bar(monkeypatch):
    capture = capture_figures(monkeypatch)
    curves = stay_switch_curves_by_row(row_test_pairs(), pool_trials=True)
    plot_stay_switch_curves(curves, "True", True, NO_TIES, "split", None)
    (fig,) = capture.shown[0]
    chosen_panel = titled_axes(fig)[1]
    assert chosen_panel.containers  # the error bars
    assert "mean ± se" in chosen_panel.get_title()


def test_a_row_without_pairs_draws_empty_panels(monkeypatch):
    capture = capture_figures(monkeypatch)
    pairs = row_test_pairs()
    curves = stay_switch_curves_by_row(pairs[pairs["v_chosen"] > 0], pool_trials=True)
    plot_stay_switch_curves(curves, "True", True, NO_TIES, "split", None)
    (fig,) = capture.shown[0]
    top = titled_axes(fig)[:3]
    assert all(ax.get_title() == "Chose 0-Value Option" for ax in top)
    assert all(t.get_text() == "No pairs" for ax in top for t in ax.texts)


def test_switch_target_figure_has_one_panel_per_row_against_the_alternatives_gap(monkeypatch):
    capture = capture_figures(monkeypatch)
    curves = switch_target_curves_by_row(row_test_pairs(), pool_trials=True)
    plot_switch_target_curves(curves, "True", True, NO_TIES, "split", None)
    (fig,) = capture.shown[0]
    assert fig.get_suptitle() == "Switch Target Curves (True Value)"
    zero, nonzero = titled_axes(fig)
    # row_test_pairs: after a 0, m1 and m2 switch once each (gaps 2 and 1); after a non-0, m1
    # and m2 switch once each (gaps 4 and 2) and m3's switch between tied alternatives is out
    # each gap holds one mouse's switch, so there is no se to draw
    assert zero.get_title() == "Chose 0-Value Option\nmean across mice, n=2 mice, n=2 switches"
    assert nonzero.get_title() == "Chose Non-0 Option\nmean across mice, n=2 mice, n=2 switches"
    for ax in (zero, nonzero):
        assert ax.get_xlabel() == "Value of Best Alternative - Other Arm"
        assert ax.get_ylabel() == "P(Best Alternative | Switch)"
        assert ax.get_ylim() == (0.0, 1.0)
        assert any(list(line.get_ydata()) == [0.5, 0.5] for line in ax.lines)  # chance
    drawn = {line.get_label() for ax in fig.axes for line in ax.lines}
    assert "Switch to Best Alternative" in drawn and "Switch to Other Arm" not in drawn


def test_a_row_without_switches_draws_an_empty_switch_target_panel(monkeypatch):
    capture = capture_figures(monkeypatch)
    pairs = row_test_pairs()
    curves = switch_target_curves_by_row(pairs[pairs["v_chosen"] > 0], pool_trials=True)
    plot_switch_target_curves(curves, "True", True, NO_TIES, "split", None)
    (fig,) = capture.shown[0]
    zero = titled_axes(fig)[0]
    assert [t.get_text() for t in zero.texts] == ["No switches"]


def test_switch_target_caption_defines_the_conditional_and_the_gap(tmp_path):
    results = run_stay_switch_analysis(
        data(), tmp_path, value_bases=("believed",), tie_modes=("split",)
    )
    targets = results["believed"]["split"]["switch_target_curves"][True]
    assert set(targets) == {"chosen_zero", "chosen_nonzero"}
    caption = caption_path(
        tmp_path / "Switch Target Curves (Believed Value) (Ties Split) (Pool Trials True)"
    ).read_text()
    assert "Only the pairs whose next choice is a switch" in caption
    assert "V(best alternative) - V(other arm)" in caption
    assert "gap 0" in caption
    assert "top, V(chosen) = 0" in caption
    assert "Believed 0: the arm last paid 0" in caption
