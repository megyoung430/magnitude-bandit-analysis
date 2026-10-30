"""run_stay_switch_analysis and its figures on synthetic sessions: files, titles, captions."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.behavior_visualization.plot_helpers import caption_path
from src.behavior_visualization.plot_stay_switch import (
    plot_stay_heatmap_panel,
    plot_stay_switch_panel,
)
from src.pipelines.stay_switch_psychometrics import run_stay_switch_analysis


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
            assert set(entry) == {"pairs", "heatmap", "curves"}
            assert set(entry["curves"]) == {True, False}
    assert plt.get_fignums() == []


def test_exclude_mode_has_no_tied_pairs_and_split_mode_keeps_them(tmp_path):
    results = run_stay_switch_analysis(data(), tmp_path, value_bases=("true",))
    exclude, split = results["true"]["exclude"], results["true"]["split"]
    assert exclude["pairs"].pairs["tie"].isna().all()
    assert split["pairs"].pairs["tie"].notna().any()
    assert 0 not in exclude["curves"][True]["stay"]["diff"].x
    assert 0 in split["curves"][True]["stay"]["diff"].x
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
    curves = results["true"]["split"]["curves"][True]
    ax = plot_stay_switch_panel(
        {"stay": curves["stay"]["diff"], "switch": curves["switch"]["diff"]},
        "Value of Chosen - Best Alternative",
    )
    assert ax.get_title() == ("Value of Chosen - Best Alternative\nmean ± se across mice, n=3 mice")
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Stay", "Switch"]
    assert ax.get_xlabel() == "Value of Chosen - Best Alternative"
    assert ax.get_ylim() == (0, 1)


def test_the_heatmap_prints_each_cell_with_its_pairs(tmp_path):
    results = run_stay_switch_analysis(
        data(), tmp_path, value_bases=("true",), tie_modes=("split",)
    )
    tables = results["true"]["split"]["heatmap"]
    ax = plot_stay_heatmap_panel(tables, "P(stay) - True Value")
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
