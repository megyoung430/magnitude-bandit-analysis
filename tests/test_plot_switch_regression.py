"""Switch-regression figures: file names, short titles with the statistics line, captions."""

import re

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.behavior_analysis import switch_regression as sr  # noqa: E402
from src.behavior_visualization import plot_switch_regression as plots  # noqa: E402
from src.pipelines.switch_regression import SourceSwitchRegression  # noqa: E402
from tests.test_plot_choice_logit import check_files  # noqa: E402
from tests.test_plot_regression_extensions import saved_figures  # noqa: E402, F401

STATS_LINE = re.compile(r"mean ± se across mice, n=\d+ mice$")
PER_MOUSE_LINE = re.compile(r"per mouse, n=\d+ mice$")
FEATURES = ("choice_lag2", "choice_lag3", "trace", "recency", "never_chosen")


def fake_source(n, seed=0, values=sr.VALUES):
    rng = np.random.default_rng(seed)
    mice = [f"m{i}" for i in range(1, n + 1)]
    rows = [
        dict(mouse=m, condition=c, value=v, beta=rng.normal(0.3, 0.1), se=0.05,
             nats_history=0.6, nats_value=0.59, gain=rng.normal(0.01, 0.005))
        for m in mice for c in sr.CONDITIONS for v in values
    ]  # fmt: skip
    history = [
        dict(mouse=m, condition=c, feature=f, weight=rng.normal(0.5, 0.2), se=0.1)
        for m in mice for c in sr.CONDITIONS for f in FEATURES
    ]  # fmt: skip
    counts = [
        dict(mouse=m, condition=c, n_switches=300 + i, n_problems=4)
        for i, m in enumerate(mice) for c in sr.CONDITIONS
    ]  # fmt: skip
    return SourceSwitchRegression(
        values=pd.DataFrame(rows),
        history=pd.DataFrame(history),
        counts=pd.DataFrame(counts),
        decays=pd.DataFrame({"mouse": mice, "trace_decay": 0.9}),
        decay_table=pd.DataFrame(),
        skipped={},
        n_lags=3,
        l2=1.0,
        trace_grid=(0.5, 0.9),
        recency_cap=100,
        min_switches=100,
    )


def _panels(seen, name):
    suptitle, panels, legend, over = seen[name]
    assert "\n" not in suptitle
    assert over == []  # legends sit outside the data
    return suptitle, panels, legend


def test_value_weight_figure_has_a_panel_per_value_and_condition(tmp_path, saved_figures):  # noqa: F811
    sources = {"Mice": fake_source(6), "WSLS": fake_source(6, 1)}
    base = tmp_path / "Switch Regression Value Weights"
    plots.plot_value_weights(sources, base)
    check_files(base)
    suptitle, panels, legend = _panels(saved_figures, base.name)
    assert suptitle == "Switch Regression Value Weights"
    assert [p.split("\n")[0] for p in panels] == [
        "True Value, After Zero", "True Value, After Non-Zero",
        "Believed Value, After Zero", "Believed Value, After Non-Zero",
        "Model Value, After Zero", "Model Value, After Non-Zero",
    ]  # fmt: skip
    assert all(STATS_LINE.search(p.split("\n")[-1]) for p in panels), panels
    assert "m1" in legend
    caption = base.with_name(f"{base.name}.caption.txt").read_text()
    assert "consistency check" in caption and "not an independent test" in caption
    assert "leave-one-problem-out" in caption.lower()
    assert "no intercept" in caption.lower()
    assert "m1: 300 after zero, 300 after non-zero" in caption
    assert "Wilcoxon" in caption


def test_value_weight_figure_with_few_mice_is_per_mouse(tmp_path, saved_figures):  # noqa: F811
    base = tmp_path / "Switch Regression Value Weights"
    plots.plot_value_weights({"Mice": fake_source(5)}, base)
    _, panels, _ = _panels(saved_figures, base.name)
    assert all(PER_MOUSE_LINE.search(p.split("\n")[-1]) for p in panels), panels
    assert not any("±" in p for p in panels)
    caption = base.with_name(f"{base.name}.caption.txt").read_text()
    assert "fewer than 6 mice" in caption


def test_model_comparison_figure_shows_the_gain_of_each_value(tmp_path, saved_figures):  # noqa: F811
    sources = {"Mice": fake_source(6), "Q-Learner": fake_source(6, 2)}
    base = tmp_path / "Switch Regression Model Comparison"
    plots.plot_model_comparison(sources, base)
    check_files(base)
    suptitle, panels, _ = _panels(saved_figures, base.name)
    assert suptitle == "Switch Regression Model Comparison"
    assert len(panels) == 6
    caption = base.with_name(f"{base.name}.caption.txt").read_text()
    assert "history-only" in caption and "nats per switch" in caption.lower()


def test_history_weight_figure_has_a_panel_per_condition_and_source(tmp_path, saved_figures):  # noqa: F811
    sources = {"Mice": fake_source(4), "WSLS": fake_source(6, 1)}
    base = tmp_path / "Switch Regression History Weights"
    plots.plot_history_weights(sources, base)
    check_files(base)
    suptitle, panels, _ = _panels(saved_figures, base.name)
    assert suptitle == "Switch Regression History Weights"
    assert [p.split("\n")[0] for p in panels] == [
        "Mice, After Zero", "WSLS, After Zero", "Mice, After Non-Zero", "WSLS, After Non-Zero",
    ]  # fmt: skip
    last = [p.split("\n")[-1] for p in panels]
    assert PER_MOUSE_LINE.search(last[0]) and STATS_LINE.search(last[1]), last
    caption = base.with_name(f"{base.name}.caption.txt").read_text()
    assert "trace decay" in caption.lower() and "Mice: 0.9 (4 mice)" in caption


def test_a_source_without_model_values_leaves_its_column_empty(tmp_path, saved_figures):  # noqa: F811
    sources = {"Mice": fake_source(6), "WSLS": fake_source(6, 1, values=("true", "believed"))}
    plots.plot_value_weights(sources, tmp_path / "v")
    _, panels, _ = _panels(saved_figures, "v")
    assert len(panels) == 6


def test_every_figure_needs_sources(tmp_path):
    for draw in (plots.plot_value_weights, plots.plot_model_comparison, plots.plot_history_weights):
        with pytest.raises(ValueError, match="no sources"):
            draw({}, tmp_path / "x")


def test_plot_switch_regression_writes_every_figure(tmp_path):
    out = tmp_path / "switch-regression"
    plots.plot_switch_regression({"Mice": fake_source(6), "WSLS": fake_source(6, 1)}, out)
    for name in plots.FIGURE_NAMES:
        for suffix in ("pdf", "png", "svg", "caption.txt"):
            assert (out / f"{name}.{suffix}").exists(), (name, suffix)
